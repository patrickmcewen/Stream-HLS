"""Differentiable optimization of a design point against diff_timing, with
periodic rounding to a valid point for Vitis HLS fidelity checks.

Two modes:

  optimize  Adam over DesignParams (tile sizes + soft permutations) minimizing
            latency under a DSP-budget penalty (without it the latency model is
            monotone in tile size and the optimum is the degenerate all-tiles-
            maxed point). Every --ckpt-interval steps the continuous params are
            rounded to a valid design point (tiles -> divisors of trip <=
            tiling_limit, soft perms -> bijection), written as a checkpoint JSON
            next to the base point, and re-evaluated. Emits:
              <out>/trajectory.csv  step, lat_cont, dsp           (every step)
              <out>/manifest.csv    step, json, lat_cont, lat_round  (checkpoints)
              <design_dir>/mlir/intermediates/<stem>_opt<step>.json
            Stage 2 (HLS) is then run separately and asynchronously:
              python eval_timing.py <those JSONs> --phase launch --runner sbatch

  reconcile Join a manifest with eval_timing's collect CSV (Stage 3): match by
            JSON stem and write <out>/fidelity.csv with lat_cont, lat_round and
            the HLS actual side by side (round_gap, model_err).

DSP proxy: each tiled loop is fully unrolled, so DSP(node) ~ prod(tile) and
total DSP = sum over nodes. This is monotone in tiles (the counter-pressure to
latency); it ignores op type (no op info in the JSON) and should be validated
against the .rpt DSP count in reconcile.

Usage:
    python optimize_timing.py optimize <base_space.json> [--steps N] [--lr R]
        [--ckpt-interval K] [--board-dsps D] [--tiling-limit L] [--lam W]
    python optimize_timing.py reconcile <out_dir> <eval_collect.csv>
"""

import argparse
import csv
import datetime
import json
import os

import torch

import diff_timing

# This model is thousands of tiny (<=4x4) tensor ops per step; with the default
# intra-op thread pool (one thread per core) the per-op thread launch/sync
# overhead dominates -- ~54ms vs ~5us for a 4x4 softmax on a 128-core host, a
# >10000x slowdown. Pin to a single thread so steps run in milliseconds.
torch.set_num_threads(1)


def _upper_bounds(graph, tiling_limit):
    """Per-node tensor of max tiling factor (min(trip, tiling_limit)) per loop."""
    ub = {}
    for node in graph["nodes"]:
        ub[node["id"]] = torch.tensor(
            [min(float(l["trip_count"]), float(tiling_limit)) for l in node["loops"]])
    return ub


def total_dsp(params):
    """Differentiable DSP proxy: sum over nodes of the product of tile sizes
    (the fully-unrolled lane count)."""
    return sum(diff_timing._prod(list(params.tile[nid])) for nid in params.tile)


def perm_penalty(params):
    """Two penalties guiding every soft permutation toward a permutation matrix,
    returned separately because they play different roles in the loss schedule.

    soft_perm row-softmaxes the logits, so each loop's row already sums to 1 but
    nothing stops several loops from piling onto the same nest position (a
    column). Two terms fix that (Birkhoff: among doubly-stochastic matrices the
    one maximizing sum(P^2) is a permutation matrix):
      col  -- each position used exactly once: sum_j (sum_i P[i,j] - 1)^2. This
              discourages two loops from picking the same dim. It is flat (zero)
              across the whole Birkhoff polytope, so it never blocks movement
              *between* permutations -- it only forbids true collisions and can
              stay on throughout.
      peak -- one-hot rows/cols: n - sum(P^2), zero only at a permutation matrix,
              maximal in the blurry interior. This crystallizes a vertex, but is
              also a barrier between vertices, so it must be annealed in late
              (see peak_weight) or it freezes the ordering in a local minimum."""
    col = peak = 0.0
    for nid in params.perm_logits:
        P = params.soft_perm(nid)
        n = P.shape[0]
        col = col + ((P.sum(dim=0) - 1.0) ** 2).sum()
        peak = peak + (n - (P ** 2).sum())
    return col, peak


def peak_weight(step, steps, start_frac):
    """Annealing schedule for the sharpness (peak) term: 0 while the optimizer
    explores orderings, ramping linearly to 1 over [start_frac, 0.9] * steps so
    the permutation only crystallizes near the end."""
    s0 = start_frac * steps
    s1 = 0.9 * steps
    if step <= s0:
        return 0.0
    if step >= s1:
        return 1.0
    return (step - s0) / (s1 - s0)


def _fmt(vals, fmt):
    return "[" + ", ".join(format(v, fmt) for v in vals) + "]"


def log_verbose_params(vf, step, lat_cont, lat_round, graph, params, rounded):
    """Append the concrete continuous params and their rounded counterparts for
    every node at a checkpoint. Tiles are logged continuous vs divisor-rounded;
    permutations as the continuous expected loop position per loop (0 = outermost)
    vs the rounded bijection actually written to the checkpoint JSON."""
    rnodes = {n["id"]: n for n in rounded["nodes"]}
    vf.write(f"=== step {step} (lat_cont={lat_cont:.1f} lat_round={lat_round:.1f}) ===\n")
    for node in graph["nodes"]:
        nid = node["id"]
        tile_cont = params.tile[nid].detach().tolist()
        tile_round = [l["tiling_factor"] for l in rnodes[nid]["loops"]]
        soft = params.soft_perm(nid).detach()
        perm_cont = [diff_timing._expected_pos(row).item() for row in soft]
        perm_round = rnodes[nid]["permutation"]
        vf.write(f"  node {nid}:\n")
        vf.write(f"    tile cont={_fmt(tile_cont, '.3f')} round={_fmt(tile_round, 'd')}\n")
        vf.write(f"    perm cont={_fmt(perm_cont, '.3f')} round={_fmt(perm_round, 'd')}\n")
        vf.write(f"    soft_perm (loop x position):\n")
        for i, row in enumerate(soft.tolist()):
            vf.write(f"      loop {i}: {_fmt(row, '.3f')}\n")
    vf.write("\n")
    vf.flush()


def optimize(base_json, out_dir, steps, lr, ckpt_interval, board_dsps,
             tiling_limit, lam, lam_perm, perm_sharpen, lam_fuse, fuse_align):
    graph = diff_timing.load_graph(base_json)
    params = diff_timing.DesignParams(graph)
    model = diff_timing.DFGTiming(graph)
    ub = _upper_bounds(graph, tiling_limit)

    lat0 = model.analyze(params)["total_cycles"].item()
    opt = torch.optim.Adam(list(params.parameters()), lr=lr)

    design_dir, _ = os.path.split(os.path.dirname(os.path.dirname(base_json)))
    interm = os.path.dirname(base_json)
    stem = os.path.splitext(os.path.basename(base_json))[0]
    os.makedirs(out_dir, exist_ok=True)

    model_name = os.path.basename(design_dir)
    single_tile = diff_timing.leading_singleton_boundary_nodes(
        graph, f"{design_dir}/mlir/input/{model_name}.mlir")
    if single_tile:
        print(f"  leading-singleton interface nodes capped to 1 tiled loop: "
              f"{sorted(single_tile)}")

    traj = open(f"{out_dir}/trajectory.csv", "w", newline="")
    manifest = open(f"{out_dir}/manifest.csv", "w", newline="")
    verbose = open(f"{out_dir}/params_verbose.log", "w")
    tw = csv.writer(traj)
    tw.writerow(["step", "lat_cont", "dsp", "perm_col", "perm_peak", "peak_w",
                 "fuse_pen"])
    mw = csv.writer(manifest)
    mw.writerow(["step", "json", "lat_cont", "lat_round"])

    def checkpoint(step, lat_cont):
        rounded = diff_timing.round_design(graph, params, tiling_limit,
                                           single_tile_nodes=single_tile,
                                           fuse_align=fuse_align)
        ckpt_json = f"{interm}/{stem}_opt{step}.json"
        with open(ckpt_json, "w") as f:
            json.dump(rounded, f, indent=2)
        lat_round = diff_timing.DFGTiming(rounded).analyze(
            diff_timing.DesignParams(rounded))["total_cycles"].item()
        mw.writerow([step, ckpt_json, f"{lat_cont:.1f}", f"{lat_round:.1f}"])
        manifest.flush()
        log_verbose_params(verbose, step, lat_cont, lat_round, graph, params, rounded)
        print(f"  [ckpt {step}] lat_cont={lat_cont:.0f} lat_round={lat_round:.0f} "
              f"-> {os.path.basename(ckpt_json)}")

    best = (float("inf"), -1)
    for step in range(steps + 1):
        out = model.analyze(params)
        lat = out["total_cycles"]
        dsp = total_dsp(params)
        col_pen, peak_pen = perm_penalty(params)
        peak_w = peak_weight(step, steps, perm_sharpen)
        fuse_pen = model.fusion_penalty(params)
        loss = (lat / lat0 + lam * torch.relu(dsp / board_dsps - 1.0)
                + lam_perm * (col_pen + peak_w * peak_pen)
                + lam_fuse * fuse_pen)

        lat_cont = lat.item()
        tw.writerow([step, f"{lat_cont:.1f}", f"{dsp.item():.1f}",
                     f"{col_pen.item():.4f}", f"{peak_pen.item():.4f}",
                     f"{peak_w:.3f}", f"{fuse_pen.item():.4f}"])
        if lat_cont < best[0]:
            best = (lat_cont, step)
        if step % ckpt_interval == 0:
            checkpoint(step, lat_cont)

        if step == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():  # keep tiles in [1, min(trip, tiling_limit)]
            for nid, t in params.tile.items():
                t.clamp_(min=1.0)
                t.copy_(torch.minimum(t, ub[nid]))

    traj.close(); manifest.close(); verbose.close()
    print(f"best continuous latency {best[0]:.0f} at step {best[1]}")
    print(f"wrote {out_dir}/trajectory.csv, {out_dir}/manifest.csv, "
          f"{out_dir}/params_verbose.log")
    print(f"Stage 2: python eval_timing.py "
          f"{interm}/{stem}_opt*.json --phase launch --runner sbatch")


def reconcile(out_dir, collect_csv):
    """Join manifest.csv with an eval_timing collect CSV (TOTAL rows) by JSON
    stem, writing fidelity.csv."""
    hls = {}
    with open(collect_csv) as f:
        for row in csv.DictReader(f):
            if row["node"] == "TOTAL":
                hls[os.path.splitext(row["design_point"])[0]] = float(row["hls_lat"])

    rows = []
    with open(f"{out_dir}/manifest.csv") as f:
        for m in csv.DictReader(f):
            stem = os.path.splitext(os.path.basename(m["json"]))[0]
            if stem not in hls:
                print(f"  no HLS total for {stem}, skipping")
                continue
            cont, rnd, actual = float(m["lat_cont"]), float(m["lat_round"]), hls[stem]
            rows.append([m["step"], f"{cont:.1f}", f"{rnd:.1f}", f"{actual:.1f}",
                         f"{abs(cont - rnd) / rnd:.4f}", f"{abs(rnd - actual) / actual:.4f}"])

    out = f"{out_dir}/fidelity.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lat_cont", "lat_round", "hls_actual",
                    "round_gap", "model_err"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} checkpoints reconciled)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    o = sub.add_parser("optimize")
    o.add_argument("base_json")
    o.add_argument("--steps", type=int, default=300)
    o.add_argument("--lr", type=float, default=.1)
    o.add_argument("--ckpt-interval", type=int, default=50)
    o.add_argument("--board-dsps", type=float, default=6840)  # xcu200
    o.add_argument("--tiling-limit", type=int, default=16)
    o.add_argument("--lam", type=float, default=1.0, help="DSP-budget penalty weight")
    o.add_argument("--lam-perm", type=float, default=0.5,
                   help="permutation-matrix penalty weight (column collisions + sharpness)")
    o.add_argument("--perm-sharpen", type=float, default=0.9,
                   help="fraction of steps before the sharpness term begins ramping "
                        "in (0..1; reaches full strength by 0.9*steps). Lower = "
                        "commit the ordering earlier, higher = explore longer.")
    o.add_argument("--lam-fuse", type=float, default=0.5,
                   help="fusion-alignment penalty weight: pulls connected nodes onto "
                        "matching shared-dim order and tiles so edges can stream")
    o.add_argument("--no-fuse-align", dest="fuse_align", action="store_false",
                   help="disable joint rounding of shared-dim tiles (round each loop "
                        "independently instead)")
    o.add_argument("--out-dir", default=None)

    r = sub.add_parser("reconcile")
    r.add_argument("out_dir")
    r.add_argument("collect_csv")

    args = ap.parse_args()
    if args.mode == "optimize":
        out_dir = args.out_dir or (
            f"opt_log/{os.path.splitext(os.path.basename(args.base_json))[0]}_"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        optimize(args.base_json, out_dir, args.steps, args.lr, args.ckpt_interval,
                 args.board_dsps, args.tiling_limit, args.lam, args.lam_perm,
                 args.perm_sharpen, args.lam_fuse, args.fuse_align)
    else:
        reconcile(args.out_dir, args.collect_csv)
