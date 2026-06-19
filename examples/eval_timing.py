"""Two-phase evaluation harness: differentiable timing model vs Vitis HLS.

Phase 'launch': for each design-point JSON, emit its HLS C++ (streamhls-opt
apply mode + streamhls-translate, via compare_timing) and submit a Vitis csynth
('syn') SLURM job that writes the *.verbose.rpt reports.

Phase 'collect': once those reports exist, re-run the (deterministic) diff model
on the same JSONs, parse the reports, and write a per-node comparison -- diff
N_out*T_fire, scheduled span (lw-st), st/fw/lw vs the HLS module latency and
achieved II, plus the top-level total (diff total_cycles vs HLS dataflow
latency) -- to a pretty .log and a machine-readable .csv.

Design points are given as solution-JSON paths -- either listed on the command
line or, more conveniently, as a named group in a YAML config (--group). Each
path has the form <design_dir>/mlir/intermediates/<stem>.json, and the
design_dir (and model = its last path component) is derived from the path, so a
group may span multiple benchmarks.

Usage:
    # after `module load vitis/2022.1` to put vitis_hls on PATH:
    python eval_timing.py --group polybench --phase launch --runner local --jobs 4
    python eval_timing.py --group polybench --phase launch --runner sbatch
    python eval_timing.py --group polybench --phase collect
    python eval_timing.py <solution.json> [<solution.json> ...] --phase collect
"""

import argparse
import concurrent.futures as cf
import datetime
import os
import shutil
import subprocess

import yaml

import compare_timing
import parse_hls_rpt

# Vitis module load for the generated SLURM batch script.
MODULE_LOAD = "source /etc/profile.d/modules.sh\nmodule load vitis/2022.1"

SYN_BATCH = """#!/bin/bash
#SBATCH -J eval_{kernel}
#SBATCH -o {kernel}_syn.%J.out
#SBATCH -p batch -c4 --mem=16GB
{module_load}
cd {hls_dir}
export PRJ_PATH=$PWD
vitis_hls hls.tcl {kernel} syn -l {kernel}_syn.log
"""


def _stem(solution_json):
    return os.path.splitext(os.path.basename(solution_json))[0]


def _derive(solution_json):
    """(design_dir, model) for a <design_dir>/mlir/intermediates/<stem>.json path."""
    interm = os.path.dirname(solution_json)
    mlir = os.path.dirname(interm)
    design_dir = os.path.dirname(mlir)
    assert (os.path.basename(interm) == "intermediates"
            and os.path.basename(mlir) == "mlir"), (
        f"unexpected layout for {solution_json!r}; expected "
        "<design_dir>/mlir/intermediates/<stem>.json")
    return design_dir, os.path.basename(design_dir)


def _resolve(args):
    """(list of (solution_json, bufferize), label) from --group or positional paths.

    A YAML entry is either a plain path string (bufferize defaults to 0) or a
    mapping {json: <path>, bufferize: 0|1} for models whose input MLIR keeps
    tensor func args (e.g. MHSA needs bufferize: 1)."""
    if args.group:
        with open(args.config) as f:
            groups = yaml.safe_load(f)
        assert args.group in groups, \
            f"group {args.group!r} not in {args.config}; have {sorted(groups)}"
        entries = groups[args.group]
        assert entries, f"group {args.group!r} is empty"
        label = args.group
    else:
        assert args.solutions, "provide solution JSON paths or --group"
        entries = args.solutions
        label = "adhoc"

    out = []
    for e in entries:
        if isinstance(e, dict):
            assert "json" in e, f"entry {e} missing 'json' key"
            out.append((e["json"], int(e.get("bufferize", 0))))
        else:
            out.append((e, 0))
    return out, label


def _report_db(design_dir, model, stem):
    return f"{design_dir}/hls/hls_{model}_{stem}/solution1/.autopilot/db"


def prepare(design_dir, model, solution_json, bufferize):
    """Emit the design point's HLS C++, testbench, and csynth batch script.
    Run serially -- the emit step writes a shared /tmp model file. Returns
    (kernel, hls_dir, batch_name)."""
    stem = _stem(solution_json)
    kernel = f"{model}_{stem}"
    # Emit HLS C++ for this design point (the analytical latency is unused here).
    compare_timing.streamhls_latency(design_dir, model, solution_json,
                                     emit_hls=True, bufferize=bufferize)

    # csynth registers a testbench via hls.tcl; the kernel interface is identical
    # across design points, so reuse the base testbench under the per-point name.
    base_tb = f"{design_dir}/hls/src/{model}_tb.cpp"
    assert os.path.exists(base_tb), f"missing base testbench {base_tb}"
    tb = f"{design_dir}/hls/src/{kernel}_tb.cpp"
    if not os.path.exists(tb):
        shutil.copy(base_tb, tb)

    hls_dir = f"{design_dir}/hls"
    batch = f"syn_{stem}.batch"
    with open(f"{hls_dir}/{batch}", "w") as f:
        f.write(SYN_BATCH.format(kernel=kernel, module_load=MODULE_LOAD,
                                 hls_dir=os.path.abspath(hls_dir)))
    return kernel, hls_dir, batch


def run_syn_local(kernel, hls_dir):
    """Run csynth in the foreground for one kernel. Returns (kernel, ok, hls_dir).
    Output is captured to <kernel>_syn.log (via vitis_hls -l) so concurrent runs
    do not interleave on the console."""
    proc = subprocess.run(["vitis_hls", "hls.tcl", kernel, "syn",
                           "-l", f"{kernel}_syn.log"], cwd=hls_dir,
                          capture_output=True, text=True)
    return kernel, proc.returncode == 0, hls_dir


def collect(design_dir, model, solution_json, log, csv):
    stem = _stem(solution_json)
    db = _report_db(design_dir, model, stem)
    fwd = f"{db}/forward.verbose.rpt"
    if not os.path.exists(fwd):
        print(f"  SKIP {os.path.basename(solution_json)}: no HLS report ({fwd})")
        return
    hls = parse_hls_rpt.parse_forward(fwd)
    graph, out = compare_timing.diff_analyze(solution_json)
    per_node = out["per_node"]
    per_edge = out["per_edge"]
    total = out["total_cycles"].item()

    name = os.path.basename(solution_json)
    log.write(f"\n=== {name} ===\n")
    log.write(f"{'node':>5}{'Nout*Tfire':>12}{'lw-st':>11}{'st':>10}{'fw':>10}"
              f"{'lw':>10}{'hls_lat':>10}{'hls_II':>7}{'rel.err':>9}\n")
    for nid in sorted(per_node):
        d = per_node[nid]
        diff_lat = (d["N_out"] * d["T_fire"]).item()
        span = (d["lw"] - d["st"]).item()
        st, fw, lw = d["st"].item(), d["fw"].item(), d["lw"].item()
        assert nid in hls["nodes"], f"node {nid} absent from HLS instance table"
        hls_lat = hls["nodes"][nid]
        node_rpt = f"{db}/node{nid}.verbose.rpt"
        ii = parse_hls_rpt.parse_node_ii(node_rpt) if os.path.exists(node_rpt) else None
        rel = abs(diff_lat - hls_lat) / hls_lat
        log.write(f"{nid:>5}{diff_lat:>12.1f}{span:>11.1f}{st:>10.1f}{fw:>10.1f}"
                  f"{lw:>10.1f}{hls_lat:>10}{str(ii):>7}{rel:>8.2%}\n")
        csv.write(f"{name},{nid},{diff_lat:.1f},{span:.1f},{st:.1f},{fw:.1f},"
                  f"{lw:.1f},{hls_lat},{ii},{rel:.4f}\n")

    rel_total = abs(total - hls["total"]) / hls["total"]
    log.write(f"{'TOTAL':>5}{total:>12.1f}{'':>11}{'':>10}{'':>10}{'':>10}"
              f"{hls['total']:>10}{'':>7}{rel_total:>8.2%}\n")
    csv.write(f"{name},TOTAL,{total:.1f},,,,,{hls['total']},,{rel_total:.4f}\n")

    # Per-edge buffer-fill (producer tiles the analytical model says the consumer
    # waits for before firing). edge_<id> rows in the CSV carry the same value.
    log.write(f"{'edge':>5}{'src->dst':>11}{'fill':>10}\n")
    for e in sorted(graph["edges"], key=lambda e: e["id"]):
        fill = per_edge[e["id"]].item()
        arc = f"{e['src']}->{e['dst']}"
        log.write(f"{e['id']:>5}{arc:>11}{fill:>10.2f}\n")
        csv.write(f"{name},edge_{e['id']},{fill:.4f},,,,,,,\n")

    print(f"  {name}: diff total={total:.0f} hls total={hls['total']} rel={rel_total:.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("solutions", nargs="*",
                    help="solution JSON paths (design_dir derived from the path)")
    ap.add_argument("--group", help="use a named group of JSONs from --config")
    ap.add_argument("--config", default="eval_groups.yaml",
                    help="YAML mapping group name -> list of solution JSON paths")
    ap.add_argument("--phase", choices=["launch", "collect"], required=True)
    ap.add_argument("--runner", choices=["sbatch", "local", "none"], default="sbatch",
                    help="launch: how to run csynth -- submit SLURM (sbatch), run "
                         "vitis_hls in the foreground (local), or only prepare "
                         "artifacts (none)")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="launch --runner local: number of concurrent vitis_hls runs")
    ap.add_argument("--log-dir", default="eval_log")
    args = ap.parse_args()

    solutions, label = _resolve(args)

    if args.phase == "launch":
        # Emit artifacts serially (shared /tmp model file), then run csynth. A
        # design point that fails to emit (e.g. streamhls-opt crashes on it) is
        # skipped so it does not abort the rest of the batch.
        prepared, failed = [], []
        for sol, bufferize in solutions:
            print(f"prepare {sol}")
            design_dir, model = _derive(sol)
            try:
                prepared.append(prepare(design_dir, model, sol, bufferize))
            except Exception as e:
                print(f"  SKIP {sol}: {str(e).splitlines()[0]}")
                failed.append(sol)

        if args.runner == "none":
            for kernel, hls_dir, batch in prepared:
                print(f"  prepared {kernel}: (cd {hls_dir} && sbatch {batch})")
        elif args.runner == "sbatch":
            assert shutil.which("sbatch"), \
                "sbatch not on PATH; use --runner local (after module load vitis) or none"
            for kernel, hls_dir, batch in prepared:
                proc = subprocess.run(["sbatch", batch], cwd=hls_dir,
                                       capture_output=True, text=True)
                assert proc.returncode == 0, f"sbatch failed:\n{proc.stderr}"
                print(f"  submitted {kernel}: {proc.stdout.strip()}")
        else:  # local
            assert shutil.which("vitis_hls"), \
                "vitis_hls not on PATH; run 'module load vitis/2022.1' first"
            print(f"  running csynth locally ({args.jobs} at a time, slow)...")
            with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(run_syn_local, k, d) for k, d, _ in prepared]
                for fut in cf.as_completed(futs):
                    kernel, ok, hls_dir = fut.result()
                    tail = "" if ok else f" -- FAILED, see {hls_dir}/{kernel}_syn.log"
                    print(f"  {'done' if ok else 'FAIL'} {kernel}{tail}")

        if failed:
            print(f"\nskipped {len(failed)} design point(s) that failed to emit:")
            for sol in failed:
                print(f"  {sol}")
    else:
        os.makedirs(args.log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"{args.log_dir}/eval_{label}_{ts}.log"
        csv_path = f"{args.log_dir}/eval_{label}_{ts}.csv"
        with open(log_path, "w") as log, open(csv_path, "w") as csv:
            csv.write("design_point,node,diff_Nout_Tfire,diff_lw_st,diff_st,"
                      "diff_fw,diff_lw,hls_lat,hls_II,rel_err\n")
            for sol, _bufferize in solutions:
                design_dir, model = _derive(sol)
                collect(design_dir, model, sol, log, csv)
        print(f"wrote {log_path}\n      {csv_path}")
