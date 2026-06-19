"""Compare the differentiable timing model (diff_timing.py) against the
Stream-HLS analytical model evaluated at the same applied design point.

The Stream-HLS side runs `streamhls-opt` in codesign apply mode on a design
point JSON; the ApplyTransformSolution path emits and evaluates the analytical
st/fw/lw recurrence and prints `Applied Latency: <cycles>` on stderr. The
differentiable side runs diff_timing on the same JSON. Both are sequential
(II=1, no parallelization), so the numbers are directly comparable.

With -v/--verbose the per-node start (st), first-write (fw) and last-write (lw)
times of *both* models are dumped side by side. The Stream-HLS per-node values
are recovered by re-evaluating the generated model script that apply mode leaves
at GENERATED_MODEL (it binds st<id>/fw<id>/lw<id> for every node).

Usage:
    python compare_timing.py <design_dir> <solution.json> [<solution.json> ...] [-v]

  design_dir  directory holding mlir/input/<model>.mlir, mlir/graphs/graph, ...
              e.g. designs/polybench/gemm2/gemm
"""

import argparse
import contextlib
import io
import os
import re
import subprocess

import diff_timing

APPLIED_RE = re.compile(r"Applied Latency:\s*(\d+)")
GENERATED_MODEL = "/tmp/streamhls_apply_combined.py"


def streamhls_latency(design_dir, model, solution_json, tiling_limit=16,
                      emit_hls=False, bufferize=0):
    """Run streamhls-opt apply mode and return the analytical latency. As a side
    effect, leaves the evaluated model script at GENERATED_MODEL. With emit_hls,
    the lowered kernel (streamhls-opt's stdout) is also translated to HLS C++.

    `bufferize` sets bufferize-func-args, which must match how the model's input
    MLIR was produced: 0 for the (already-bufferized) polybench inputs, 1 for
    models whose func args are still tensors (e.g. MHSA) -- the wrong value makes
    streamhls-opt abort."""
    pipeline = (
        f"top-func=forward "
        f"graph-file={design_dir}/mlir/graphs/graph "
        f"report-file={design_dir}/mlir/intermediates/{model} "
        f"mode=apply parallelize-nodes=true tiling-limit={tiling_limit} "
        f"bufferize-func-args={bufferize} "
        f"solution-file={solution_json}"
    )
    cmd = [
        "streamhls-opt",
        f"{design_dir}/mlir/input/{model}.mlir",
        f"-streamhls-codesign-pipeline={pipeline}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    m = APPLIED_RE.search(proc.stderr)
    assert m, f"no 'Applied Latency' in streamhls-opt output:\n{proc.stderr[-2000:]}"
    if emit_hls:
        emit_hls_cpp(design_dir, model, solution_json, proc.stdout)
    return int(m.group(1))


def emit_hls_cpp(design_dir, model, solution_json, kernel_mlir):
    """Translate the applied design point to Vivado HLS C++. `kernel_mlir` is
    streamhls-opt's stdout (the lowered kernel); streamhls-translate writes the
    .cpp alongside the project's other HLS sources, suffixed with the solution
    name so different design points do not clobber one another."""
    stem = os.path.splitext(os.path.basename(solution_json))[0]
    kernel_path = f"{design_dir}/mlir/kernel/{model}_{stem}.mlir"
    cpp_path = f"{design_dir}/hls/src/{model}_{stem}.cpp"
    os.makedirs(os.path.dirname(kernel_path), exist_ok=True)
    os.makedirs(os.path.dirname(cpp_path), exist_ok=True)
    with open(kernel_path, "w") as f:
        f.write(kernel_mlir)
    subprocess.run(
        ["streamhls-translate", kernel_path, "-emit-vivado-hls", "-o", cpp_path],
        check=True,
    )
    print(f"  emitted HLS C++: {cpp_path}")


def streamhls_per_node(graph, script_path=GENERATED_MODEL):
    """Per-node {st, fw, lw} from the Stream-HLS model, by re-evaluating the
    generated script and reading its st<id>/fw<id>/lw<id> bindings. A quantity a
    node does not define (e.g. fw for a graph sink) comes back as None."""
    ns = {}
    with open(script_path) as f, contextlib.redirect_stdout(io.StringIO()):
        exec(compile(f.read(), script_path, "exec"), ns)
    return {
        n["id"]: {k: ns.get(f"{k}{n['id']}") for k in ("st", "fw", "lw")}
        for n in graph["nodes"]
    }


def diff_analyze(solution_json):
    graph = diff_timing.load_graph(solution_json)
    params = diff_timing.DesignParams(graph)
    return graph, diff_timing.DFGTiming(graph).analyze(params)


def _fmt(v):
    return "-" if v is None else f"{float(v):.1f}"


def print_per_node(shls_nodes, diff_nodes):
    for nid in sorted(diff_nodes):
        s, d = shls_nodes[nid], diff_nodes[nid]
        print(f"  node {nid}")
        print(f"    streamhls: st={_fmt(s['st']):>13} "
              f"fw={_fmt(s['fw']):>13} lw={_fmt(s['lw']):>13}")
        print(f"    diff     : st={_fmt(d['st'].item()):>13} "
              f"fw={_fmt(d['fw'].item()):>13} lw={_fmt(d['lw'].item()):>13} "
              f"T_fire={d['T_fire'].item():.1f} N_out={d['N_out'].item():.1f}")


def print_per_edge(edges, per_edge):
    for e in sorted(edges, key=lambda e: e["id"]):
        fill = per_edge[e["id"]].item()
        print(f"  edge {e['id']} (node {e['src']}->{e['dst']}): "
              f"diff fill={fill:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir")
    ap.add_argument("solutions", nargs="+")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="dump per-node st/fw/lw of both models")
    ap.add_argument("--emit-hls", action="store_true",
                    help="also translate each applied design point to HLS C++")
    args = ap.parse_args()

    design_dir = args.design_dir.rstrip("/")
    model = design_dir.split("/")[-1]

    print(f"{'design point':<28}{'streamhls':>14}{'diff_model':>14}{'rel.err':>10}")
    print("-" * 66)
    for sol in args.solutions:
        shls = streamhls_latency(design_dir, model, sol, emit_hls=args.emit_hls)
        graph, out = diff_analyze(sol)
        mine = out["total_cycles"].item()
        rel = abs(mine - shls) / shls
        name = sol.split("/")[-1]
        print(f"{name:<28}{shls:>14}{mine:>14.0f}{rel:>9.2%}")
        if args.verbose:
            print_per_node(streamhls_per_node(graph), out["per_node"])
            print_per_edge(graph["edges"], out["per_edge"])
            print()
