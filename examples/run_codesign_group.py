#!/usr/bin/env python3
"""Batch-run codesign emit-optimize over named groups of benchmark kernels.

The input YAML maps a group name to a list of kernel names (each must be
registered under the "codesign" label in data.py's model_configs):

    polybench:
      - gemm
      - k3mm

For every kernel this invokes run_streamhls.py with
`--codesign-mode emit --codesign-optimize`, which regenerates the input MLIR
from the PyTorch source, runs the combined optimization solver, and writes the
*optimized* design point. The design directory is created by the run itself --
you do not pass one; it is <out-root>/<group>/<kernel>/<kernel>, and the emitted
point lands at <design_dir>/mlir/intermediates/<kernel>_space.json.

The PyTorch source for a group lives at <pymodels-root>/<group> (passed as the
benchmark path); the group name must match that pymodels subdirectory.

It then writes (and prints) a groups YAML of the emitted _space.json paths --
the same format eval_timing.py --group consumes.

Run from Stream-HLS/examples with the env sourced (streamhls-opt, ampl, Gurobi
on PATH; see setup-env.sh).
"""
import argparse
import os
import subprocess

import yaml

# This script lives in Stream-HLS/examples; run_streamhls.py and the relative
# pymodels / designs paths resolve from there.
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


def run_one(kernel, group, out_root, pymodels_root, dsps, tilelimit, timelimit):
    """Emit the optimized design point for one kernel; return its _space.json."""
    benchmark_path = os.path.join(pymodels_root, group)
    assert os.path.isdir(os.path.join(EXAMPLES_DIR, benchmark_path)), \
        f"benchmark path is not a directory: {benchmark_path}"
    # run_streamhls sets prj_path = <prjsdir>/<kernel>, so the design dir (and
    # thus the emitted JSON) ends up one level below the --outdir we pass.
    prjsdir = os.path.join(out_root, group, kernel)
    design_dir = os.path.join(prjsdir, kernel)
    cmd = [
        "python", "run_streamhls.py",
        "-b", benchmark_path, "-k", kernel, "-d", prjsdir,
        # -O 5 only ensures run_streamhls's perm/paral/comb flags are defined;
        # the combined optimizer is driven by --codesign-optimize.
        "-O", "5",
        "--codesign-mode", "emit", "--codesign-optimize",
        "--dsps", str(dsps), "--tilelimit", str(tilelimit),
        "--timelimit", str(timelimit),
    ]
    subprocess.run(cmd, check=True, cwd=EXAMPLES_DIR)
    out = os.path.join(design_dir, "mlir", "intermediates", f"{kernel}_space.json")
    assert os.path.isfile(os.path.join(EXAMPLES_DIR, out)), \
        f"emit did not produce {out}"
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config",
                    help="YAML mapping group name -> list of kernel names")
    ap.add_argument("-o", "--out", default="codesign_groups.yaml",
                    help="output groups YAML of emitted _space.json paths")
    ap.add_argument("--out-root", default="designs",
                    help="root for created design dirs (<root>/<group>/<kernel>/<kernel>)")
    ap.add_argument("--pymodels-root", default="pymodels",
                    help="root of PyTorch sources; benchmark path is <root>/<group>")
    ap.add_argument("--dsps", type=int, default=7680, help="board DSP budget")
    ap.add_argument("--tilelimit", type=int, default=10, help="tiling factor limit")
    ap.add_argument("--timelimit", type=int, default=20,
                    help="MINLP solver time limit (minutes)")
    args = ap.parse_args()

    with open(args.config) as f:
        groups = yaml.safe_load(f)
    assert isinstance(groups, dict), "config must map group name -> list of kernels"

    out_groups = {}
    for group, kernels in groups.items():
        out_groups[group] = [
            run_one(k, group, args.out_root, args.pymodels_root,
                    args.dsps, args.tilelimit, args.timelimit)
            for k in kernels
        ]

    out_path = os.path.join(EXAMPLES_DIR, args.out)
    with open(out_path, "w") as f:
        yaml.safe_dump(out_groups, f, default_flow_style=False, sort_keys=False)

    print("\nEmitted optimized design points:")
    for group, paths in out_groups.items():
        print(f"{group}:")
        for p in paths:
            print(f"  - {p}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
