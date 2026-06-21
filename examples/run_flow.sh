#!/bin/bash
# End-to-end differentiable-optimization + HLS-fidelity flow for one benchmark.
#
# Runs all three stages in one shot, threading a single run directory through
# them so there are no timestamps to copy by hand:
#   Stage 1   optimize_timing.py optimize   (diff model, fast)
#   Stage 1b  compare_timing.py              (diff vs Stream-HLS analytical model)
#   Stage 2   eval_timing.py --phase launch  (Vitis csynth, local & blocking)
#   Stage 3   eval_timing.py --phase collect + optimize_timing.py reconcile
#
# Usage:
#   ./run_flow.sh <base_space.json> [optimize args...]
# e.g.
#   ./run_flow.sh designs/polybench/k3mm/k3mm/mlir/intermediates/k3mm_space.json \
#                 --steps 300 --ckpt-interval 50 --board-dsps 6840
#
# Env knobs:
#   JOBS=N       concurrent csynth runs (default 4)
#   BUFFERIZE=1  bufferize-func-args for Stage 2 (set for models with tensor
#                func args, e.g. MHSA; default 0 for the polybench inputs)
# Needs streamhls-opt/translate + the streamhls python on PATH, and vitis_hls
# (run `module load vitis/2022.1` first). Stage 2 is local/blocking; for async
# SLURM runs use the manual `--runner sbatch` steps instead.
set -euo pipefail

REPO=/pool0/pmcewen/rsgvm13dir/codesign2
export PATH=$REPO/Stream-HLS/build/bin:/pool0/pmcewen/rsgvm13dir/miniforge3/envs/streamhls/bin:$PATH
cd "$REPO/Stream-HLS/examples"

base=${1:?usage: run_flow.sh <base_space.json> [optimize args...]}
shift || true
command -v vitis_hls >/dev/null \
    || { echo "vitis_hls not on PATH; run: module load vitis/2022.1"; exit 1; }

interm=$(dirname "$base")
stem=$(basename "$base" .json)
out=opt_log/${stem}_$(date +%Y%m%d_%H%M%S)

echo "== Stage 1: optimize -> $out =="
rm -f "$interm/${stem}"_opt*.json          # clear stale checkpoints so the glob is clean
python3 optimize_timing.py optimize "$base" --out-dir "$out" "$@"

ckpts=( "$interm/${stem}"_opt*.json )

design_dir=$(dirname "$(dirname "$interm")")    # interm = <design_dir>/mlir/intermediates
echo "== Stage 1b: diff vs Stream-HLS analytical model (bufferize=${BUFFERIZE:-0}) =="
python3 compare_timing.py "$design_dir" "${ckpts[@]}" \
    --bufferize "${BUFFERIZE:-0}" | tee "$out/compare.log"

echo "== Stage 2: csynth ${#ckpts[@]} checkpoints (local, JOBS=${JOBS:-4}, bufferize=${BUFFERIZE:-0}) =="
python3 eval_timing.py "${ckpts[@]}" --phase launch --runner local \
    --jobs "${JOBS:-4}" --bufferize "${BUFFERIZE:-0}"

echo "== Stage 3: collect + reconcile =="
python3 eval_timing.py "${ckpts[@]}" --phase collect --log-dir "$out"
collect_csv=$(ls "$out"/eval_adhoc_*.csv)
python3 optimize_timing.py reconcile "$out" "$collect_csv"

echo
echo "DONE. Results in $out/:"
echo "  trajectory.csv  -- analytical latency & DSP vs optimization step"
echo "  manifest.csv    -- checkpoint design points + predictions"
echo "  compare.log     -- diff vs Stream-HLS analytical latency per checkpoint"
echo "  fidelity.csv    -- step, lat_cont, lat_round, hls_actual, round_gap, model_err"
