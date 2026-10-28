#!/usr/bin/env bash
# Replay every finished focal-sweep run's full-val eval per instance (CPU only),
# then run the analyses behind docs/experiments/scaling_experiment.md.
#
#   bash tools/scaling/analyze_focal_sweep.sh
#   OUT=runs/_analysis/focal_sweep_newval bash tools/scaling/analyze_focal_sweep.sh
#   EVAL_DIR=eval_full_v2 OUT=runs/_analysis/focal_sweep_newval_dataset_v2 bash tools/scaling/analyze_focal_sweep.sh
#
# Per runs/*_focal_<level>_seed<seed> dir that has $EVAL_DIR/ (default eval_full/,
# eval_full_<tag>/ from run_focal_sweep.sh VAL_TAG=<tag>), two replays, all in
# parallel (~1-2 min each on 8 threads):
#   inst_<level>_seed<seed>.npz  replay_focal_instances.py  (official matching, scores 0.5 / 0.3)
#   sw_<level>_seed<seed>.npz    sweep_focal_thresholds.py  (one label per mask, scores 0.10-0.90)
# Replays already present in OUT are reused - delete a .npz to redo it. If a
# level/seed was run twice, the newest run dir wins.
# Reports (OUT is under runs/, so gitignored):
#   $OUT/focal_analysis.txt      paired miss / swap / correct, mask quality, AP
#   $OUT/threshold_analysis.txt  best-threshold F1 per level, 03-vs-04 separability
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

OUT="${OUT:-runs/_analysis/focal_sweep}"
EVAL_DIR="${EVAL_DIR:-eval_full}"
mkdir -p "$OUT"

declare -A newest
for r in runs/*_focal_*_seed*/; do
  r=${r%/}
  [ -f "$r/$EVAL_DIR/inference/instances_predictions.pth" ] || { echo "[skip] $r: no $EVAL_DIR yet"; continue; }
  newest[${r#*_focal_}]=$r   # glob is sorted by timestamp prefix, so the last one wins
done

pids=()
for key in "${!newest[@]}"; do
  for kind in inst sw; do
    npz="$OUT/${kind}_${key}.npz"
    [ -f "$npz" ] && { echo "[reuse] $npz"; continue; }
    script=tools/scaling/replay_focal_instances.py
    [ "$kind" = sw ] && script=tools/scaling/sweep_focal_thresholds.py
    ./.venv/bin/python "$script" "${newest[$key]}" "$npz" --eval-dir "$EVAL_DIR" \
      2>&1 | grep -E "^\[(replay|sweep)\]|Error|Traceback" &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p"; done

./.venv/bin/python tools/scaling/analyze_focal_sweep.py "$OUT" | tee "$OUT/focal_analysis.txt"
./.venv/bin/python tools/scaling/analyze_focal_thresholds.py "$OUT" | tee "$OUT/threshold_analysis.txt"
