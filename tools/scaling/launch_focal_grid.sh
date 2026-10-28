#!/usr/bin/env bash
# Launch the focal-length grid (levels x seeds) over several GPUs, one run per GPU
# at a time (IMS_PER_BATCH 6 needs a whole 40 GB card). Jobs are ordered
# seed-major and dealt round-robin, so each seed's three levels run side by side
# and a partially finished grid is still a balanced comparison.
#
#   bash tools/scaling/launch_focal_grid.sh "0 2 6 7"                 # seeds 42 1 2, all levels
#   bash tools/scaling/launch_focal_grid.sh "0 2 6" "42"              # one seed, one wave
#   LEVELS="1.0x 1.5x" bash tools/scaling/launch_focal_grid.sh "0 2" "42 1"
#   FORCE=1 bash tools/scaling/launch_focal_grid.sh "3"               # GPU already has memory in use
#   DRY_RUN=1 bash tools/scaling/launch_focal_grid.sh "0 2 6 7"       # print the queues, launch nothing
#   EVAL_ONLY=1 bash tools/scaling/launch_focal_grid.sh "0 0 0"        # re-run only the full-val evals, 3 side by side on GPU 0
#   EVAL_ONLY=1 VAL_TAG=v2 bash tools/scaling/launch_focal_grid.sh "1 1 1"  # evals on the _v2 val sets -> <run>/eval_full_v2/
#
# Each GPU works through its own queue in the background (survives logout).
# Queue progress: runs/_logs/focal_queue_gpu<N>_<stamp>.log; each run's full
# output: the runs/_logs/focal_<level>_seed<seed>_*.log that run_focal_sweep.sh tees.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

read -r -a GPUS <<< "${1:?usage: launch_focal_grid.sh \"<gpu ids>\" [\"<seeds>\"]}"
read -r -a SEEDS <<< "${2:-42 1 2}"
read -r -a LEVELS <<< "${LEVELS:-0.67x 1.0x 1.5x}"

# Another user's job on a card means OOM at batch 6, not a slow run.
for g in "${GPUS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 1000 ] && [ -z "${FORCE:-}" ]; then
    echo "GPU $g already has ${used} MiB in use - pick another or set FORCE=1" >&2; exit 2
  fi
done

# ~0.5 GB per run (323 MB model_final + ~120 MB full eval + logs). A full quota
# kills runs mid-write, so refuse up front rather than lose the grid's tail.
# EVAL_ONLY=1 (passed through to run_focal_sweep.sh) only rewrites an existing
# eval_full/ in place, so it needs little more than a log - unless VAL_TAG sends
# it to a fresh eval_full_<tag>/ (75-150 MB, most at 1.5x).
per_run_mb=500
[ -z "${EVAL_ONLY:-}" ] || per_run_mb=50
[ -z "${EVAL_ONLY:-}" ] || [ -z "${VAL_TAG:-}" ] || per_run_mb=150
need_mb=$(( ${#SEEDS[@]} * ${#LEVELS[@]} * per_run_mb ))
read -r used_kb limit_kb < <(quota -w 2>/dev/null | awk 'NR > 2 { gsub(/\*/, "", $2); print $2, $4; exit }') || true
if [ -n "${limit_kb:-}" ] && [ "$limit_kb" -gt 0 ]; then
  free_mb=$(( (limit_kb - used_kb) / 1024 ))
  echo "[grid] quota: ${free_mb} MB free, grid needs ~${need_mb} MB"
  [ "$free_mb" -gt "$need_mb" ] || { echo "not enough quota headroom - free space or launch fewer seeds" >&2; exit 3; }
else
  echo "[grid] could not read quota - check 'quota -s' yourself (grid needs ~${need_mb} MB)" >&2
fi

JOBS=()
for s in "${SEEDS[@]}"; do
  for l in "${LEVELS[@]}"; do
    JOBS+=("$s:$l")
  done
done

# PID suffix: two launches within one second must not share log names
STAMP=$(date +%Y%m%d%H%M%S)_$$
mkdir -p runs/_logs
for i in "${!GPUS[@]}"; do
  g=${GPUS[$i]}
  queue=()
  for j in "${!JOBS[@]}"; do
    if (( j % ${#GPUS[@]} == i )); then queue+=("${JOBS[$j]}"); fi
  done
  [ ${#queue[@]} -gt 0 ] || continue
  # queue index in the name: a GPU listed twice ("1 1") runs two queues side by
  # side, which suits the light eval-only jobs (~5 GB, ~30 % GPU each)
  qlog="runs/_logs/focal_queue_gpu${g}_q${i}_${STAMP}.log"
  echo "[grid] GPU $g: ${queue[*]} -> $qlog"
  [ -z "${DRY_RUN:-}" ] || continue
  # A failed arm is logged and skipped so it can't stall the rest of its queue.
  nohup setsid bash -c '
    gpu=$1; shift
    for job in "$@"; do
      seed=${job%%:*}; level=${job#*:}
      echo "[queue] $(date "+%F %T") start level=$level seed=$seed"
      if SEED=$seed bash tools/scaling/run_focal_sweep.sh "$level" "$gpu" > /dev/null 2>&1; then
        echo "[queue] $(date "+%F %T") done  level=$level seed=$seed"
      else
        echo "[queue] $(date "+%F %T") FAILED level=$level seed=$seed - see runs/_logs/focal_${level}_seed${seed}_*.log"
      fi
    done' _ "$g" "${queue[@]}" > "$qlog" 2>&1 < /dev/null &
done
[ -z "${DRY_RUN:-}" ] || { echo "[grid] DRY_RUN - nothing launched"; exit 0; }
echo "[grid] launched ${#JOBS[@]} runs on ${#GPUS[@]} GPU(s); follow with: tail -f runs/_logs/focal_queue_gpu*_${STAMP}.log"
