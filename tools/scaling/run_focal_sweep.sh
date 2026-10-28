#!/usr/bin/env bash
# Focal-length sweep, one arm per run on a single GPU: phase-2 finetune on one
# level, then re-evaluate model_final on the FULL val set (stride 1) into
# <run>/eval_full/ - the in-training evals only see every 4th frame.
#
#   bash tools/scaling/run_focal_sweep.sh 1.0x 0
#   SEED=1 bash tools/scaling/run_focal_sweep.sh 1.5x 2
#   EVAL_ONLY=1 SEED=1 bash tools/scaling/run_focal_sweep.sh 1.5x 2      # redo the full eval of the newest such run
#   EVAL_ONLY=1 VAL_TAG=v2 bash tools/scaling/run_focal_sweep.sh 1.5x 2  # eval on val_setab_fixedsurf_mm_v2-1.5x -> <run>/eval_full_v2/
#   bash tools/scaling/run_focal_sweep.sh 0.67x 6 SOLVER.MAX_ITER 4000   # extra trailing overrides
#
# LEVEL is 0.67x | 1.0x | 1.5x and selects the train AND val set together, so an
# arm can never train on one focal length and validate on another. 1.0x is the
# control. VAL_TAG picks another val set variant of the same level
# (val_setab_fixedsurf_mm_<tag>-<level>); its full eval goes to eval_full_<tag>/, so
# it never mixes with eval_full/. Everything else comes from
# configs/.../scaling/focal_sweep.yaml. The whole grid (levels x seeds over several
# GPUs): tools/scaling/launch_focal_grid.sh.
set -euo pipefail

# All paths below are relative to the repo root (configs/, runs/, .venv/).
cd "$(dirname "$0")/../.." || exit 1

LEVEL="${1:?usage: run_focal_sweep.sh <0.67x|1.0x|1.5x> <gpu> [extra opts...]}"
GPU="${2:?usage: run_focal_sweep.sh <level> <gpu> [extra opts...]}"
shift 2
case "$LEVEL" in
  0.67x|1.0x|1.5x) ;;
  *) echo "unknown level: $LEVEL (expected 0.67x|1.0x|1.5x)" >&2; exit 2 ;;
esac

SEED="${SEED:-42}"
CFG="configs/coco/instance-segmentation/scaling/focal_sweep.yaml"
IMAGES="/home/janick.bilang/training/images_dgx-a100"
TRAIN_DS="reclass_setab_fixedsurf_mm-${LEVEL}"
TAG="${VAL_TAG:+_$VAL_TAG}"
VAL_DS="val_setab_fixedsurf_mm${TAG}-${LEVEL}"

# NOTE: do NOT timestamp here - train_net.py prefixes OUTPUT_DIR with its own
# %Y%m%d%H%M%S (see run_aug_sweep.sh for why stamping twice breaks things).
NAME="focal_${LEVEL}_seed${SEED}"
LOGDIR="runs/_logs"; mkdir -p "$LOGDIR"
LOG="${LOGDIR}/${NAME}_$(date +%Y%m%d%H%M%S).log"

# Preflight. A missing set only fails deep inside train_net.py's setup, and a
# half-copied val set silently evaluates on a subset - so every level's val set
# must hold the same number of frames (same simulations, one per focal length).
# (|| true: under pipefail, find on a missing dir would abort the script via set -e)
count_frames() { find "$1" -maxdepth 1 -name '*.hdf5' -size +0 2>/dev/null | wc -l || true; }
n_train=$(count_frames "$IMAGES/train_phase2_set_ab_fixedsurf_mm-${LEVEL}")
# EVAL_ONLY never reads the train set. Without it, the train/test class-mapping
# check in train_net.py silently passes - the eval must then be checked by hand
# against model_final.pth's own class_mapping.
[ "$n_train" -gt 0 ] || [ -n "${EVAL_ONLY:-}" ] || { echo "no train frames for $LEVEL" >&2; exit 3; }
n_val=""
for l in 0.67x 1.0x 1.5x; do
  n=$(count_frames "$IMAGES/val_set_ab_fixedsurf_mm${TAG}-${l}")
  [ "$n" -gt 0 ] || { echo "no val frames for $l (val_set_ab_fixedsurf_mm${TAG}-${l})" >&2; exit 3; }
  [ -z "$n_val" ] || [ "$n" -eq "$n_val" ] || {
    echo "val frame counts differ across levels ($n_val vs $n for $l) - copy still running?" >&2; exit 3; }
  n_val=$n
done
echo "[focal] level=$LEVEL gpu=$GPU seed=$SEED train=$TRAIN_DS ($n_train frames) val=$VAL_DS ($n_val frames) log=$LOG"

if [ -z "${EVAL_ONLY:-}" ]; then
  CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python train_net.py \
    --num-gpus 1 --config-file "$CFG" \
    DATASETS.TRAIN "(\"$TRAIN_DS\",)" DATASETS.TEST "(\"$VAL_DS\",)" \
    SEED "$SEED" OUTPUT_DIR "runs/$NAME" "$@" \
    2>&1 | tee "$LOG"
fi

# The run train_net.py just created (or, with EVAL_ONLY, the newest one of this arm).
RUN=$(ls -dt runs/*_"$NAME" 2>/dev/null | head -1 || true)
[ -n "$RUN" ] && [ -f "$RUN/model_final.pth" ] || {
  echo "no finished run matching runs/*_$NAME (model_final.pth missing)" >&2; exit 4; }

# Separate OUTPUT_DIR: --eval-only into the run's own dir would overwrite the
# in-training inference/ artifacts. The run's saved config.yaml already holds
# any trailing overrides; the val set is set explicitly so VAL_TAG applies.
EVAL_OUT="$RUN/eval_full${TAG}"
echo "[focal] full-val eval (stride ${EVAL_STRIDE:-1}) of $RUN on $VAL_DS -> $EVAL_OUT" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python train_net.py --eval-only \
  --num-gpus 1 --config-file "$RUN/config.yaml" \
  MODEL.WEIGHTS "$RUN/model_final.pth" DATASETS.TEST_SAMPLE_STRIDE "${EVAL_STRIDE:-1}" \
  DATASETS.TEST "(\"$VAL_DS\",)" OUTPUT_DIR "$EVAL_OUT" \
  2>&1 | tee -a "$LOG"
echo "[focal] done: $RUN" | tee -a "$LOG"
