#!/usr/bin/env bash
# Appearance-dose sweep, one arm per GPU. Replaces the lost scratch/run_scale_ab.sh.
#
#   bash tools/augmentation/run_aug_sweep.sh control 1
#   bash tools/augmentation/run_aug_sweep.sh realism 3 SEED 1      # extra trailing overrides
#
# Sweeping one knob, e.g. perspective aggressiveness:
#   bash tools/augmentation/run_aug_sweep.sh perspective 1 INPUT.COPY_PASTE.PERSPECTIVE.TILT_DEGREES 30.0
#
# NOTE: yacs type-checks overrides against the value already in the config, so a
# float knob needs a float literal - TILT_DEGREES 30 is rejected, 30.0 works. Same
# rule in train_net.py and tools/augmentation/dump_training_samples.py.
#
# The ARM CONFIG carries nothing but the photometric dose. Everything that must be
# identical across arms for the comparison to mean anything is set HERE, in one
# place, so no arm file can quietly drift:
#   - MIN_SCALE/MAX_SCALE : frozen by wave 0 (the scale A/B rerun on top of the
#                           FixedSizeCrop seg_pad_value fix). Until wave 0 reports,
#                           SCALE_MIN/SCALE_MAX below are the 1.0/1.0 no-op, which
#                           is also the condition with no pad band at all.
#   - MAX_ITER / SEED     : same budget and same seed for every arm.
#   - IMS_PER_BATCH       : 2 img/GPU hangs during eval on an 11GB 2080 Ti (see
#                           scale_jitter_ab_phase2.md gotcha 1), so one arm per GPU
#                           means batch 1. Absolute numbers are therefore comparable
#                           WITHIN this sweep only - that is what the control is for.
set -euo pipefail

# All paths below are relative to the repo root (configs/, runs/, .venv/),
# so cd there regardless of where this is invoked from - it moved out of
# scratch/ into tools/augmentation/ and is no longer one level from root.
cd "$(dirname "$0")/../.." || exit 1

ARM="${1:?usage: run_aug_sweep.sh <mild|control|strong|extreme> <gpu> [extra opts...]}"
GPU="${2:?usage: run_aug_sweep.sh <arm> <gpu> [extra opts...]}"
shift 2

SCALE_MIN="${SCALE_MIN:-1.0}"
SCALE_MAX="${SCALE_MAX:-1.0}"
MAX_ITER="${MAX_ITER:-6000}"
SEED="${SEED:-42}"
BATCH="${BATCH:-1}"        # total images per iteration, across all GPUs used

CFG="configs/coco/instance-segmentation/maskdino_R50_surgical_tools_reclassify_unfreeze_encoder_decoder_aug_${ARM}.yaml"
[ -f "$CFG" ] || { echo "no such arm config: $CFG" >&2; exit 2; }

# NOTE: do NOT timestamp here. train_net.py already prefixes OUTPUT_DIR with its
# own %Y%m%d%H%M%S (see its setup(), ~line 640) so that concurrent ranks agree on
# one directory. Stamping twice produced names like
# "20261005211545_20261005211541_aug_control" AND split the tee'd log into a
# different directory from the run's own metrics.json.
OUT="runs/aug_${ARM}_seed${SEED}"
LOGDIR="runs/_logs"; mkdir -p "$LOGDIR"
LOG="${LOGDIR}/aug_${ARM}_seed${SEED}_$(date +%Y%m%d%H%M%S).log"

echo "[sweep] arm=$ARM gpu=$GPU out=${OUT} (train_net.py will timestamp it) log=$LOG"
echo "[sweep] scale=$SCALE_MIN/$SCALE_MAX iter=$MAX_ITER seed=$SEED batch=$BATCH gpus=$GPU"

# GPU may be a single index ("1") or a comma list ("0,1,2,3"); NUM_GPUS is derived
# from it, so effective batch = NUM_GPUS * IMS_PER_BATCH-per-gpu. Multi-GPU is how
# the effective batch gets above 1 at all: 2 images on ONE 11GB 2080 Ti peaks at
# ~9.5GB and hangs during eval (scale_jitter_ab_phase2.md gotcha 1), so the only
# safe way to a larger batch is more cards at 1 image each.
NUM_GPUS=$(echo "$GPU" | tr ',' '\n' | grep -c .)

CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python train_net.py \
  --num-gpus "$NUM_GPUS" --config-file "$CFG" \
  INPUT.MIN_SCALE "$SCALE_MIN" INPUT.MAX_SCALE "$SCALE_MAX" \
  SOLVER.MAX_ITER "$MAX_ITER" SOLVER.IMS_PER_BATCH "$BATCH" \
  SEED "$SEED" OUTPUT_DIR "$OUT" "$@" \
  2>&1 | tee "$LOG"
