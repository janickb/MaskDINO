#!/usr/bin/env bash
# Appearance-dose sweep, one arm per run on a single GPU. Replaces the lost
# scratch/run_scale_ab.sh.
#
#   bash tools/augmentation/run_aug_sweep.sh control 6
#   SEED=1 bash tools/augmentation/run_aug_sweep.sh low 6        # repeat seed for the noise floor
#   bash tools/augmentation/run_aug_sweep.sh medium 6 TEST.EVAL_PERIOD 500  # extra trailing overrides
#
# ARM is the basename of a config in configs/coco/instance-segmentation/augmentation/:
#   control | low | medium | object_jitter_only | realistic_measured
# The old phase-2 wave's maskdino_R50_..._aug_<arm>.yaml files also live there now,
# and run by their full basename (without .yaml).
#
# NOTE: yacs type-checks overrides against the value already in the config, so a
# float knob needs a float literal - TILT_DEGREES 30 is rejected, 30.0 works. Same
# rule in train_net.py and tools/augmentation/dump_training_samples.py.
#
# The ARM CONFIG carries nothing but the photometric dose; scale, batch, iters and
# LR all come from the shared base config every arm inherits, so arms can only
# differ in their dose. The only thing set HERE is SEED (same for every arm unless
# overridden, and part of the output name so repeat seeds don't collide).
# Absolute numbers are comparable WITHIN one sweep only - that is what the control
# is for.
set -euo pipefail

# All paths below are relative to the repo root (configs/, runs/, .venv/),
# so cd there regardless of where this is invoked from - it moved out of
# scratch/ into tools/augmentation/ and is no longer one level from root.
cd "$(dirname "$0")/../.." || exit 1

ARM="${1:?usage: run_aug_sweep.sh <control|low|medium|object_jitter_only|realistic_measured> <gpu> [extra opts...]}"
GPU="${2:?usage: run_aug_sweep.sh <arm> <gpu> [extra opts...]}"
shift 2

SEED="${SEED:-42}"

CFG="configs/coco/instance-segmentation/augmentation/${ARM}.yaml"
[ -f "$CFG" ] || { echo "no such arm config: $CFG" >&2; exit 2; }

# NOTE: do NOT timestamp here. train_net.py already prefixes OUTPUT_DIR with its
# own %Y%m%d%H%M%S (see its setup(), ~line 640) so that concurrent ranks agree on
# one directory. Stamping twice produced names like
# "20261005211545_20261005211541_aug_control" AND split the tee'd log into a
# different directory from the run's own metrics.json.
# augexp_ (matching the arm configs' own OUTPUT_DIR) rather than the old wave's
# aug_, so e.g. the new control can't be mistaken for runs/*_aug_control_seed42.
OUT="runs/augexp_${ARM}_seed${SEED}"
LOGDIR="runs/_logs"; mkdir -p "$LOGDIR"
LOG="${LOGDIR}/augexp_${ARM}_seed${SEED}_$(date +%Y%m%d%H%M%S).log"

echo "[sweep] arm=$ARM gpu=$GPU seed=$SEED out=${OUT} (train_net.py will timestamp it) log=$LOG"

CUDA_VISIBLE_DEVICES="$GPU" ./.venv/bin/python train_net.py \
  --num-gpus 1 --config-file "$CFG" \
  SEED "$SEED" OUTPUT_DIR "$OUT" "$@" \
  2>&1 | tee "$LOG"
