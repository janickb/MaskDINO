# Handoff — set-C single-class segmenter (2026-11-08)

**The deliverable going forward is the set-C trained segmenter.** Phase-1 augmentation work is
CLOSED — do not start new experiments on it (see §4).

## 1. The live run

`runs/20261108204507_p2_setc_cp` — config
[configs/coco/instance-segmentation/p2_setc_cp.yaml](configs/coco/instance-segmentation/p2_setc_cp.yaml)

| | |
|---|---|
| train | `train_phase2_set_c_mm-1.0x_singleclass` — 460 sparse frames, 1 instrument/frame, 23 types (scissor wear variants + 2 screwdrivers) |
| val (target) | `val_set_c_mm-1.0x_singleclass` — 256 dense frames, 1165 instances, mean 4.55/frame, same 23-type roster |
| val (monitor) | `val_set_ab_mm_singleclass` — 1024 frames; forgetting monitor only, its instruments are NOT in training |
| started from | `runs/20261007130733_single_class_v2/model_final.pth` → finetuned to `runs/20261108124636_p2_cp_s100/model_final.pth` (set-ab phase 2) → this run |
| recipe | copy-paste ON, scale 1.0/1.0, `MIN_VISIBILITY 0.0`, LR 1e-4, 6000 iters, 3 GPUs (2,3,4) at `IMS_PER_BATCH 6`, `NCCL_P2P_DISABLE=1` |

Results at the first eval tick (iter 1999):

| val set | f1 | precision | recall | mean IoU |
|---|---|---|---|---|
| **val_set_c** (1146 GT) | **0.9736** | 0.9831 | 0.9642 | 0.8553 |
| val_set_ab (4040 GT) | 0.7980 | 0.8702 | 0.7369 | 0.8334 |

**Open question for whoever continues:** that A+B figure was **0.9752** before this finetune, so
adding set C costs ~17.7 points on A+B. If a general A+B+C segmenter is the goal, this is the
problem to solve (replay/mixed training, or a lower LR). Nothing has been scored on all three
sets at once.

## 2. How to run things here

- `cd /home/janick.bilang/dev/MaskDINO`, interpreter `./.venv/bin/python`.
- `export SURGICAL_IMAGES_ROOT=/home/janick.bilang/training/images_lambdablade-6000` — now at the
  TOP of `~/.bashrc`, above the non-interactive guard, so ssh/nohup runs see it. Without it,
  `resolve_images_root()` raises (three image roots match).
- **Detach long runs**: `setsid nohup … & disown`. A run launched as a child of a Claude session
  dies when that session exits (this cost a sweep once).
- **GPU contention is real.** Another session runs `train_cropcls.py` (branch-42 cropcls, sibling
  worktree `MaskDINO-Unet`) on GPUs 0/1; GPUs 5/6 belong to another user. Check `nvidia-smi`
  before launching — a collision OOMs *your* job, not theirs.
- **`/home` quota is ~650 MiB free** and a checkpoint is 323 MiB. The five non-winning
  `runs/*_p2_cp_*` checkpoints are ~1.6 GB of reclaimable space.
- `train_phase2_set_ab_mm` under the images root is a **symlink** to the babylambda mount
  (physically on 131.152.136.197, read over sshfs). `config.json` records only the dataset name,
  so nothing in the artifacts reveals that.

## 3. Traps that cost time in this session

- **Copy-paste is mandatory for phase-2 renders, not optional.** They are 1 instrument/frame. With
  compositing: f1 0.9752. Without, the identical setup collapsed to **0.118** — the model emitted
  0.79 masks/image against ~3.95 GT. Never train on a sparse set uncomposited against dense val.
- **With >1 `DATASETS.TEST`, metrics.json keys are prefixed** (`val_set_ab_mm_singleclass/instance_matching/classaware_f1`).
  Parsers written for the single-dataset form silently find nothing.
- `DATASETS.TEST_SAMPLE_STRIDE: 4` is inherited from `Base-Surgical-HDF5.yaml:27`. Easy to report
  "the full val set" when you scored a quarter of it. Full-stride cost: ~0.71 s/frame.
- **`score_mode` does not exist on this branch.** CLAUDE.md's `score_mode=detector` advice cannot be
  applied. `SCORE_THRESH` is 0.5; `sweep/f1_max` is the calibration-independent read.
- `sweep/classaware_f1_max` is **never emitted** for single-class runs (`_check_classes = num_classes > 1`).
  Use `sweep/f1_max`.
- Never use `pkill -f <pattern>` where the pattern matches your own command line — it kills your shell.
- Rank by `classaware_f1` (== `classagnostic_f1` under single-class). Do **not** rank by COCO AP: it
  divides by the eval set's category count.

## 4. Phase-1 augmentation — CLOSED, do not re-run

Two sweeps (12 arms) settled it. The actionable result is two lines:

```yaml
INPUT:
  MIN_SCALE: 0.3
  MAX_SCALE: 2.5
```

`MAX_SCALE` is the only axis that matters and it saturates at ~2.5 (f1 by MAX_SCALE: 1.2 → 0.749,
2.0 → 0.785-0.791, 2.5 → 0.812-0.835, 3.5 → 0.820, 4.0 → 0.824-0.831). `MIN_SCALE` is immaterial in
0.15-0.5. Photometric dose is inert (-0.1 points). Under copy-paste (phase 2) scale stops mattering
entirely, so phase-2 configs correctly use 1.0/1.0.

**Caveat on provenance:** the 12 sweep run directories were deleted on 2026-11-08, so these numbers
survive only in this document and the session transcript — no metrics.json, no checkpoints, no
tfevents. They are not artifact-verifiable. The conclusion reproduced across two independent sweeps
including a control arm under a different data regime, but do not present the figures as verified.

Also unmeasured, and worth knowing before claiming set A "does not transfer" to set B: the
`val_set_ab` score was only ever **pooled** over a ~50/50 A/B instance mix (521 vs 555). It was
never decomposed per-set, so a transfer gap is consistent with the data but not demonstrated by it.
