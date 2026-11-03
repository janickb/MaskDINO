# Giving the crop classifier back the scale the letterbox removed

Design note, 2026-11-02. **Status: not built, not specced, no run.**

This is **not** a `requirements_*.md` and must not be used as one. The review gate in
`CLAUDE.md` requires the user's verbatim ask as the reviewers' source of truth; this file is
analysis written by the session, so on its own it proves only self-consistency. If this gets
built, write `requirements_scale.md` from the raw ask first.

What is settled here: the cue has been **measured** (section 2), the design is chosen
(section 5), and the plumbing is enumerated (section 6). What is not: whether it moves
`classaware_f1`.

---

## 1. What the letterbox actually discards

`letterbox_crop` scales uniformly on the longest axis — `cropcls/crops.py`:

```python
scale = size / max(h, w)
```

Aspect ratio **survives** the letterbox: under `blackout: true` the pad region is black and
the content box's extent is directly readable, so the trunk can see `min(h,w)/max(h,w)`.
What dies is the one scalar `scale`. A 12 px forcep04 mask and a 446 px scalpel arrive
filling the same 448 canvas, and nothing in the input distinguishes them.

Two separate things follow from that, and they are worth keeping apart:

1. **Scale as a class prior.** Real instruments have real sizes. Measured in section 2.
2. **Scale as disclosure of the resize factor.** At `crop_size: 448` a 12 px mask is
   upsampled 37x and a 446 px one 1x. Those two crops have completely different
   interpolation blur and texture statistics, and the trunk is currently told nothing about
   which it is looking at. This is the same family as the 224 -> 448 finding
   (`requirements_feature_map_resolution.md`) and the probe below cannot measure it.

Mechanism 2 is the better reason to build this. Mechanism 1 is the one that looks impressive
and mostly is not — see section 3.

## 2. Measured: the cue is real

Probe over the static crop folder `crops/train` — patches are stored at **native
resolution** (`cropcls/crop_folder.py` module docstring), so the PNG's own dimensions are the
tight box and alpha is the mask. No model, no training run, no re-extraction. Script in the
appendix.

4623 crops / 918 frames. Split is **frame-level** (`crc32(frame_stem) % 5 == 0`), so no frame
contributes to both sides; train 3724 / val 899. Features are `log2` of pixel quantities.

| class | n | long edge med | p5 | p95 | mask area med | fill |
|---|---|---|---|---|---|---|
| forcep01_mm | 568 | 132 | 108 | 146 | 932 | 0.129 |
| forcep02_mm | 595 | 134 | 106 | 145 | 1178 | 0.149 |
| forcep03_mm | 537 | 185 | 159 | 196 | 1788 | 0.088 |
| forcep04_mm | 562 | 187 | 163 | 199 | 1803 | 0.085 |
| scalpel01_mm | 511 | 247 | 202 | 272 | 4671 | 0.151 |
| hammer01_mm | 658 | 320 | 255 | 340 | 7117 | 0.135 |
| sharpspoon01_mm | 621 | 410 | 321 | 447 | 3672 | 0.051 |
| scarstick01_mm | 571 | 294 | 232 | 323 | 2843 | 0.076 |

8-way val top-1 from geometry alone:

| features | dim | linear | mlp 64-64 |
|---|---|---|---|
| majority class | — | 0.149 | — |
| box: `hi`, `lo`, `sqrt(hi*lo)` | 3 | 0.614 | 0.722 |
| **mask area: `sqrt(area)`** | **1** | **0.740** | 0.746 |
| box + mask area + fill | 4 | **0.776** | 0.774 |

Two things to take from the table:

- **`sqrt(mask area)` is the single best feature** — one number beats three box numbers.
  Expected: a diagonal shaft has a large box and a small area, so the box conflates size
  with orientation while area does not. If only one scalar is shipped, ship this one.
- A linear model is enough once area is included (0.776 linear vs 0.774 mlp). The cue is
  monotone in size; there is no capacity to be gained in the side network.

## 3. Where it helps, and where it does not

Pairwise val top-1 from the 3 box features (0.5 = size says nothing):

```
forcep01 vs forcep02   0.599      <- chance
forcep03 vs forcep04   0.600      <- chance
all 26 other pairs     0.942 - 0.996
```

Adding mask area and fill barely moves the two that matter: forcep01/02 0.599 -> 0.653,
forcep03/04 0.600 -> **0.585**.

Read the per-class table for why. forcep03/04 are 185 vs 187 px long and 1788 vs 1803 px in
area — **0.8% apart**. `cropcls-results.md` section 3 already recorded this from the other
direction ("median area 1421 vs 1469 px, aspect 1.80 vs 1.69, fill 0.072 vs 0.070 — all
within 6%"; different crop population, same conclusion).

**So scale separates every pair except the only pair that is broken.** And the classes it
does separate are, per `cropcls-results.md` section 3, already "within 0.2–2.3 points of the
baseline" — there is no headroom there to collect. The 0.776 number means *the cue exists*,
not *this is a gain*. Do not quote it as a predicted improvement.

The honest expectation: a small gain from mechanism 2, bounded above by a couple of points of
per-class slack on the non-forceps classes, and **nothing on the forceps collapse**. The
thing with real headroom there is still mask repair (`dil16` / `close16`, +4.2 points).

## 4. The shortcut warning

Look at the p5–p95 spread: every class sits within about ±15% of its own median long edge
(hammer 255–340 on a median of 320). Apparent size is `real size * focal / depth`, so that
narrow spread means **camera distance barely varies across this scene set**. Apparent size is
therefore nearly a constant per class — which is exactly why the probe scores 0.776, and
exactly why it may be a shortcut that evaporates under a different camera setup.

Before trusting any gain from this feature:

- Score it on a **genuinely different camera setup**. The obvious candidate on this host is
  not one: `images_dgx-a100/train_phase1_pool_set_a_mm` and
  `images_lambdablade-6000/train_phase1_pool_set_a_mm` are byte-identical copies as of
  2026-11-03 (same 1024 frame names and sizes, digest `0d90fc4eb1bf` for both), so scoring on
  both measures nothing. An older note here cited digest `d0cd49821f66` as the discriminator;
  it matches nothing on disk now. Testing this needs a render with varied camera distance,
  which does not exist yet.
- Both probes above ran on `crops/train`, i.e. the `IoU >= 0.5 AND runner_up < 0.2`
  population. That is the same clean-isolated-instance bias that made crop top-1 mislead in
  both directions (`CLAUDE.md`). Treat these numbers as cue-existence only.
- A scale feature that helps on a fixed-camera render and is a liability on a varied one is a
  real outcome, not a bug. Decide in advance which you care about.

## 5. The design

Fuse at the **pooled features**, not as an input channel. `global_pool="avg"` already yields
`trunk.num_features` (2048 for resnet50, 768 for convnext_tiny), and reading that attribute
keeps the change trunk-agnostic — no per-trunk table, same reasoning as
`add_mask_input_channel` locating the first conv via `pretrained_cfg["first_conv"]`.

```python
# Rotation-invariant by construction: rotations 0/90/180/270 swap w and h, so min/max and
# area survive. Feeding h/w unpermuted while the pixels ARE permuted is a silent
# train/score mismatch, not an error.
def scale_feats(h, w, area, ref=32.0):
    hi, lo = max(h, w), min(h, w)
    return torch.tensor([
        math.log2(hi / ref),
        math.log2(lo / ref),
        math.log2(math.sqrt(max(area, 1.0)) / ref),
        math.log2(max(area, 1.0) / (hi * lo)),        # fill
    ]) / 2.0        # lands in ~+-2, the range of normalized RGB


class CropClassifier(nn.Module):
    def __init__(self, ..., scale_feats: bool = False):
        ...
        self.scale_mlp = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
        ) if scale_feats else None
        self.scale_head = nn.Linear(64, num_classes) if scale_feats else None
        if scale_feats:
            nn.init.zeros_(self.scale_head.weight)
            nn.init.zeros_(self.scale_head.bias)

    def forward(self, x, aux=None):
        logits = self.head(self.trunk(x))
        if self.scale_head is not None:
            logits = logits + self.scale_head(self.scale_mlp(aux))
        return logits
```

**Zero-initializing `scale_head` is the point**, exactly as the zero-initialized 4th input
channel is the point in the mask-channel arm: at iteration 0 the run is numerically identical
to the current model, so the arm measures the feature and not a perturbed trunk.

Two knobs to decide, not sweep blindly:

- **Additive logits (above) vs FiLM.** Additive lets scale shift logits only — it can express
  mechanism 1 and not mechanism 2. FiLM on the pooled vector, `f * (1 + gamma) + beta` with
  `gamma, beta = MLP(aux)` zero-initialized, lets scale *modulate* features, which is what
  mechanism 2 needs. If the stated motivation is "calibrate for upsampling blur", build FiLM;
  additive-only cannot test that hypothesis.
- **`ref` and the `/2`.** Pinned constants, not hyperparameters. They exist to put the
  feature in the same range as normalized RGB so the side network's gradients are comparable;
  the same argument as the mask channel being 0.0/1.0 rather than 0/255.

`area` must be the **undilated** mask area, or `mask_dilate` becomes a second variable
inside this feature.

## 6. Plumbing — most of the work

`letterbox_crop` has `h`, `w` and `msk` in hand and returns none of them.
`to_model_input`'s docstring states the hazard plainly: one function serves all three crop
paths "because the training path and the scoring path disagreeing about the channel layout
does not raise - it just reports a worse number." Same hazard, new field.

1. **`cropcls/crops.py`** — `letterbox_crop` additionally returns the pre-resize
   `(h, w, area)`, or `to_model_input` returns `(x, aux)`. One contract, three callers.
2. **Three crop call sites**, all must change together: `cropcls/data.py` (live),
   `cropcls/crop_folder.py` (static), `cropcls/export.py` (Hungarian scoring).
3. **`CropBatch`** gains `aux: Tensor` of shape `(B, 4)`. The empty branch in `collate`
   needs its zero case — `torch.zeros((0, 4))`.
4. **Three `model(x)` call sites** become `model(x, aux)`: `train_cropcls.py` (eval and
   train loops) and `cropcls/export.py`. The export path is the one that produces
   `classaware_f1`; if it passes `aux=None` while training passed real values, the result is
   a silently worse number, never an exception. This is the single most likely way to waste
   a run.
5. **`in_chans` stays 3 or 4.** This adds no channel, so it composes with the mask-channel
   arms rather than conflicting with them.
6. **Static folder needs no re-extraction.** Patches are stored at native resolution and
   alpha carries the mask, so `(h, w, area)` is recoverable from the PNG as it sits.
   Verify `margin: 0` in the leaf config — a non-zero margin makes the PNG dimensions the
   *grown* box, not the tight one.
7. **Config:** one flag, `scale_feats: bool = False`, plus whatever names the
   additive/FiLM choice. `validate()` should reject `scale_feats` with `freeze_trunk` only if
   FiLM is used on frozen features — the additive head is trainable regardless, so unlike
   `mask_channel` this is not automatically a no-op.

## 7. Rejected, with reasons

- **Scale as a constant 5th input channel.** Cheap to wire through the existing first-conv
  surgery, and it would inherit the zero-init property for free. Rejected: a spatially
  constant scalar injected at conv1 has to survive 50 layers and a global average pool to
  reach the decision, when the pooled-feature route is one `Linear` from the logits. Also
  burns a channel and entangles this arm with `mask_channel`.
- **CoordConv-style original-frame coordinate channels.** Two channels holding each pixel's
  normalized coordinate *in the source frame*: the per-pixel step encodes scale, and BN
  cannot flatten it. Genuinely elegant, and it supplies position as well. Rejected for now
  because it conflates scale with position in one arm, and position is its own hypothesis
  (depth correlates with image y) that deserves its own measurement.
- **Scale-preserving crop** (fixed px-per-frame-px, pad to 448, no fit-to-canvas). The
  honest version of "keep the real size". Rejected: long edges run 12–505 px, so this clips
  the large instruments and hands the small ones 12x12 of real detail on a 448 canvas —
  discarding the upsampling that produced the 448 gain in the first place.
- **Capped scale**, `min(size / max(h, w), 4)`. The middle road, and defensible. Rejected as
  a first step because it makes resolution and scale a single variable, so a result is not
  attributable to either.

## 8. If this gets built

1. `requirements_scale.md` — the raw ask, pasted unedited. Without it `spec-conformance` is
   instructed to BLOCK.
2. `spec_cropcls.md` Step 11, with its own DoD item.
3. `spec-conformance` in `mode: preflight`. No code until BLOCK findings clear.
4. Code, then `/code-review high` on the diff.
5. `spec-conformance` (`mode: postflight`) and `claim-verifier` in one message.

Rank arms by `classaware_f1` from `HungarianInstanceEvaluator` with `score_mode=detector`.
**Crop top-1 is a training monitor only** — it scores only `IoU >= 0.5 AND runner_up < 0.2`
crops, and it has been retracted as a ranking signal more than once. Seed variance on
`classaware_f1` is 0.82 points (`cropcls-results.md` section 5), so a gap below ~1.6 points
is not rankable; given section 3's expectation of a small effect, decide up front whether one
seed per arm can answer the question at all.

The control is the same config with `scale_feats: false` and its own `output_dir`, launched
from the same file — not a separate config, or trunk / crop size / LR drift silently.

## Appendix: the probe

Read-only, no run artifacts, ~20 s. Needs `PYTHONPATH=.` and the sibling venv per `CLAUDE.md`:

```bash
cd /home/janick.bilang/dev/MaskDINO-Unet
PYTHONPATH=. /home/janick.bilang/dev/MaskDINO/.venv/bin/python scale_probe.py
```

```python
import glob, json, zlib
from pathlib import Path
import numpy as np, torch
from PIL import Image

root = Path("crops/train")
names = json.loads((root / "meta.json").read_text())["thing_classes"]
idx = {n: i for i, n in enumerate(names)}

rows = []
for p in sorted(glob.glob(str(root / "*" / "*.png"))):
    p = Path(p); c = p.parent.name
    if c not in idx:
        continue
    a = np.array(Image.open(p))                       # full decode: alpha is the mask
    h, w = a.shape[:2]
    rows.append((idx[c], w, h, int((a[..., 3] > 127).sum()), p.name.rsplit("_p", 1)[0]))

y  = torch.tensor([r[0] for r in rows])
w  = np.array([r[1] for r in rows], float)
h  = np.array([r[2] for r in rows], float)
ar = np.array([r[3] for r in rows], float).clip(1)
hi, lo = np.maximum(w, h), np.minimum(w, h)
# frame-level split: every crop of a frame lands on the same side
val = torch.tensor([zlib.crc32(r[4].encode()) % 5 == 0 for r in rows])

X = torch.tensor(np.stack([
    np.log2(hi / 32), np.log2(lo / 32),
    np.log2(np.sqrt(ar) / 32), np.log2(ar / (hi * lo)),
], 1), dtype=torch.float32)

Xtr, ytr, Xva, yva = X[~val], y[~val], X[val], y[val]
mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

torch.manual_seed(0)
net = torch.nn.Linear(X.shape[1], len(names))
opt = torch.optim.AdamW(net.parameters(), lr=3e-2, weight_decay=1e-4)
for _ in range(3000):
    opt.zero_grad()
    torch.nn.functional.cross_entropy(net(Xtr), ytr).backward()
    opt.step()
with torch.no_grad():
    print("val top-1 from geometry alone:",
          f"{(net(Xva).argmax(1) == yva).float().mean():.3f}")
    print("majority class:",
          f"{yva.bincount(minlength=len(names)).max().item() / len(yva):.3f}")
```
