# New Test runs to show capability of Phase 2 Augmentation

## Goal
Run a control for the phase-2 training run, plus arms with different configurations.

## Data
Input data:

Training
- /home/janick.bilang/training/images_babylambda1-2080/train_phase2_set_ab_mm

Validation
- /home/janick.bilang/training/images_babylambda1-2080/val_set_ab_mm

## Experiments to run

Each block below is the arm's own config delta, in the exact key paths the trainer
reads: per-instance appearance jitter under `INPUT.COPY_PASTE.INSTANCE_JITTER`
(applies to pasted crops only), frame-level augmentation under `INPUT.PHOTOMETRIC`.
Every range is a two-element `[lo, hi]` list - `[1.0, 1.0]` is the no-op.

### Control

```yaml
INPUT:
  COPY_PASTE:
    INSTANCE_JITTER:
      BRIGHTNESS:  [1.0, 1.0]
      CONTRAST:    [1.0, 1.0]
      SATURATION:  [1.0, 1.0]
      HUE_DEGREES: 0.0
  PHOTOMETRIC:
    BRIGHTNESS: [1.0, 1.0]
    CONTRAST:   [1.0, 1.0]
    SATURATION: [1.0, 1.0]
    NOISE_PROB: 0.0
    NOISE_STD:  [0.0, 0.0]
```

### Low

```yaml
INPUT:
  COPY_PASTE:
    INSTANCE_JITTER:
      BRIGHTNESS:  [1.0, 1.0]
      CONTRAST:    [1.0, 1.0]
      SATURATION:  [1.0, 1.0]
      HUE_DEGREES: 0.0
  PHOTOMETRIC:
    BRIGHTNESS: [0.85, 1.15]
    CONTRAST:   [0.85, 1.15]
    SATURATION: [0.85, 1.15]
    NOISE_PROB: 0.25
    NOISE_STD:  [0.0, 1.0]
```

### Medium

```yaml
INPUT:
  COPY_PASTE:
    INSTANCE_JITTER:
      BRIGHTNESS:  [1.0, 1.0]
      CONTRAST:    [1.0, 1.0]
      SATURATION:  [1.0, 1.0]
      HUE_DEGREES: 0.0
  PHOTOMETRIC:
    BRIGHTNESS: [0.75, 1.25]
    CONTRAST:   [0.75, 1.25]
    SATURATION: [0.75, 1.25]
    NOISE_PROB: 0.25
    NOISE_STD:  [0.0, 1.0]
```

### Object_Jitter_only

```yaml
INPUT:
  COPY_PASTE:
    INSTANCE_JITTER:
      BRIGHTNESS:  [0.85, 1.15]
      CONTRAST:    [0.85, 1.15]
      SATURATION:  [0.85, 1.15]
      HUE_DEGREES: 4.0
  PHOTOMETRIC:
    BRIGHTNESS: [1.0, 1.0]
    CONTRAST:   [1.0, 1.0]
    SATURATION: [1.0, 1.0]
    NOISE_PROB: 0.0
    NOISE_STD:  [0.0, 0.0]
```

### Realistic (Measured) - TBD, DO NOT DO NOW

```yaml
INPUT:
  COPY_PASTE:
    INSTANCE_JITTER:
      BRIGHTNESS:  [0.85, 1.15]
      CONTRAST:    [0.85, 1.15]
      SATURATION:  [0.85, 1.15]
      HUE_DEGREES: 4.0
  PHOTOMETRIC:
    BRIGHTNESS: [0.75, 1.25]
    CONTRAST:   [0.75, 1.25]
    SATURATION: [0.75, 1.25]
    NOISE_PROB: 0.25
    NOISE_STD:  [0.0, 1.0]
```
