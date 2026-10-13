# ------------------------------------------------------------------------
# Photometric augmentation for the reclassify phase.
#
# build_photometric_augmentations is the config-driven replacement for the four
# hardcoded augs that used to sit at the tail of build_transform_gen. Its defaults
# (maskdino/config.py, INPUT.PHOTOMETRIC) reproduce those four exactly - same
# augs, same order, same RNG draws - so a config that sets nothing trains the
# pipeline it trained before.
#
# The point of making it configurable is the appearance-dose A/B: phase-2 trains
# on ~2 effective appearances per class (the side-A/B renders) and is evaluated on
# real photographs, so what training lacks is lighting/finish variety, not geometry.
# How hard to push that is an empirical question, hence arms.
#
# The two added axes (sensor noise, grayscale) are both strictly photometric,
# so each is wrapped in detectron2's ColorTransform, whose apply_coords and
# apply_segmentation are already identity. That is deliberate: it makes it
# impossible for these to desync the image from its masks the way a geometric
# transform can (see the FixedSizeCrop seg_pad_value note in
# coco_instance_new_baseline_dataset_mapper.build_transform_gen).
# ------------------------------------------------------------------------
import numpy as np
from detectron2.data import transforms as T
from fvcore.transforms.transform import NoOpTransform

__all__ = [
    "RandomGaussianNoise",
    "RandomGrayscale",
    "build_photometric_augmentations",
]

# Rec.601 luma weights, in RGB order; reversed for BGR input.
_LUMA_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _luma_weights(image_format):
    return _LUMA_RGB if image_format != "BGR" else _LUMA_RGB[::-1]


class RandomGaussianNoise(T.Augmentation):
    """Additive per-pixel sensor noise, std ~ U(std_range) in uint8 units. The
    probability lives here rather than in a wrapping T.RandomApply so that the
    augmentation list the mapper logs at startup names its own parameters - an arm's
    log has to be auditable after the fact."""

    def __init__(self, std_range, prob=1.0):
        super().__init__()
        self.std_range = tuple(float(s) for s in std_range)
        self.prob = float(prob)

    def get_transform(self, image):
        if np.random.rand() >= self.prob:
            return NoOpTransform()
        std = np.random.uniform(*self.std_range)
        if std <= 0:
            return NoOpTransform()
        # The noise field is drawn inside get_transform, not inside the op, so the
        # transform stays deterministic once created - re-applying a TransformList
        # (as the per-instance annotation loop does) must not redraw it.
        noise = np.random.normal(0.0, std, size=image.shape).astype(np.float32)
        return T.ColorTransform(lambda img, n=noise: _add_noise(img, n))

    def __repr__(self):
        return f"RandomGaussianNoise(std_range={self.std_range}, prob={self.prob})"


def _add_noise(img, noise):
    out = img.astype(np.float32) + noise[: img.shape[0], : img.shape[1]]
    return np.clip(out, 0, 255).astype(img.dtype)


class RandomGrayscale(T.Augmentation):
    """Full desaturation to luma, broadcast back to 3 channels. Applied whole-frame
    (RandomApply gates it), as a hard case of the color-constancy the real photos
    break."""

    def __init__(self, image_format, prob=1.0):
        super().__init__()
        self.image_format = image_format
        self.prob = float(prob)

    def get_transform(self, image):
        if np.random.rand() >= self.prob:
            return NoOpTransform()
        weights = _luma_weights(self.image_format)
        return T.ColorTransform(lambda img, w=weights: _to_gray(img, w))

    def __repr__(self):
        return f"RandomGrayscale(image_format={self.image_format!r}, prob={self.prob})"


def _to_gray(img, weights):
    luma = (img.astype(np.float32) * weights).sum(axis=2, keepdims=True)
    return np.clip(np.repeat(luma, 3, axis=2), 0, 255).astype(img.dtype)


def build_photometric_augmentations(cfg):
    """The photometric tail of build_transform_gen, from cfg.INPUT.PHOTOMETRIC.

    The four classic augs are always appended (even at a no-op range) so that the
    default config is RNG-identical to the hardcoded version it replaced. The two
    new axes are appended only when their probability is > 0, so an arm that does
    not use them consumes no draws for them either.

    There is deliberately no global blur axis. It existed to close a measured
    synthetic-vs-real sharpness gap (renders 198 vs real 191/165, see the realism
    arm's header), but phase-2 trains on real photographs, whose camera defocus is
    already in the data - and the residual real-session-to-real-session spread only
    justifies sigma <= ~0.3, which is numerically a no-op anyway (at sigma 0.3
    cv2's derived 3x3 kernel keeps weight 0.985 on the centre pixel, so the uint8
    result is unchanged). Per-instance defocus survives as
    INPUT.COPY_PASTE.INSTANCE_JITTER.BLUR_SIGMA, which exists for a different
    reason: pasted crops are softened by their own rotation/resampling, and that
    mismatch is independent of how the source frames were produced.
    """
    p = cfg.INPUT.PHOTOMETRIC
    augs = [
        T.RandomBrightness(*p.BRIGHTNESS),
        T.RandomContrast(*p.CONTRAST),
        T.RandomSaturation(*p.SATURATION),
        T.RandomLighting(p.LIGHTING),
    ]
    if p.NOISE_PROB > 0:
        augs.append(RandomGaussianNoise(p.NOISE_STD, prob=p.NOISE_PROB))
    if p.GRAYSCALE_PROB > 0:
        augs.append(RandomGrayscale(cfg.INPUT.FORMAT, prob=p.GRAYSCALE_PROB))
    return augs
