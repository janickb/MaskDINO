# ------------------------------------------------------------------------
# Slight projective (homography) deformation.
#
# Motivation: every synthetic training render comes from one fixed overhead camera,
# while the real validation photos do not - a tool photographed from slightly off-axis
# is foreshortened, and no amount of in-plane rotation or photometric jitter produces
# that. This covers mild off-axis viewpoints without pretending to be a full 3D
# re-render.
#
# Deliberately moderate. Two constraints shape the design:
#
# 1. NO EMPTY CORNERS. The source quad is sampled strictly INSIDE the frame and
#    rectified onto the full canvas, rather than warping the frame and filling
#    whatever falls outside. A constant-filled corner would be exactly the kind of
#    learnable artifact the 128-gray FixedSizeCrop pad band already is.
#    Consequence: pulling all four corners inward also zooms in. Both opposite edges
#    move toward each other, so the effect compounds - measured over 400 draws, a
#    distortion_scale of 0.05 gives a mean LINEAR zoom of 1.053 (area 1.108), and
#    0.10 gives 1.112 (area 1.237). That is comparable to the low end of the scale
#    jitter the pipeline already applies, but it is systematic (always a zoom IN,
#    never out), which plain scale jitter is not - so an arm using this should be
#    compared against a control sharing its scale setting.
#
# 2. THE MASK MUST FOLLOW THE IMAGE EXACTLY. This is a geometric transform, so unlike
#    everything in photometric.py it CAN desync the two (cf. the FixedSizeCrop
#    seg_pad_value bug). apply_segmentation therefore warps the mask as float32 with
#    the same INTER_LINEAR kernel as the image and thresholds at 0.5 - the same
#    sub-pixel-accurate recipe copy_paste.py uses for its own rotation - rather than
#    detectron2's usual INTER_NEAREST, which would staircase the boundary away from
#    where the image content actually landed.
# ------------------------------------------------------------------------
import cv2
import numpy as np
from detectron2.data import transforms as T
from fvcore.transforms.transform import NoOpTransform, Transform

__all__ = ["PerspectiveTransform", "RandomPerspective"]


class PerspectiveTransform(Transform):
    """Applies a fixed 3x3 homography. Deterministic once constructed - a
    TransformList is re-applied once per annotation after the image, so a transform
    that re-randomized per call would scatter an instance away from its own pixels."""

    def __init__(self, matrix, w, h):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        return cv2.warpPerspective(
            img,
            self.matrix,
            (self.w, self.h),
            flags=interp if interp is not None else cv2.INTER_LINEAR,
            # The source quad lies inside the frame, so nothing outside is ever
            # sampled; REPLICATE only guards the boundary rounding case.
            borderMode=cv2.BORDER_REPLICATE,
        )

    def apply_coords(self, coords):
        coords = np.asarray(coords, dtype=np.float64)
        homogeneous = np.concatenate(
            [coords, np.ones((len(coords), 1), dtype=np.float64)], axis=1
        )
        projected = homogeneous @ np.asarray(self.matrix, dtype=np.float64).T
        # w can approach 0 only for points near the horizon of an extreme warp;
        # at these distortion scales it stays close to 1.
        w = np.where(np.abs(projected[:, 2:3]) < 1e-12, 1e-12, projected[:, 2:3])
        return projected[:, :2] / w

    def apply_segmentation(self, segmentation):
        warped = cv2.warpPerspective(
            segmentation.astype(np.float32),
            self.matrix,
            (self.w, self.h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return (warped > 0.5).astype(segmentation.dtype)

    def inverse(self):
        return PerspectiveTransform(np.linalg.inv(self.matrix), self.w, self.h)


class RandomPerspective(T.Augmentation):
    """With probability `prob`, rectify a randomly perturbed inner quad onto the full
    canvas. `distortion_scale` is the maximum inward displacement of each corner, as
    a fraction of the frame's width/height - 0.05 is a visible tilt, not a funhouse
    mirror. Each corner is drawn independently, so the result is a genuine projective
    change (a shared displacement would only be a zoom)."""

    def __init__(self, distortion_scale=0.05, prob=0.5):
        super().__init__()
        self.distortion_scale = float(distortion_scale)
        self.prob = float(prob)

    def get_transform(self, image):
        if self.distortion_scale <= 0 or np.random.rand() >= self.prob:
            return NoOpTransform()

        h, w = image.shape[:2]
        d = self.distortion_scale
        dx = np.random.uniform(0.0, d, size=4) * (w - 1)
        dy = np.random.uniform(0.0, d, size=4) * (h - 1)
        src = np.float32([
            [dx[0], dy[0]],                      # top-left, pulled in and down
            [w - 1 - dx[1], dy[1]],              # top-right
            [w - 1 - dx[2], h - 1 - dy[2]],      # bottom-right
            [dx[3], h - 1 - dy[3]],              # bottom-left
        ])
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        return PerspectiveTransform(cv2.getPerspectiveTransform(src, dst), w, h)

    def __repr__(self):
        return (
            f"RandomPerspective(distortion_scale={self.distortion_scale}, "
            f"prob={self.prob})"
        )
