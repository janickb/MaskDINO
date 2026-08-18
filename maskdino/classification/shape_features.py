# ------------------------------------------------------------------------
# Pure-geometry mask utilities: no torch/model dependency, operates on a
# single instance's binary segmentation mask at its native resolution.
# ------------------------------------------------------------------------
from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np


@dataclass
class PrincipalAxis:
    centroid: tuple  # (x, y)
    angle: float  # radians, in [0, pi) - the axis is a line, not a direction
    major_length: float
    minor_length: float


def principal_axis(mask: np.ndarray) -> PrincipalAxis:
    """mask: (H, W) boolean/0-1 array for a single instance. Returns the
    instrument's major/minor axes via PCA over every foreground pixel.
    `angle` is normalized into [0, pi) since a principal axis is a line, not
    a direction - the 180-degree sign ambiguity is not resolved here.

    Uses all mask pixels rather than a single contour's boundary, so a mask
    split into disconnected fragments by an occluder (with both fragments
    still correctly assigned to the same instance) still contributes its
    full extent to the fit, instead of silently truncating to whichever
    fragment happens to be largest.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return PrincipalAxis(centroid=(0.0, 0.0), angle=0.0, major_length=0.0, minor_length=0.0)

    points = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute2(points, mean=None)[:2]
    centroid = (float(mean[0, 0]), float(mean[0, 1]))
    major_axis, minor_axis = eigenvectors[0], eigenvectors[1]

    angle = atan2(float(major_axis[1]), float(major_axis[0])) % np.pi

    centered = points - mean
    major_proj = centered @ major_axis
    minor_proj = centered @ minor_axis
    major_length = float(major_proj.max() - major_proj.min())
    minor_length = float(minor_proj.max() - minor_proj.min())

    return PrincipalAxis(centroid=centroid, angle=angle, major_length=major_length, minor_length=minor_length)


def rotate_to_canonical(img: np.ndarray, axis: PrincipalAxis, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Rotates `img` (H, W) or (H, W, C) about `axis.centroid` so the major
    axis becomes horizontal. Works on an RGB crop or a mask - pass the same
    `axis` to both to keep them in registration. Use `interpolation=
    cv2.INTER_NEAREST` for a boolean/label mask so rotation doesn't blend
    edge pixels into non-boolean values."""
    h, w = img.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D(axis.centroid, degrees(axis.angle), 1.0)
    return cv2.warpAffine(img, rot_matrix, (w, h), flags=interpolation)


def canonical_crop(img: np.ndarray, mask: np.ndarray, axis: PrincipalAxis, pad: int = 3) -> np.ndarray:
    """Rotates `img`/`mask` to canonical orientation (see `rotate_to_canonical`)
    and crops to the mask's bounding box plus `pad` pixels of margin on every
    side, keeping the original background - no resizing, so the object's
    pixel size (and thus scale information, e.g. a 2mm vs 3mm instrument) is
    preserved rather than normalized away. Returns an empty (0, 0, ...)
    array if the mask is empty.
    """
    rotated_img = rotate_to_canonical(img, axis)
    rotated_mask = rotate_to_canonical(mask.astype(np.uint8), axis, interpolation=cv2.INTER_NEAREST).astype(bool)

    ys, xs = np.where(rotated_mask)
    if len(ys) == 0:
        return rotated_img[:0, :0]
    h, w = rotated_img.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    return rotated_img[y0:y1, x0:x1]
