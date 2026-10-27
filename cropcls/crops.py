"""Letterboxed, background-blacked instance crops.

Pure tensor ops - no detectron2, no model, no config object - so this is unit-testable
on its own (spec_cropcls.md section 6, step 2).

The crop is taken from the **mask's tight bounding box**, not from the detector's learned
`pred_boxes`. That keeps the crop and the blackout self-consistent: every non-zero pixel
of the returned crop is inside the mask by construction, which a learned box can violate
in either direction (clipping the mask, or padding it with slack).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "close_mask",
    "dilate_mask",
    "mask_tight_bbox",
    "letterbox_crop",
    "rotate_k90",
    "normalize_crop",
]


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Grow a (H, W) boolean mask by `radius` pixels (square structuring element).

    Separable: a square max-pool factors into a 1xk pass and a kx1 pass, turning O(k^2)
    per pixel into O(2k). At radius 16 on a 1024^2 mask that is the difference between
    ~1.1 billion comparisons and ~67 million - on CPU, between 16 s/iter and usable.
    Callers should also crop before dilating (see `letterbox_crop`).
    """
    if radius <= 0:
        return mask.bool()
    r = int(radius)
    k = 2 * r + 1
    x = mask[None, None].float()
    x = F.max_pool2d(x, kernel_size=(1, k), stride=1, padding=(0, r))
    x = F.max_pool2d(x, kernel_size=(k, 1), stride=1, padding=(r, 0))
    return x[0, 0].bool()


def close_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological closing: dilate by `radius`, then erode by `radius`.

    Fills holes and rejoins broken fragments **without growing the outer silhouette** -
    which separates "the detector's mask is fragmented" from "surrounding context helps".
    The predicted masks here are visibly broken (IoU 0.55-0.83, shafts rendered as dashed
    segments), so repair and context are two different hypotheses that plain dilation
    conflates.

    Erosion is the complement of the dilation of the complement. `max_pool2d` pads with
    -inf, so outside-frame is treated as foreground during the erode step and an object
    touching the border is not eaten away from it.
    """
    if radius <= 0:
        return mask.bool()
    d = dilate_mask(mask, radius)
    return ~dilate_mask(~d, radius)


def mask_tight_bbox(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Tight (x0, y0, x1, y1) half-open box of a (H, W) boolean mask.

    Returns None for a genuinely empty mask - callers must drop it rather than
    producing an all-black crop or a 0/0 NaN downstream.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a (H, W) mask, got shape {tuple(mask.shape)}")
    m = mask.bool()
    rows = torch.any(m, dim=1)
    cols = torch.any(m, dim=0)
    if not bool(rows.any()):
        return None
    y = torch.where(rows)[0]
    x = torch.where(cols)[0]
    return int(x[0]), int(y[0]), int(x[-1]) + 1, int(y[-1]) + 1


def letterbox_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    size: int,
    pad_value: float = 0.0,
    blackout: bool = True,
    dilate: int = 0,
    close: bool = False,
    box: tuple[int, int, int, int] | None = None,
    photometric=None,
    mask_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Crop to the mask's bbox, black out the background, uniform-scale and pad to square.

    Args:
        image: (3, H, W) float tensor in whatever range the caller uses (not normalized).
        mask:  (H, W) bool tensor, same H/W as `image`.
        size:  output edge length; the result is (3, size, size).
        blackout: zero every pixel outside the mask. False keeps the real background,
            which only the step-4 contact sheet wants (to show what the mask removed).
        close: use morphological closing (dilate then erode) instead of plain dilation,
            repairing the mask without growing the silhouette.
        dilate: grow the mask by this many pixels first. Applied before BOTH the bbox and
            the blackout, so the grown region is not clipped by a tight box.
        box: crop region override (e.g. a jittered bbox). Defaults to the mask's tight box.
        photometric: callable applied to the RAW crop before blackout. It must run
            pre-blackout - contrast blends toward the crop mean, and on a ~95%-black crop
            that mean is near zero, so post-blackout it would behave nothing like the
            frame-level op it replaces.
        mask_only: return the silhouette (255 inside the mask, 0 outside) instead of RGB.
            The shape-only ablation: if it scores near the RGB model, the task is shape
            matching and the RGB channels are carrying little.

    Returns:
        (crop, crop_mask) where crop is (3, size, size) and crop_mask is (size, size) bool,
        or None if the mask is empty or its bbox is degenerate.

    Scaling is uniform on the longest axis - never skewed - and the content is centred so
    that `rotate_k90` keeps it centred.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) image, got shape {tuple(image.shape)}")
    if mask.shape != image.shape[-2:]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match image {tuple(image.shape[-2:])}"
        )

    tight = mask_tight_bbox(mask)
    if dilate > 0:
        if tight is None:
            return None
        # Dilate only the neighbourhood we will crop, not the whole frame. Expanding the
        # tight box by `dilate` first makes this EXACT - every original mask pixel is
        # inside `tight`, so dilation by `dilate` reaches at most the expanded box - while
        # turning a 1024^2 op into one over roughly the instrument's own area.
        H, W = mask.shape
        # Closing needs 2*r of margin: the dilate reaches tight+r, and the erode then
        # reads an r-neighbourhood of that. Plain dilation only needs r.
        m = dilate * (2 if close else 1)
        x0 = max(0, tight[0] - m); y0 = max(0, tight[1] - m)
        x1 = min(W, tight[2] + m); y1 = min(H, tight[3] + m)
        op = close_mask if close else dilate_mask
        sub = op(mask[y0:y1, x0:x1], dilate)
        mask = torch.zeros_like(mask)
        mask[y0:y1, x0:x1] = sub
    tight = mask_tight_bbox(mask)
    if tight is None:
        return None
    x0, y0, x1, y1 = tight if box is None else box
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return None

    img = image[:, y0:y1, x0:x1].float()
    msk = mask[y0:y1, x0:x1].bool()
    if not bool(msk.any()):          # a jittered box can only ever grow, but be safe
        return None

    if photometric is not None:
        img = photometric(img)
    if mask_only:
        img = msk.unsqueeze(0).float().expand(3, -1, -1).mul(255.0).contiguous()

    # Black out first, so the subsequent bilinear resize bleeds black into the instrument
    # edge (benign, reads as antialiasing) rather than bleeding real background in.
    if blackout:
        img = img * msk.unsqueeze(0)

    scale = size / max(h, w)
    new_h = max(1, min(size, int(round(h * scale))))
    new_w = max(1, min(size, int(round(w * scale))))

    img = F.interpolate(
        img.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    msk = (
        F.interpolate(
            msk.unsqueeze(0).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .bool()
    )
    # Re-apply the resized mask so the resize leaves no soft halo outside the silhouette.
    if blackout:
        img = img * msk.unsqueeze(0)

    crop = img.new_full((3, size, size), float(pad_value))
    crop_mask = torch.zeros((size, size), dtype=torch.bool, device=img.device)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    crop[:, top : top + new_h, left : left + new_w] = img
    crop_mask[top : top + new_h, left : left + new_w] = msk
    return crop, crop_mask


def rotate_k90(x: torch.Tensor, degrees: int) -> torch.Tensor:
    """Rotate the last two dims by a multiple of 90 degrees.

    Lossless on a square canvas: a pure pixel permutation, no interpolation, so the crop
    and its mask stay exactly aligned. Applied after cropping, which is why one detector
    forward can yield four classifier samples (spec_cropcls.md step 5).
    """
    if degrees % 90 != 0:
        raise ValueError(f"only multiples of 90 are lossless, got {degrees}")
    k = (degrees // 90) % 4
    if k == 0:
        return x
    if x.shape[-1] != x.shape[-2]:
        raise ValueError(
            f"rotate_k90 needs a square canvas, got {tuple(x.shape[-2:])}"
        )
    return torch.rot90(x, k, dims=(-2, -1))


def normalize_crop(
    crop: torch.Tensor, pixel_mean: list[float], pixel_std: list[float]
) -> torch.Tensor:
    """(x - mean) / std with MaskDINO's 0-255 RGB statistics."""
    mean = torch.as_tensor(pixel_mean, dtype=crop.dtype, device=crop.device).view(-1, 1, 1)
    std = torch.as_tensor(pixel_std, dtype=crop.dtype, device=crop.device).view(-1, 1, 1)
    return (crop - mean) / std
