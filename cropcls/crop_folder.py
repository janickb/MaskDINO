"""Crops on disk: one folder per class, one RGBA PNG per crop.

The folder name gives the class NAME, so there is no sidecar mapping each crop to a label.
The class ORDER still comes from `meta.json`'s `thing_classes`, not from `sorted()` - see
`CropFolder`. `tools/extract_crops.py` writes the folder; this reads it.

Alpha carries the segmentation mask rather than a blacked-out RGB, which is what keeps
photometric jitter usable: it has to run on the raw crop *before* blackout, because contrast
blends toward the image mean and a blacked crop's mean is near zero.

Patches are stored at native resolution, never pre-letterboxed. `mask_dilate` has to stay a
real-pixel operation - dilating an already-resized mask by 8 px would mean a different real
radius for a large instrument than a small one, which is not the operation that was measured.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .augment import PhotometricJitter
from .config import CropClsConfig
from .crops import letterbox_crop, mask_tight_bbox, normalize_crop, rotate_k90

__all__ = [
    "CropFolder",
    "build_loader",
    "load_crop",
    "patch_bbox",
    "save_crop",
    "write_frame_crops",
]


def patch_bbox(mask: torch.Tensor, margin: int) -> tuple[int, int, int, int] | None:
    """The mask's tight box grown by `margin`, clipped to the frame."""
    tight = mask_tight_bbox(mask)
    if tight is None:
        return None
    H, W = mask.shape
    x0, y0, x1, y1 = tight
    return (max(0, x0 - margin), max(0, y0 - margin), min(W, x1 + margin), min(H, y1 + margin))


def save_crop(path: Path, rgb: torch.Tensor, mask: torch.Tensor) -> None:
    """(3,h,w) uint8 + (h,w) bool -> RGBA PNG. Lossless; a float frame would quantize."""
    if rgb.dtype != torch.uint8:
        raise TypeError(f"expected uint8 RGB, got {rgb.dtype}: PNG would quantize it")
    a = mask.numpy().astype(np.uint8) * 255
    Image.fromarray(np.dstack([rgb.permute(1, 2, 0).numpy(), a]), "RGBA").save(
        path, optimize=True
    )


def load_crop(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.array(Image.open(path).convert("RGBA"))
    rgb = torch.from_numpy(arr[..., :3].copy()).permute(2, 0, 1)
    return rgb, torch.from_numpy(arr[..., 3].copy()) > 127


def write_frame_crops(
    out: Path,
    stem: str,
    image: torch.Tensor,
    masks: torch.Tensor,
    asg,
    class_names: list[str],
    margin: int,
) -> list[dict]:
    """Write one frame's kept crops into `out/<class>/`. Returns a row per DETECTION.

    Split out of the extraction script so it is testable without a detector: everything that
    can go wrong here - which detections get a crop, what the filename is, what the row says
    about the ones that do not - is decidable from an Assignment and a mask tensor.
    """
    rows = []
    for k in range(len(asg)):
        row = {
            "frame": stem, "pred_index": k, "verdict": asg.verdict[k],
            "best_iou": round(float(asg.best_iou[k]), 4),
            "runner_up_iou": round(float(asg.runner_up_iou[k]), 4),
            "class_name": "", "crop": "",
        }
        if bool(asg.keep[k]):
            box = patch_bbox(masks[k], margin)
            if box is None:
                # A kept detection can still have an empty mask. It gets a row, not silence.
                row["verdict"] = "dropped_empty_box"
            else:
                x0, y0, x1, y1 = box
                cls = class_names[int(asg.labels[k])]
                (out / cls).mkdir(parents=True, exist_ok=True)
                rel = f"{cls}/{stem}_p{k:03d}.png"
                save_crop(out / rel, image[:, y0:y1, x0:x1], masks[k][y0:y1, x0:x1])
                row["class_name"], row["crop"] = cls, rel
        rows.append(row)
    return rows


class CropFolder(torch.utils.data.Dataset):
    """`root/<class_name>/*.png`. The folder gives the class NAME; `meta.json` gives the ORDER.

    The order must NOT come from `sorted()` of the directory names. For this data the two
    disagree: `thing_classes` is
    `[forcep01..04, scalpel01, hammer01, sharpspoon01, scarstick01]` while sorted() puts
    hammer before scalpel and scarstick before sharpspoon. The four forceps coincide, so a
    sorted order trains four classes correctly and silently mislabels the other four - which
    is exactly what it did: forceps scored 0.27-0.82 and the rest 0.00-0.03.
    """

    def __init__(
        self,
        root: str | Path,
        cfg: CropClsConfig,
        is_train: bool = True,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.cfg = cfg
        dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not dirs:
            raise FileNotFoundError(f"{self.root} has no class folders; run tools/extract_crops.py")

        meta_path = self.root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"{meta_path} is missing, so the class ORDER is unknown. It cannot be inferred "
                "from the directory names: sorted() and the dataset's thing_classes disagree "
                "for this data, and training on one order while evaluating against the other "
                "silently mislabels whichever classes the two orders disagree on. Re-extract "
                "with tools/extract_crops.py, which records thing_classes."
            )
        self.meta = json.loads(meta_path.read_text())
        self.class_names = list(self.meta["thing_classes"])
        idx = {n: i for i, n in enumerate(self.class_names)}
        unknown = sorted(d.name for d in dirs if d.name not in idx)
        if unknown:
            raise ValueError(
                f"{self.root} has class folders {unknown} that are not in the recorded label "
                f"space {self.class_names}. The crops were extracted against a different "
                "dataset; re-extract."
            )
        self.items: list[tuple[Path, int]] = [
            (f, idx[d.name]) for d in dirs for f in sorted(d.glob("*.png"))
        ]
        if not self.items:
            raise FileNotFoundError(f"{self.root} has class folders but no crops")
        self._check_margin()
        self.photometric = PhotometricJitter() if (is_train and cfg.photometric) else None
        self.rotations = list(cfg.rotations) if is_train else [0]
        # Own generator: the global RNG is also seeded for model init, so drawing rotations
        # from it would couple augmentation to unrelated RNG consumption.
        self.g = torch.Generator().manual_seed(seed + 7919)

    def _check_margin(self) -> None:
        """Closing is the only op that needs the stored border, and it needs `margin >= dilate`.

        Measured: plain dilation is exact at `margin=0`, because every mask pixel is inside its
        own tight box and max-pool pads with -inf. Closing is different - `close_mask` is
        `~dilate_mask(~d, r)`, and the pad makes the window edge read as foreground during the
        erode, so foreground leaks inward by up to `radius`. In the full-frame path the window
        is `tight +- 2d` so the leak stops outside the box; here the window is the stored patch,
        so it is invisible only when `margin >= dilate`. Silent otherwise, hence the raise.
        """
        cfg = self.cfg
        if not (cfg.mask_close and cfg.mask_dilate > 0):
            return
        margin = self.meta["margin"]
        if cfg.mask_dilate > margin:
            raise ValueError(
                f"mask_close with mask_dilate={cfg.mask_dilate} needs margin >= {cfg.mask_dilate} "
                f"but these crops were stored with margin={margin}. The closing window would be "
                "truncated at the patch edge and the crop would quietly stop matching the live "
                "path. Re-extract with a larger margin."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        rgb, mask = load_crop(path)
        cfg = self.cfg
        tight = mask_tight_bbox(mask)
        if tight is None:
            raise ValueError(f"{path} has an empty mask; re-extract")
        # Same call as data.py:221-231, including box= the UNDILATED tight box. A passed box
        # overrides letterbox_crop's dilated one (crops.py:150), so dilation grows the mask
        # inside this box but never widens it. The live path does that; match it.
        out = letterbox_crop(
            rgb.float(), mask, cfg.crop_size, box=tight,
            photometric=self.photometric, mask_only=cfg.mask_only, blackout=cfg.blackout,
            dilate=cfg.mask_dilate, close=cfg.mask_close,
        )
        if out is None:
            raise ValueError(f"{path} produced no crop; re-extract")
        deg = self.rotations[int(torch.randint(len(self.rotations), (), generator=self.g))]
        crop = rotate_k90(out[0], deg)
        return normalize_crop(crop, cfg.pixel_mean, cfg.pixel_std), label


def build_loader(cfg: CropClsConfig, root: str | Path, is_train: bool = True, seed: int = 0):
    ds = CropFolder(root, cfg, is_train=is_train, seed=seed)
    if is_train and len(ds) < cfg.crops_per_batch:
        raise ValueError(
            f"{len(ds)} crops but crops_per_batch={cfg.crops_per_batch} with drop_last: the "
            "loader would yield nothing and the training loop would spin silently"
        )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.crops_per_batch,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )
