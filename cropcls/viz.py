"""Step-4 gate: render the crops the classifier will actually be trained on.

Writes one contact sheet per frame batch with three views per instance - the raw crop,
the blacked-out crop, and the predicted mask overlaid on the original frame - plus the
assignment verdict, so the IoU floor and runner-up ceiling can be calibrated by eye
before any GPU time is spent on training. The *dropped* instances are rendered too:
if that pile does not look genuinely ambiguous, the thresholds are wrong.

A trained classifier can be passed in, and then each row grows a 4th view: the Grad-CAM
of the predicted class (`cropcls/gradcam.py`). `write_gradcam_sheets` is the other half -
one sheet per GT class for an `--eval-only` run, errors first.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from .assign import ASSIGNED, assign_gt_classes
from .crops import letterbox_crop, mask_tight_bbox, to_model_input
from .data import CropPipeline
from .gradcam import CropGradCAM, cam_in_chunks

__all__ = ["cam_overlay", "denorm_crop", "write_contact_sheets", "write_gradcam_sheets"]

# Right/wrong borders on the Grad-CAM sheets. The green is _VERDICT_COLOR["assigned"] and the
# red _VERDICT_COLOR["dropped_contaminated"], so one palette reads across both sheet kinds.
_RIGHT = "#2e7d32"
_WRONG = "#c62828"

_VERDICT_COLOR = {
    "assigned": "#2e7d32",
    "dropped_low_iou": "#e65100",
    "dropped_contaminated": "#c62828",
    "dropped_no_gt": "#6a1b9a",
}


def _to_uint8(chw: torch.Tensor) -> np.ndarray:
    return chw.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


def _overlay(frame: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    """Predicted mask tinted onto the full original frame, so mask errors are visible."""
    img = frame.clamp(0, 255).float().clone()
    tint = torch.tensor([0.0, 255.0, 255.0], device=img.device).view(3, 1, 1)
    m = mask.unsqueeze(0).float()
    img = img * (1 - alpha * m) + tint * (alpha * m)
    return _to_uint8(img)


def denorm_crop(crop: torch.Tensor, pixel_mean, pixel_std) -> np.ndarray:
    """A model-input crop back to an (S, S, 3) uint8 image.

    The one copy of this: `tools/viz_misclassified.py` imports it rather than keeping its own,
    because a renderer that de-normalizes with different statistics than the pipeline used
    does not raise - it just draws a differently tinted instrument.

    A `mask_channel` run carries a 4th channel that was never normalized by these 3-element
    RGB statistics, so it is dropped rather than de-normalized.
    """
    m = torch.as_tensor(pixel_mean).view(-1, 1, 1)
    s = torch.as_tensor(pixel_std).view(-1, 1, 1)
    rgb = crop.detach().cpu()[: len(pixel_mean)]
    return (rgb * s + m).clamp(0, 255).byte().permute(1, 2, 0).numpy()


def cam_overlay(rgb: np.ndarray, cam: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
    """A [0, 1] Grad-CAM blended over an (S, S, 3) uint8 crop, as uint8.

    `inferno`, never `jet`: this repo's stance on rainbow colormaps is already written down in
    `tools/instrument_landing_stats.py` ("no rainbow/jet, which invents false structure"), and
    the detector-side attention panel on branch 49 uses inferno too.

    Returns a new array; `rgb` is not modified.
    """
    heat = cam.detach().float().cpu().numpy() if torch.is_tensor(cam) else np.asarray(cam)
    if heat.shape != rgb.shape[:2]:
        raise ValueError(f"cam {heat.shape} does not match crop {rgb.shape[:2]}")
    colour = matplotlib.colormaps["inferno"](np.clip(heat, 0.0, 1.0))[..., :3] * 255.0
    blended = (1.0 - alpha) * rgb.astype(np.float32) + alpha * colour
    return blended.clip(0, 255).astype(np.uint8)


def _model_input(cfg, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """The exact tensor the classifier is fed for one predicted mask.

    Rebuilt from cfg rather than reusing the sheet's illustrative `blackout=True` crop: a CAM
    has to explain what the model actually sees - dilation, `mask_only`, the mask channel - or
    it is a map of a different input than the prediction it is captioned with.
    """
    # box= is what data.py and crop_folder.py both pass, and letterbox_crop treats a passed
    # box as overriding the dilated one. Without it, mask_dilate > 0 widens the field of view
    # here but not in training, so the map would explain a crop the classifier never sees.
    tight = mask_tight_bbox(mask)
    if tight is None:
        return None
    out = letterbox_crop(
        image, mask, cfg.crop_size, box=tight,
        blackout=cfg.blackout, dilate=cfg.mask_dilate, close=cfg.mask_close,
        mask_only=cfg.mask_only,
    )
    if out is None:
        return None
    return to_model_input(out, cfg.pixel_mean, cfg.pixel_std, 0, cfg.mask_channel)


def _cam_panel(
    cam: CropGradCAM, cfg, image: torch.Tensor, mask: torch.Tensor, class_names: list[str]
) -> tuple[str, np.ndarray | None]:
    """(title, image) for the Grad-CAM column of one contact-sheet row."""
    x = _model_input(cfg, image, mask)
    if x is None:
        return "grad-cam", None
    device = next(cam.model.parameters()).device
    heat, logits = cam(x.unsqueeze(0).to(device))
    prob = torch.softmax(logits[0].float(), 0)
    k = int(prob.argmax())
    rgb = denorm_crop(x, cfg.pixel_mean, cfg.pixel_std)
    return f"grad-cam -> {class_names[k]} {float(prob[k]):.2f}", cam_overlay(rgb, heat[0])


def write_contact_sheets(
    pipeline: CropPipeline,
    out_dir: str | Path,
    frame_indices: list[int],
    max_instances_per_sheet: int = 8,
    cam: CropGradCAM | None = None,
) -> list[Path]:
    """One PNG per frame. Returns the paths written.

    `cam` adds a 4th column, the Grad-CAM of the predicted class. It is optional because the
    step-4 gate runs before any classifier checkpoint exists - without one the sheet is
    exactly the three columns it always was.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cfg = pipeline.cfg

    for fi in frame_indices:
        mapped = pipeline.mapper(copy.deepcopy(pipeline.dicts[fi]))
        image = mapped["image"].float()
        inst = mapped["instances"]
        gt_masks = inst.gt_masks
        if not torch.is_tensor(gt_masks):
            gt_masks = gt_masks.tensor

        det = pipeline.detector.detect([mapped["image"]])[0]
        pred = det.masks.cpu()
        a = assign_gt_classes(
            pred,
            gt_masks,
            inst.gt_classes,
            iou_floor=cfg.iou_floor,
            runner_up_ceiling=cfg.runner_up_ceiling,
        )

        n = min(len(a), max_instances_per_sheet)
        if n == 0:
            continue
        n_col = 4 if cam is not None else 3
        fig, axes = plt.subplots(
            n, n_col, figsize=(3 * n_col, 3.1 * n), squeeze=False
        )
        for row in range(n):
            raw = letterbox_crop(image, pred[row], cfg.crop_size, blackout=False)
            black = letterbox_crop(image, pred[row], cfg.crop_size, blackout=True)
            verdict = a.verdict[row]
            colour = _VERDICT_COLOR.get(verdict, "#424242")
            label = (
                pipeline.class_names[int(a.labels[row])]
                if a.keep[row]
                else "-"
            )

            panels = [
                ("raw crop", _to_uint8(raw[0]) if raw else None),
                ("blacked (model input)", _to_uint8(black[0]) if black else None),
                ("mask on frame", _overlay(image, pred[row])),
            ]
            if cam is not None:
                panels.append(_cam_panel(cam, cfg, image, pred[row], pipeline.class_names))

            for col, (title, arr) in enumerate(panels):
                ax = axes[row][col]
                if arr is None:
                    ax.text(0.5, 0.5, "empty mask\n(dropped)", ha="center", va="center")
                else:
                    ax.imshow(arr)
                ax.set_xticks([])
                ax.set_yticks([])
                # The first three titles are column headers and belong on row 0 only. The
                # grad-cam title names THIS row's prediction, so it is set on every row -
                # otherwise row 0's class is printed over every instrument on the sheet.
                if row == 0 or (cam is not None and col == len(panels) - 1):
                    ax.set_title(title, fontsize=9)
            axes[row][0].set_ylabel(
                f"{label}\nIoU {a.best_iou[row]:.2f}\n2nd {a.runner_up_iou[row]:.2f}\n"
                f"score {det.scores[row]:.2f}",
                fontsize=8,
                color=colour,
            )
            for s in axes[row][0].spines.values():
                s.set_color(colour)
                s.set_linewidth(3)

        counts = a.counts()
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        handles = [
            mpatches.Patch(color=c, label=k)
            for k, c in _VERDICT_COLOR.items()
            if k in counts
        ]
        fig.suptitle(
            f"frame {fi}  {Path(str(mapped.get('file_name',''))).name}\n"
            f"gt={len(inst)}  pred={len(a)}  {summary}   "
            f"(floor {cfg.iou_floor}, runner-up ceiling {cfg.runner_up_ceiling})",
            fontsize=10,
        )
        if handles:
            fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
        fig.tight_layout(rect=(0.02, 0.03, 1, 0.95))
        path = out / f"frame_{fi:05d}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        written.append(path)
    return written


def _keep_best(bucket: list[dict], item: dict, limit: int) -> None:
    """Bounded top-`limit` by `item["p"]`, evicting the weakest.

    Streaming, because "the most confident over the whole val pass" must not mean holding the
    whole val pass - 1176 crops at 3x224x224 float32 is ~700 MB. The ranking key is each crop's
    own probability, known on arrival, so the bounded version is exact. Stopping the scan once
    every class had filled up is NOT: it keeps whichever errors appear early and never sees the
    confident ones later in the set.
    """
    bucket.append(item)
    if len(bucket) > limit:
        bucket.pop(min(range(len(bucket)), key=lambda i: bucket[i]["p"]))


def write_gradcam_sheets(
    model,
    pipeline: CropPipeline,
    device,
    out_dir: str | Path,
    per_class: int = 8,
    max_frames: int | None = None,
    layer: str | None = None,
    per_sheet: int = 24,
    items_per_row: int = 3,
) -> list[Path]:
    """One Grad-CAM sheet per GT class, errors first. Returns the paths written.

    Each item is a pair of panels - the crop as the model saw it, and the Grad-CAM of the
    **predicted** class over it - bordered green when the prediction is right and red when it
    is wrong. Per class: the most-confident errors first, then the most-confident correct ones
    fill the remaining slots, so a class with no errors still shows what a healthy map looks
    like.

    This needs its own pass over the val set: `train_cropcls.evaluate()` runs under `no_grad`
    and a CAM needs the backward. Scored on matched masks only (the crops that pass the IoU
    floor and runner-up ceiling), i.e. the same population as the `val_top1` monitor - so it
    diagnoses the CLASSIFIER, and masks the detector missed never appear here.
    """
    out = Path(out_dir)
    if per_class <= 0:
        return []
    out.mkdir(parents=True, exist_ok=True)

    cam = CropGradCAM(model, layer=layer)
    cfg = pipeline.cfg
    n = len(pipeline) if max_frames is None else min(max_frames, len(pipeline))
    step = cfg.ims_per_batch
    best: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    seen: dict[tuple[str, bool], int] = defaultdict(int)

    for start in range(0, n, step):
        batch = pipeline.build_batch(list(range(start, min(start + step, n))), rotations=[0])
        if len(batch) == 0:
            continue
        heat, logits = cam_in_chunks(cam, batch.crops, device, cfg.crops_per_batch)
        prob = torch.softmax(logits.float(), dim=1)
        p_pred, pred = prob.max(1)
        for i in range(len(batch)):
            gt, pi = int(batch.labels[i]), int(pred[i])
            wrong = pi != gt
            key = (pipeline.class_names[gt], wrong)
            seen[key] += 1
            r = batch.records[i]
            _keep_best(
                best[key],
                {
                    "p": float(p_pred[i]),
                    "pred": pipeline.class_names[pi],
                    "wrong": wrong,
                    "iou": r.best_iou,
                    "frame": Path(r.file_name).stem or f"idx{r.frame_index}",
                    "crop": denorm_crop(batch.crops[i], cfg.pixel_mean, cfg.pixel_std),
                    "cam": heat[i].cpu(),
                },
                per_class,
            )

    written: list[Path] = []
    for name in sorted({k[0] for k in seen}):
        wrong = sorted(best.get((name, True), []), key=lambda d: -d["p"])
        right = sorted(best.get((name, False), []), key=lambda d: -d["p"])
        items = (wrong + right)[:per_class]
        if not items:
            continue
        # Counted over every crop of the class, not over the kept ones: `wrong` is capped at
        # per_class, so printing len(wrong) as the error count would understate it silently.
        n_wrong = seen[(name, True)]
        n_total = n_wrong + seen[(name, False)]

        for page, s0 in enumerate(range(0, len(items), per_sheet)):
            chunk = items[s0 : s0 + per_sheet]
            rows = (len(chunk) + items_per_row - 1) // items_per_row
            fig, axes = plt.subplots(
                rows, 2 * items_per_row,
                figsize=(4.6 * items_per_row, 2.9 * rows), squeeze=False,
            )
            for ax in axes.ravel():
                ax.axis("off")
            for i, item in enumerate(chunk):
                row, col = divmod(i, items_per_row)
                colour = _WRONG if item["wrong"] else _RIGHT
                views = (
                    (item["crop"], f"{item['frame']}\nIoU {item['iou']:.2f}"),
                    (
                        cam_overlay(item["crop"], item["cam"]),
                        f"-> {item['pred']} {item['p']:.2f}",
                    ),
                )
                for off, (arr, title) in enumerate(views):
                    ax = axes[row][2 * col + off]
                    ax.imshow(arr)
                    ax.set_title(title, fontsize=7, color=colour)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis("on")
                    for sp in ax.spines.values():
                        sp.set_color(colour)
                        sp.set_linewidth(2)
            fig.suptitle(
                f"GT {name}: grad-cam of the PREDICTED class on {len(items)} of "
                f"{n_total} matched crops ({n_wrong} wrong), most confident first"
                + (f"  (page {page + 1})" if len(items) > per_sheet else ""),
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            path = out / f"{name}{'' if len(items) <= per_sheet else f'_p{page + 1}'}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            written.append(path)
    return written
