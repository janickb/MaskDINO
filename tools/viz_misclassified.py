#!/usr/bin/env python
"""List and render the val crops a trained classifier gets wrong.

    PYTHONPATH=. python tools/viz_misclassified.py --run runs/cropcls_phase1_seta

Writes `<run>/misclassified/crops.csv` plus one contact sheet per GT class. The rendered
image is the *model input* - blacked out, letterboxed, de-normalized - not the raw frame
patch, so what the sheet shows is all the evidence the classifier had.

Scored on matched masks only (the crops that pass the IoU floor and runner-up ceiling),
i.e. the same population as train_cropcls.py's `val_top1` monitor. That makes this a
diagnostic of the CLASSIFIER, not of the run's classaware_f1: masks the detector missed
never appear here.

Each panel is paired with the Grad-CAM of the predicted class (`cropcls/gradcam.py`) - where
the model looked to make the error. `--no-gradcam` renders the crops alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import maskdino  # noqa: F401  - registers the hdf5 datasets
from cropcls.config import load_config
from cropcls.data import CropPipeline, apply_truncated_filter, apply_val_stride
from cropcls.detector import FrozenSingleClassDetector
from cropcls.gradcam import CropGradCAM, cam_in_chunks
from cropcls.modeling import build_classifier
from cropcls.viz import cam_overlay, denorm_crop

COLS = [
    "frame", "pred_index", "gt_class", "pred_class", "p_pred", "p_gt",
    "runner_up_class", "p_runner_up", "margin", "best_iou", "runner_up_iou", "det_score",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir; its config.json is authoritative")
    ap.add_argument("--weights", default=None, help="default <run>/model_final.pth")
    ap.add_argument("--out", default=None, help="default <run>/misclassified")
    ap.add_argument("--frames", type=int, default=None, help="cap val frames (smoke runs)")
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--no-gradcam", action="store_true", help="crops only, no grad-cam panel")
    ap.add_argument("--gradcam-layer", default=None,
                    help="trunk submodule to read the grad-cam from; default the last stage")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def _run_config(run: Path) -> tuple[Path, bool]:
    """The run's own config, and whether it is an --eval-only record.

    An --eval-only run writes `config_eval.json` and never a `config.json`
    (train_cropcls.py:188), so insisting on the latter locks this tool out of exactly the
    directories worth inspecting.
    """
    train = run / "config.json"
    if train.is_file():
        return train, False
    ev = run / "config_eval.json"
    if ev.is_file():
        return ev, True
    raise FileNotFoundError(f"{run} has neither config.json nor config_eval.json")


@torch.no_grad()
def collect(model, pipe, device, max_frames=None, cam: CropGradCAM | None = None):
    """Every misclassified crop, with its probabilities and the crop as the model saw it.

    `cam` adds the Grad-CAM of the predicted class. It runs on the wrong crops only, and
    re-enables grad from inside this no_grad scope - which is what `CropGradCAM` does and why
    the decorator can stay on the scoring forward.
    """
    n = len(pipe) if max_frames is None else min(max_frames, len(pipe))
    step = pipe.cfg.ims_per_batch
    wrong, total = [], 0

    for start in range(0, n, step):
        batch = pipe.build_batch(list(range(start, min(start + step, n))), rotations=[0])
        if len(batch) == 0:
            continue
        x = batch.crops.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        prob = F.softmax(logits.float(), dim=1).cpu()
        pred = prob.argmax(1)
        total += len(batch)

        wrong_idx = (pred != batch.labels).nonzero(as_tuple=True)[0].tolist()
        # targets=pred, not the CAM's own argmax: the caption comes from the fp16 autocast
        # forward above and the CAM pass is fp32, so on a near-tie the two can disagree and
        # the panel would explain one class under the other's label.
        heat = (
            cam_in_chunks(
                cam, x[wrong_idx], x.device, pipe.cfg.crops_per_batch,
                targets=pred[wrong_idx],
            )[0]
            if (cam is not None and wrong_idx)
            else None
        )
        for j, i in enumerate(wrong_idx):
            p = prob[i]
            gt, pi = int(batch.labels[i]), int(pred[i])
            # Runner-up behind the prediction: with pred != gt it is often the GT class,
            # and when it is not, the error is not a near-miss between two look-alikes.
            order = p.argsort(descending=True)
            second = int(order[1])
            r = batch.records[i]
            wrong.append({
                "frame": Path(r.file_name).stem or f"idx{r.frame_index}",
                "pred_index": r.pred_index,
                "gt_class": pipe.class_names[gt],
                "pred_class": pipe.class_names[pi],
                "p_pred": round(float(p[pi]), 4),
                "p_gt": round(float(p[gt]), 4),
                "runner_up_class": pipe.class_names[second],
                "p_runner_up": round(float(p[second]), 4),
                "margin": round(float(p[pi] - p[second]), 4),
                "best_iou": round(r.best_iou, 4),
                "runner_up_iou": round(r.runner_up_iou, 4),
                "det_score": round(r.score, 4),
                "_crop": denorm_crop(batch.crops[i], pipe.cfg.pixel_mean, pipe.cfg.pixel_std),
                "_cam": None if heat is None else heat[j],
            })
    return wrong, total


def sheets(wrong, out: Path, per_sheet: int, cols: int = 6) -> list[Path]:
    """One contact sheet per GT class, most-confident error first.

    With a Grad-CAM on each record the panels come in pairs (crop, cam), so a row holds
    cols // 2 items rather than cols.
    """
    by_gt = defaultdict(list)
    for w in wrong:
        by_gt[w["gt_class"]].append(w)

    per_item = 2 if any(w.get("_cam") is not None for w in wrong) else 1
    per_row = max(1, cols // per_item)

    written = []
    for gt, items in sorted(by_gt.items()):
        items.sort(key=lambda w: -w["p_pred"])
        for page, s in enumerate(range(0, len(items), per_sheet)):
            chunk = items[s : s + per_sheet]
            rows = (len(chunk) + per_row - 1) // per_row
            fig, axes = plt.subplots(
                rows, per_row * per_item,
                figsize=(2.3 * per_row * per_item, 2.75 * rows), squeeze=False,
            )
            for ax in axes.ravel():
                ax.axis("off")
            for i, w in enumerate(chunk):
                row, col = divmod(i, per_row)
                crop = w.pop("_crop")
                heat = w.pop("_cam", None)
                caption = (
                    f"-> {w['pred_class']}  {w['p_pred']:.2f}\n"
                    f"p(gt)={w['p_gt']:.2f}  IoU {w['best_iou']:.2f}"
                )
                views = [(crop, caption)]
                if heat is not None:
                    views.append((cam_overlay(crop, heat), "grad-cam (predicted class)"))
                for off, (arr, title) in enumerate(views):
                    ax = axes[row][per_item * col + off]
                    ax.imshow(arr)
                    ax.set_title(title, fontsize=7, color="#c62828")
                    for sp in ax.spines.values():
                        sp.set_color("#c62828")
                        sp.set_linewidth(2)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis("on")
            fig.suptitle(
                f"GT {gt}: {len(items)} misclassified crops"
                + (f"  (page {page + 1})" if len(items) > per_sheet else ""),
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            path = out / f"{gt}{'' if len(items) <= per_sheet else f'_p{page + 1}'}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            written.append(path)
    return written


def main():
    args = parse_args()
    run = Path(args.run)
    cfg_path, is_eval = _run_config(run)
    cfg = load_config(cfg_path)
    # An EVAL record's `weights` is the checkpoint that was actually scored, so it is the right
    # default. A TRAINING config's `weights` is only the run's starting point - possibly an
    # earlier phase - and defaulting to that would silently inspect the wrong model, which is
    # the trap train_cropcls.py:175-186 exists to prevent.
    fallback = cfg.weights if (is_eval and cfg.weights) else str(run / "model_final.pth")
    cfg.weights = args.weights or fallback
    out = Path(args.out or run / "misclassified")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    n_frames = apply_val_stride(cfg.val_dataset, cfg.val_stride)
    n_gt = apply_truncated_filter(cfg.val_dataset, cfg.exclude_truncated)
    print(f"val={cfg.val_dataset} stride={cfg.val_stride} -> {n_frames} frames, {n_gt} GT")

    det = FrozenSingleClassDetector(cfg, device=args.device)
    pipe = CropPipeline(cfg, det, cfg.val_dataset, is_train=False)
    model = build_classifier(cfg, len(pipe.class_names)).to(device)
    state = torch.load(cfg.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"config={cfg_path.name}  weights={cfg.weights} (iter {state.get('iteration')})  "
          f"mask_dilate={cfg.mask_dilate} mask_channel={cfg.mask_channel} blackout={cfg.blackout}")

    cam = None if args.no_gradcam else CropGradCAM(model, layer=args.gradcam_layer)
    if cam is not None:
        print(f"grad-cam layer: {cam.layer_name}")
    wrong, total = collect(model, pipe, device, max_frames=args.frames, cam=cam)
    acc = 1 - len(wrong) / max(total, 1)
    print(f"\n{len(wrong)} wrong / {total} matched crops  top1={acc:.4f}")

    pairs = Counter((w["gt_class"], w["pred_class"]) for w in wrong)
    print("\nconfusions (gt -> pred, count):")
    for (gt, pr), c in pairs.most_common():
        print(f"  {c:4d}  {gt:<16} -> {pr}")

    paths = sheets(wrong, out, args.per_sheet)
    with (out / "crops.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        wr.writerows([{k: w[k] for k in COLS} for w in wrong])
    (out / "summary.json").write_text(json.dumps({
        "weights": cfg.weights,
        "dataset": cfg.val_dataset,
        "val_stride": cfg.val_stride,
        "matched_crops": total,
        "n_wrong": len(wrong),
        "top1": round(acc, 4),
        "confusions": {f"{gt}->{pr}": c for (gt, pr), c in pairs.most_common()},
    }, indent=2))
    print(f"\nwrote {out / 'crops.csv'}, summary.json and {len(paths)} sheets to {out}")


if __name__ == "__main__":
    main()
