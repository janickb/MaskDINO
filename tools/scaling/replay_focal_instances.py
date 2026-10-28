#!/usr/bin/env python
"""Per-GT-instance replay of one focal-sweep run's full-val eval, for
tools/scaling/analyze_focal_sweep.py.

Reads <run>/eval_full/inference/ (written by run_focal_sweep.sh), no GPU. GT is
exactly the official eval's - stride 1, frame-edge-truncated instances as iscrowd
ignore regions, the visibility skip of HungarianInstanceEvaluator._load_gt_masks -
and so are the matcher and the crowd rule for false positives, so at the config's
SCORE_THRESH the counts reproduce that run's instance_matching_confusion.csv.
Caveat: eval_full/ directories written before the iscrowd change cached their
predictions AFTER the old box-touches-the-edge prediction drop, which cannot be
undone here - re-run their eval (EVAL_ONLY=1 run_focal_sweep.sh) for clean numbers.
On top of the official per-image tallies
it keeps, per kept GT instance:

  * the object's index in instance_segmaps - coco_annotations carry no instance
    id and drop objects that leave the view, so this is what pairs the same
    physical object across the 0.67x / 1.0x / 1.5x renders of one frame;
  * matched class, mask IoU and score, and the best IoU with ANY kept prediction
    (a miss can then be split into "nothing there" vs. "mask below the IoU gate").

Each SCORE_THRESH needs its own replay: predictions are gated before matching.
Also stores per-class COCO segm AP / AP50 / AP75, so the analysis needs nothing
but these files.

    ./.venv/bin/python tools/scaling/replay_focal_instances.py \
        runs/20261026164331_focal_1.0x_seed42 runs/_analysis/focal_sweep_newval/inst_1.0x_seed42.npz \
        [--score-thresh 0.5 0.3] [--eval-dir eval_full_v2]
"""
import argparse
import contextlib
import glob
import io
import os
import sys
from collections import Counter

import h5py
import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog, MetadataCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402

from maskdino import add_maskdino_config, set_num_classes_from_metadata  # noqa: E402
from maskdino.data.datasets.register_hdf5_instance import (  # noqa: E402
    apply_test_sample_stride,
    apply_truncated_instance_filter,
)
from maskdino.evaluation.hungarian_instance_evaluation import (  # noqa: E402
    crowd_overlap,
    hungarian_match,
    mask_iou_matrix,
)


def decode(seg):
    if isinstance(seg["counts"], str):
        seg = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
    return mask_util.decode(seg).astype(bool)


def coco_segm_ap(inference_dir):
    """(class names, [K, 3] AP/AP50/AP75 per class, [3] overall), in percent."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(glob.glob(os.path.join(inference_dir, "*_coco_format.json"))[0])
        dt = gt.loadRes(os.path.join(inference_dir, "coco_instances_results.json"))
        ev = COCOeval(gt, dt, "segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    p = ev.eval["precision"]  # [iou thr, recall, class, area range, max dets]

    def ap(k, t=None):
        x = p[:, :, k, 0, -1] if t is None else p[t, :, k, 0, -1]
        x = x[x > -1]
        return 100 * float(x.mean()) if x.size else float("nan")

    cats = [gt.loadCats(int(c))[0]["name"] for c in ev.params.catIds]
    per_class = np.array([[ap(k), ap(k, 0), ap(k, 5)] for k in range(len(cats))])
    return cats, per_class, 100 * np.asarray(ev.stats[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_npz")
    ap.add_argument("--score-thresh", type=float, nargs="+", default=[0.5, 0.3])
    ap.add_argument("--eval-dir", default="eval_full", help="eval output dir inside the run (run_focal_sweep.sh VAL_TAG)")
    args = ap.parse_args()
    torch.set_num_threads(8)  # the driver runs one replay per run in parallel

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(os.path.join(args.run_dir, args.eval_dir, "config.yaml"))
    cfg.DATASETS.TEST_SAMPLE_STRIDE = 1
    dataset = cfg.DATASETS.TEST[0]
    set_num_classes_from_metadata(cfg, dataset)
    apply_test_sample_stride(cfg)
    apply_truncated_instance_filter(cfg)
    he = cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL
    min_vis = float(cfg.INPUT.MIN_VISIBILITY)
    names = list(MetadataCatalog.get(dataset).thing_classes)

    inference_dir = os.path.join(args.run_dir, args.eval_dir, "inference")
    preds = torch.load(os.path.join(inference_dir, "instances_predictions.pth"), weights_only=False)
    preds_by_id = {p["image_id"]: p["instances"] for p in preds}

    rows = {t: [] for t in args.score_thresh}
    fps = {t: Counter() for t in args.score_thresh}
    for d in DatasetCatalog.get(dataset):
        h, w = d["height"], d["width"]
        anns = [
            a for a in d["annotations"]
            if a.get("iscrowd", 0) != 1 and a.get("visibility_fraction", 1.0) >= min_vis
        ]
        crowd = [decode(a["segmentation"]) for a in d["annotations"] if a.get("iscrowd", 0) == 1]
        crowd = torch.from_numpy(np.stack(crowd)) if crowd else torch.zeros((0, h, w), dtype=torch.bool)
        frame = os.path.basename(d["file_name"])
        with h5py.File(d["file_name"], "r") as f:
            segmap = f["instance_segmaps"][()]
        gt_masks, gt_cls, obj_idx, areas = [], [], [], []
        for a in anns:
            m = decode(a["segmentation"])
            vals, cnt = np.unique(segmap[m], return_counts=True)
            obj_idx.append(int(vals[np.argmax(cnt)]) if vals.size else -1)
            gt_masks.append(torch.from_numpy(m))
            gt_cls.append(int(a["category_id"]))
            areas.append(int(m.sum()))
        gt_masks = torch.stack(gt_masks) if gt_masks else torch.zeros((0, h, w), dtype=torch.bool)
        gt_cls = np.asarray(gt_cls, dtype=np.int64)

        dets_all = preds_by_id.get(d["image_id"], [])
        for t in args.score_thresh:
            dets = [x for x in dets_all if x["score"] >= t]
            if dets:
                pm = torch.from_numpy(np.stack([decode(x["segmentation"]) for x in dets]))
                pc = np.array([x["category_id"] for x in dets], dtype=np.int64)
                ps = np.array([x["score"] for x in dets], dtype=np.float32)
            else:
                pm = torch.zeros((0, h, w), dtype=torch.bool)
                pc = np.zeros(0, np.int64)
                ps = np.zeros(0, np.float32)
            iou = mask_iou_matrix(pm, gt_masks, box_prefilter=bool(he.BOX_PREFILTER))
            r, c, miou = hungarian_match(iou, float(he.IOU_THRESH))
            ok = miou >= float(he.IOU_THRESH)
            r, c, miou = r[ok], c[ok], miou[ok]
            pred_for_gt = {int(g): (int(p), float(v)) for p, g, v in zip(r, c, miou)}
            best_iou = iou.max(dim=0).values.numpy() if iou.numel() else np.zeros(len(anns))
            for j in range(len(anns)):
                p, v = pred_for_gt.get(j, (-1, 0.0))
                rows[t].append((
                    frame, obj_idx[j], gt_cls[j], areas[j],
                    pc[p] if p >= 0 else -1, v, ps[p] if p >= 0 else 0.0, float(best_iou[j]),
                ))
            unmatched = sorted(set(range(len(dets))) - set(r.tolist()))
            overlap = crowd_overlap(pm[unmatched], crowd).tolist() if unmatched else []
            for i, o in zip(unmatched, overlap):
                if o < float(he.IOU_THRESH):  # on a cut tool -> ignored, as in the evaluator
                    fps[t][int(pc[i])] += 1

    ap_classes, ap_per_class, ap_all = coco_segm_ap(inference_dir)
    out = {
        "names": np.array(names),
        "ap_classes": np.array(ap_classes),
        "ap_per_class": ap_per_class,
        "ap_all": ap_all,
    }
    for t in args.score_thresh:
        a = rows[t]
        k = f"s{t}"
        out[k + "_frame"] = np.array([x[0] for x in a])
        out[k + "_obj"] = np.array([x[1] for x in a], np.int64)
        out[k + "_gt"] = np.array([x[2] for x in a], np.int64)
        out[k + "_area"] = np.array([x[3] for x in a], np.int64)
        out[k + "_pred"] = np.array([x[4] for x in a], np.int64)  # -1 = missed
        out[k + "_iou"] = np.array([x[5] for x in a], np.float32)
        out[k + "_score"] = np.array([x[6] for x in a], np.float32)
        out[k + "_bestiou"] = np.array([x[7] for x in a], np.float32)
        out[k + "_fp"] = np.array([fps[t].get(i, 0) for i in range(len(names))], np.int64)
    np.savez_compressed(args.out_npz, **out)
    print(f"[replay] {args.run_dir}: {len(rows[args.score_thresh[0]])} GT -> {args.out_npz}")


if __name__ == "__main__":
    main()
