#!/usr/bin/env python
"""Score-threshold sweep of one focal-sweep run, deployment-style, for
tools/scaling/analyze_focal_thresholds.py.

Reads <run>/eval_full/inference/instances_predictions.pth, no GPU. GT and the
ignore rule are the official eval's (cut tools are iscrowd regions; a prediction
left unmatched that lies on one is ignored - HungarianInstanceEvaluator), with one
deliberate difference:

  * ONE LABEL PER MASK. MaskDINO's instance_inference takes the top-k over
    (query x class), so one query's mask is emitted once per class that scores.
    Below ~0.5 a forcep03/04 object then carries both labels (7-9 % of them at 0.3),
    the score-blind matcher picks one arbitrarily and the other becomes a false
    positive. Predictions with an identical mask are collapsed to the
    highest-scored one, i.e. what a deployed system would report.

The IoU matrix is computed once per image; every threshold (0.10 .. 0.90) re-runs
the matcher on its row subset. Also stores, per kept forcep03/04 object, the best
03-labelled and best 04-labelled score on its mask (IoU >= 0.5, RAW predictions
incl. both labels of a query) for the threshold-free separability analysis.

    ./.venv/bin/python tools/scaling/sweep_focal_thresholds.py \
        runs/20261026164331_focal_1.0x_seed42 runs/_analysis/focal_sweep_newval/sw_1.0x_seed42.npz \
        [--eval-dir eval_full_v2]
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pycocotools.mask as mask_util
import torch

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

THRESH = np.round(np.arange(0.10, 0.901, 0.05), 2)
FLOOR = 0.05  # lowest score kept at all (also the floor for the separability margins)


def decode(seg):
    if isinstance(seg["counts"], str):
        seg = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
    return mask_util.decode(seg).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out_npz")
    ap.add_argument("--eval-dir", default="eval_full", help="eval output dir inside the run (run_focal_sweep.sh VAL_TAG)")
    args = ap.parse_args()
    run, out_npz = args.run, args.out_npz
    torch.set_num_threads(8)  # the driver runs one sweep per run in parallel
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(os.path.join(run, args.eval_dir, "config.yaml"))
    cfg.DATASETS.TEST_SAMPLE_STRIDE = 1
    ds = cfg.DATASETS.TEST[0]
    set_num_classes_from_metadata(cfg, ds)
    apply_test_sample_stride(cfg)
    apply_truncated_instance_filter(cfg)
    iou_thresh = float(cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.IOU_THRESH)
    min_vis = float(cfg.INPUT.MIN_VISIBILITY)
    names = list(MetadataCatalog.get(ds).thing_classes)
    c3, c4 = names.index("forcep03_mm"), names.index("forcep04_mm")
    preds = {
        p["image_id"]: p["instances"]
        for p in torch.load(os.path.join(run, args.eval_dir, "inference", "instances_predictions.pth"), weights_only=False)
    }

    frames, objs, gts, areas, predcls, s3s, s4s = [], [], [], [], [], [], []
    fp_img, img_names = [], []
    n_dedup = n_kept_preds = 0
    for d in DatasetCatalog.get(ds):
        h, w = d["height"], d["width"]
        with h5py.File(d["file_name"], "r") as f:
            segmap = f["instance_segmaps"][()]
        gm, gcls, oidx, area, crowd = [], [], [], [], []
        for a in d["annotations"]:
            if a.get("iscrowd", 0) == 1:
                crowd.append(torch.from_numpy(decode(a["segmentation"])))
                continue
            if a.get("visibility_fraction", 1.0) < min_vis:
                continue
            m = decode(a["segmentation"])
            gm.append(torch.from_numpy(m))
            gcls.append(int(a["category_id"]))
            vals, cnt = np.unique(segmap[m], return_counts=True)
            oidx.append(int(vals[np.argmax(cnt)]) if vals.size else -1)
            area.append(int(m.sum()))
        empty = torch.zeros((0, h, w), dtype=torch.bool)
        gm = torch.stack(gm) if gm else empty
        crowd = torch.stack(crowd) if crowd else empty
        gcls = np.array(gcls, np.int64)

        # one label per mask: identical RLE -> keep the highest score
        best = {}
        for x in preds.get(d["image_id"], []):
            if x["score"] < FLOOR:
                continue
            key = x["segmentation"]["counts"]
            if key not in best or x["score"] > best[key]["score"]:
                best[key] = x
        n_dedup += sum(1 for x in preds.get(d["image_id"], []) if x["score"] >= FLOOR) - len(best)
        dets = sorted(best.values(), key=lambda x: -x["score"])
        n_kept_preds += len(dets)
        if dets:
            pm = torch.from_numpy(np.stack([decode(x["segmentation"]) for x in dets]))
            pc = np.array([x["category_id"] for x in dets], np.int64)
            ps = np.array([x["score"] for x in dets], np.float32)
        else:
            pm, pc, ps = empty, np.zeros(0, np.int64), np.zeros(0, np.float32)
        iou = mask_iou_matrix(pm, gm).numpy() if len(dets) and len(gcls) else np.zeros((len(dets), len(gcls)))
        # crowd overlap does not depend on the score threshold: once per prediction
        on_crowd = (crowd_overlap(pm, crowd) >= iou_thresh).numpy() if len(dets) else np.zeros(0, bool)

        pcl = np.full((len(gcls), len(THRESH)), -1, np.int8)
        fp = np.zeros((len(THRESH), len(names)), np.int16)
        for ti, t in enumerate(THRESH):
            rows = np.nonzero(ps >= t)[0]
            matched_rows = set()
            if rows.size and len(gcls):
                r, c, miou = hungarian_match(torch.from_numpy(iou[rows]), iou_thresh)
                ok = miou >= iou_thresh
                for rr, cc in zip(r[ok], c[ok]):
                    matched_rows.add(int(rows[rr]))
                    pcl[cc, ti] = pc[rows[rr]]
            for i in rows:
                if int(i) not in matched_rows and not on_crowd[i]:
                    fp[ti, pc[i]] += 1
        fp_img.append(fp)
        img_names.append(os.path.basename(d["file_name"]))

        raw = [x for x in preds.get(d["image_id"], []) if x["score"] >= FLOOR and x["category_id"] in (c3, c4)]
        if raw and len(gcls):
            rm = torch.from_numpy(np.stack([decode(x["segmentation"]) for x in raw]))
            riou = mask_iou_matrix(rm, gm).numpy()
            rc = np.array([x["category_id"] for x in raw])
            rs = np.array([x["score"] for x in raw])
        for j in range(len(gcls)):
            frames.append(os.path.basename(d["file_name"]))
            objs.append(oidx[j])
            gts.append(gcls[j])
            areas.append(area[j])
            predcls.append(pcl[j])
            s3 = s4 = 0.0
            if raw:
                hit = riou[:, j] >= iou_thresh
                s3 = float(rs[hit & (rc == c3)].max()) if (hit & (rc == c3)).any() else 0.0
                s4 = float(rs[hit & (rc == c4)].max()) if (hit & (rc == c4)).any() else 0.0
            s3s.append(s3)
            s4s.append(s4)

    np.savez_compressed(
        out_npz, names=np.array(names), thresh=THRESH, frame=np.array(frames), obj=np.array(objs),
        gt=np.array(gts), area=np.array(areas), pred=np.stack(predcls),
        fp=np.stack(fp_img), img=np.array(img_names),
        s3=np.array(s3s, np.float32), s4=np.array(s4s, np.float32),
    )
    print(f"[sweep] {run}: {len(gts)} kept GT, {n_kept_preds} preds >= {FLOOR} after collapsing "
          f"{n_dedup} same-mask duplicates -> {out_npz}")


if __name__ == "__main__":
    main()
