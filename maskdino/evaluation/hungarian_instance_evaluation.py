# ------------------------------------------------------------------------
# Hungarian (optimal bipartite) mask-IoU instance evaluator.
#
# A single-operating-point diagnostic that complements COCOEvaluator: at a fixed
# confidence threshold, per rendered frame, how many instruments were missed and how
# many were misclassified. Predicted and ground-truth instance masks are matched
# per image by exact mask IoU via scipy.optimize.linear_sum_assignment on cost = -IoU
# (class-agnostic); a forced match whose IoU is below IOU_THRESH is rejected, so its
# prediction becomes a false positive and its GT counts as "missed".
#
# Config: cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.{ENABLED,SCORE_THRESH,IOU_THRESH,
#         MIN_VISIBILITY,BOX_PREFILTER}  (see maskdino/config.py)
#
# NOTE: this assumes GT masks and predicted masks describe the SAME thing (both modal =
# visible pixels only, or both amodal). The validation GT is modal
# (coco_annotations.amodal: false) and MaskDINO predicts modal masks, so a partially
# occluded tool has a small GT mask and a small predicted mask over the same region and
# IoU is not gated by visibility. If GT is ever regenerated with amodal: true WITHOUT
# retraining the model on amodal masks, IoU collapses to ~visibility fraction and
# IOU_THRESH silently becomes an occlusion filter.
# ------------------------------------------------------------------------
import csv
import itertools
import logging
import os
from collections import Counter, OrderedDict

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import BitMasks, pairwise_iou
from detectron2.utils import comm
from scipy.optimize import linear_sum_assignment

__all__ = ["HungarianInstanceEvaluator", "hungarian_match", "mask_iou_matrix"]


def mask_iou_matrix(pred_masks, gt_masks, box_prefilter=True, eps=1e-6):
    """Exact pairwise mask IoU -> (N, M) float32 tensor in [0, 1] on CPU.

    Args:
        pred_masks: (N, H, W) bool/uint8/float tensor, binary.
        gt_masks:   (M, H, W) bool/uint8/float tensor, binary.
        box_prefilter: use tight bounding boxes to prune. Disjoint boxes imply zero
            mask overlap, so predictions / GTs that overlap nothing are dropped and the
            exact (matmul) mask-IoU computation is restricted to the remaining
            candidate rows x columns. Non-candidate entries stay 0. Degrades gracefully
            to the full computation when every mask overlaps something.
    """
    n, m = pred_masks.shape[0], gt_masks.shape[0]
    iou = torch.zeros((n, m), dtype=torch.float32)
    if n == 0 or m == 0:
        return iou

    pred_b = pred_masks.bool().cpu()
    gt_b = gt_masks.bool().cpu()

    cand = None
    if box_prefilter:
        pb = BitMasks(pred_b).get_bounding_boxes()
        gb = BitMasks(gt_b).get_bounding_boxes()
        cand = pairwise_iou(pb, gb) > 0  # (N, M) bool: boxes overlap
        if not bool(cand.any()):
            return iou
        rows = torch.nonzero(cand.any(dim=1), as_tuple=False).squeeze(1)
        cols = torch.nonzero(cand.any(dim=0), as_tuple=False).squeeze(1)
    else:
        rows = torch.arange(n)
        cols = torch.arange(m)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # (K, H*W) float on the compute device, restricted to candidate rows/cols. If this
    # ever OOMs on a huge candidate block, chunk the matmul: a[i:i+chunk] @ b.t().
    a = pred_b[rows].reshape(rows.numel(), -1).to(device=device, dtype=torch.float32)
    b = gt_b[cols].reshape(cols.numel(), -1).to(device=device, dtype=torch.float32)
    inter = a @ b.t()  # (R, C)
    union = a.sum(dim=1)[:, None] + b.sum(dim=1)[None, :] - inter
    sub = (inter / union.clamp_min(eps)).cpu().to(torch.float32)

    if cand is not None:
        sub = torch.where(cand[rows][:, cols], sub, torch.zeros_like(sub))

    iou[rows.unsqueeze(1), cols.unsqueeze(0)] = sub
    return iou


def hungarian_match(iou, iou_thresh):
    """Optimal bipartite matching with a lexicographic objective:
    maximize the NUMBER of pairs with IoU >= iou_thresh first, then total IoU as a
    tie-break. This is what the metric reports (accepted-match count); it avoids the
    failure where a max-sum assignment keeps one near-perfect pair while dropping two
    just-above-threshold pairs.

    Single combined weight
        w[i, j] = (iou[i, j] >= t) * (K + iou[i, j])
    with K = min(N, M) + 1 > any achievable sum of the IoU tie-break terms, so one extra
    above-threshold pair always outweighs the tie-break (delta >= K - min(N, M) = 1).

    Returns (row_ind, col_ind, matched_iou), each of length min(N, M). The caller still
    rejects pairs whose matched_iou is below iou_thresh.
    """
    if iou.numel() == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))
    iou_np = iou.cpu().numpy()

    # One weight per pair, so the best assignment maximizes the count of pairs with
    # IoU >= thresh first, then their total IoU as a tie-break:
    #   below thresh -> 0            (IoU ignored)
    #   at/above     -> k + IoU
    # k must exceed the largest possible sum of the tie-break IoUs. A matching has at
    # most min(N, M) pairs, each IoU < 1, so that sum < min(N, M); k = min(N, M) + 1.
    # Gaining one above-thresh pair adds >= k, which always beats any tie-break change.
    k = float(min(iou_np.shape) + 1)
    weight = (iou_np >= iou_thresh) * (k + iou_np)

    # linear_sum_assignment minimizes cost -> negate to maximize weight.
    row_ind, col_ind = linear_sum_assignment(-weight)
    matched_iou = iou_np[row_ind, col_ind].astype(np.float32)
    return row_ind.astype(np.int64), col_ind.astype(np.int64), matched_iou


class HungarianInstanceEvaluator(DatasetEvaluator):
    """See module docstring. Returns metrics under the ``"instance_matching"`` namespace,
    which EvalHook flattens to TensorBoard tags ``instance_matching/<metric>`` during
    training-time evaluation. Under ``--eval-only`` the dict is only printed / JSON-logged.
    """

    def __init__(self, dataset_name, cfg, distributed=True, output_dir=None):
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")

        he = cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL
        self._score_thresh = float(he.SCORE_THRESH)
        self._iou_thresh = float(he.IOU_THRESH)
        self._min_visibility = float(he.MIN_VISIBILITY)
        self._box_prefilter = bool(he.BOX_PREFILTER)

        self._num_classes = int(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
        # Singleclass (NUM_CLASSES == 1): pred_classes is always 0 while GT category_id
        # is >= 1, so a class comparison would flag every match as misclassified.
        self._check_classes = self._num_classes > 1

        meta = MetadataCatalog.get(dataset_name)
        self._thing_classes = list(getattr(meta, "thing_classes", []) or [])

        # Eval-time `inputs` carry no annotations (the HDF5 test mapper pops them), so pull
        # GT straight from the registered dataset dicts, indexed by image_id.
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self._gt_by_image_id = {
            d["image_id"]: d.get("annotations", []) for d in dataset_dicts
        }
        self._hw_by_image_id = {
            d["image_id"]: (d["height"], d["width"]) for d in dataset_dicts
        }

        if not self._check_classes:
            self._logger.info(
                "[HungarianInstanceEvaluator] NUM_CLASSES == 1: misclassification metric "
                "disabled; misclassified_* reported as 0.0."
            )

    def reset(self):
        self._records = []

    # ------------------------------------------------------------------ helpers

    def _load_gt_masks(self, anns, gt_h, gt_w, image_id):
        """Decode GT annotations -> (BoolTensor[M, gt_h, gt_w], np.int64[M] category ids)."""
        masks, classes = [], []
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            if ann.get("visibility_fraction", 1.0) < self._min_visibility:
                continue
            seg = ann["segmentation"]
            counts = seg["counts"]
            if isinstance(counts, str):
                seg = {"size": seg["size"], "counts": counts.encode("utf-8")}
            m = mask_util.decode(seg)  # (h, w) uint8
            if m.shape[:2] != (gt_h, gt_w):
                raise RuntimeError(
                    f"GT mask for image_id={image_id} is {tuple(m.shape[:2])}, "
                    f"expected {(gt_h, gt_w)}"
                )
            masks.append(torch.from_numpy(np.ascontiguousarray(m)).bool())
            classes.append(int(ann["category_id"]))

        if masks:
            gt_masks = torch.stack(masks, dim=0)
        else:
            gt_masks = torch.zeros((0, gt_h, gt_w), dtype=torch.bool)
        return gt_masks, np.asarray(classes, dtype=np.int64)

    # ------------------------------------------------------------------ process

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            image_id = inp["image_id"]
            gt_h, gt_w = self._hw_by_image_id[image_id]
            anns = self._gt_by_image_id.get(image_id, [])
            if image_id not in self._gt_by_image_id:
                self._logger.warning(
                    "[HungarianInstanceEvaluator] image_id=%s missing from GT index; "
                    "treating as no ground truth.",
                    image_id,
                )

            gt_masks, gt_classes = self._load_gt_masks(anns, gt_h, gt_w, image_id)

            instances = out["instances"].to(self._cpu_device)
            keep = instances.scores >= self._score_thresh
            instances = instances[keep]
            pred_masks = instances.pred_masks.bool()
            pred_classes = instances.pred_classes.numpy().astype(np.int64)

            if tuple(pred_masks.shape[-2:]) != (gt_h, gt_w):
                raise RuntimeError(
                    f"Prediction masks for image_id={image_id} are "
                    f"{tuple(pred_masks.shape[-2:])}, GT is {(gt_h, gt_w)} — resolution "
                    f"mismatch, refusing to evaluate"
                )

            n, m = pred_masks.shape[0], gt_masks.shape[0]
            iou = mask_iou_matrix(
                pred_masks, gt_masks, box_prefilter=self._box_prefilter
            )
            row_ind, col_ind, matched_iou = hungarian_match(iou, self._iou_thresh)

            accepted = matched_iou >= self._iou_thresh
            acc_pred = row_ind[accepted]
            acc_gt = col_ind[accepted]
            acc_iou = matched_iou[accepted]

            num_matched = int(accepted.sum())
            num_missed = m - num_matched
            num_false_pos = n - num_matched

            if self._check_classes and num_matched:
                mis = pred_classes[acc_pred] != gt_classes[acc_gt]
            else:
                mis = np.zeros((num_matched,), dtype=bool)
            num_misclassified = int(mis.sum())
            num_correct = num_matched - num_misclassified

            matched_gt_set = set(acc_gt.tolist())
            missed_gt_classes = [
                int(gt_classes[j]) for j in range(m) if j not in matched_gt_set
            ]
            matched_pred_set = set(acc_pred.tolist())
            fp_pred_classes = [
                int(pred_classes[i]) for i in range(n) if i not in matched_pred_set
            ]

            self._records.append(
                {
                    "image_id": image_id,
                    "file_name": inp.get("file_name", ""),
                    "num_gt": m,
                    "num_pred": n,
                    "num_matched": num_matched,
                    "num_correct": num_correct,
                    "num_missed": num_missed,
                    "num_false_pos": num_false_pos,
                    "num_misclassified": num_misclassified,
                    "sum_iou_matched": float(acc_iou.sum()),
                    "matched_iou": [float(x) for x in acc_iou.tolist()],
                    "gt_count_per_class": dict(
                        Counter(int(c) for c in gt_classes.tolist())
                    ),
                    "missed_per_class": dict(Counter(missed_gt_classes)),
                    "false_pos_per_class": dict(Counter(fp_pred_classes)),
                    "misclassified_per_class": dict(
                        Counter(int(gt_classes[j]) for j in acc_gt[mis].tolist())
                    ),
                    "confusion_pairs": [
                        [int(gt_classes[g]), int(pred_classes[p])]
                        for g, p in zip(acc_gt.tolist(), acc_pred.tolist())
                    ],
                }
            )

    # ------------------------------------------------------------------ evaluate

    def evaluate(self):

        def _pr(tp, fp, fn):
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            return precision, recall, f1

        if self._distributed:
            comm.synchronize()
            records = comm.gather(self._records, dst=0)
            records = list(itertools.chain(*records))
            if not comm.is_main_process():
                return {}
        else:
            records = self._records

        if not records:
            self._logger.warning("[HungarianInstanceEvaluator] no records to evaluate.")
            return {"instance_matching": {}}

        n_images = len(records)
        num_gt = sum(r["num_gt"] for r in records)
        num_pred = sum(r["num_pred"] for r in records)
        num_matched = sum(r["num_matched"] for r in records)
        num_correct = sum(r["num_correct"] for r in records)
        missed = sum(r["num_missed"] for r in records)
        false_pos = sum(r["num_false_pos"] for r in records)
        misclassified = sum(r["num_misclassified"] for r in records)
        sum_iou = sum(r["sum_iou_matched"] for r in records)

        precision, recall, f1 = _pr(num_matched, false_pos, missed)
        precision_ca, recall_ca, f1_ca = _pr(
            num_correct, false_pos + misclassified, missed + misclassified
        )

        missed_arr = np.asarray([r["num_missed"] for r in records], dtype=np.float64)
        mis_arr = np.asarray(
            [r["num_misclassified"] for r in records], dtype=np.float64
        )
        fp_arr = np.asarray([r["num_false_pos"] for r in records], dtype=np.float64)

        # per-class aggregation, keyed by integer class id
        gt_count = Counter()
        missed_pc = Counter()
        fp_pc = Counter()
        mis_pc = Counter()
        confusion = Counter()  # (gt_id, pred_id) -> count
        for r in records:
            for k, v in r["gt_count_per_class"].items():
                gt_count[int(k)] += v
            for k, v in r["missed_per_class"].items():
                missed_pc[int(k)] += v
            for k, v in r["false_pos_per_class"].items():
                fp_pc[int(k)] += v
            for k, v in r["misclassified_per_class"].items():
                mis_pc[int(k)] += v
            for g, p in r["confusion_pairs"]:
                confusion[(int(g), int(p))] += 1

        def _cname(cid):
            name = (
                self._thing_classes[cid]
                if 0 <= cid < len(self._thing_classes)
                else str(cid)
            )
            return str(name).replace("/", "_")

        missed_per_class = {}
        missed_rate_per_class = {}
        recall_per_class = {}  # class-agnostic: breakdown of headline `recall`
        recall_classaware_per_class = {}  # class-aware: breakdown of `recall_classaware`
        misclassified_per_class = {}
        for cid in sorted(gt_count):
            if cid == 0:  # background slot, never a real instance
                continue
            name = _cname(cid)
            gtc = max(gt_count[cid], 1)
            matched_c = gt_count[cid] - missed_pc.get(
                cid, 0
            )  # accepted mask matches, any label
            correct_c = matched_c - mis_pc.get(cid, 0)  # matched AND correctly labeled
            missed_per_class[name] = float(missed_pc.get(cid, 0))
            missed_rate_per_class[name] = float(missed_pc.get(cid, 0)) / gtc
            recall_per_class[name] = float(matched_c) / gtc
            recall_classaware_per_class[name] = float(correct_c) / gtc
            if self._check_classes:
                misclassified_per_class[name] = float(mis_pc.get(cid, 0))

        if self._output_dir and comm.is_main_process():
            self._write_artifacts(records, confusion, missed_pc, fp_pc)

        res = {
            # precision / recall / f1 -- class-agnostic (localization)
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            # precision / recall / f1 -- class-aware (detection + classification)
            "precision_classaware": float(precision_ca),
            "recall_classaware": float(recall_ca),
            "f1_classaware": float(f1_ca),
            # missed
            "missed_total": float(missed),
            "missed_per_image_mean": float(missed_arr.mean()),
            "missed_per_image_median": float(np.median(missed_arr)),
            "missed_per_image_max": float(missed_arr.max()),
            "missed_rate": float(missed) / max(num_gt, 1),
            # misclassified
            "misclassified_total": float(misclassified),
            "misclassified_per_image_mean": float(mis_arr.mean()),
            "misclassified_rate": float(misclassified) / max(num_matched, 1),
            # false positives
            "false_positives_total": float(false_pos),
            "false_positives_per_image_mean": float(fp_arr.mean()),
            "false_positive_rate": float(false_pos) / max(num_pred, 1),
            # quality / counts
            "mean_iou_matched": float(sum_iou) / max(num_matched, 1),
            "num_images": float(n_images),
            "num_gt_total": float(num_gt),
            "num_pred_total": float(num_pred),
            "num_matched_total": float(num_matched),
            "num_correct_total": float(num_correct),
        }
        # Per-class breakdowns are flattened into `res` with "<group>/<class>" keys rather
        # than kept as nested dicts: detectron2's print_csv_format walks res one level
        # deep and formats every value as a float, so a nested dict would crash it (it
        # still flattens correctly to TensorBoard tags via flatten_results_dict).
        per_class_groups = {
            "missed_per_class": missed_per_class,
            "missed_rate_per_class": missed_rate_per_class,
            "recall_per_class": recall_per_class,
            "recall_classaware_per_class": recall_classaware_per_class,
        }
        if self._check_classes:
            per_class_groups["misclassified_per_class"] = misclassified_per_class
        for group, values in per_class_groups.items():
            for name, val in values.items():
                res[f"{group}/{name}"] = float(val)

        self._logger.info(
            "[HungarianInstanceEvaluator] images=%d gt=%d pred=%d matched=%d correct=%d "
            "missed=%d misclassified=%d fp=%d | precision=%.4f recall=%.4f f1=%.4f | "
            "mean_iou_matched=%.4f",
            n_images,
            num_gt,
            num_pred,
            num_matched,
            num_correct,
            missed,
            misclassified,
            false_pos,
            precision,
            recall,
            f1,
            res["mean_iou_matched"],
        )
        return OrderedDict({"instance_matching": res})

    # ------------------------------------------------------------------ artifacts

    def _write_artifacts(self, records, confusion, missed_pc, fp_pc):
        os.makedirs(self._output_dir, exist_ok=True)
        max_id = 0
        for g, p in confusion:
            max_id = max(max_id, g, p)
        for cid in list(missed_pc) + list(fp_pc):
            max_id = max(max_id, cid)
        max_id = max(max_id, len(self._thing_classes) - 1, self._num_classes - 1)
        c = max_id + 1

        def _name(i):
            n = self._thing_classes[i] if i < len(self._thing_classes) else str(i)
            return str(n).replace("/", "_")

        cls_names = [_name(i) for i in range(c)]
        # Extended confusion matrix: an extra "(missed)" column (GT with no accepted
        # match) and an extra "(false positive)" row (predictions that matched nothing).
        # Then row sum = total GT of that class, column sum = total predictions of that
        # class (score >= SCORE_THRESH). The C x C block is accepted matches
        # (diagonal = correct, off-diagonal = misclassified).
        col_names = cls_names + ["(missed)"]
        row_names = cls_names + ["(false positive)"]
        mat = np.zeros((c + 1, c + 1), dtype=np.int64)  # rows = GT, cols = predicted
        for (g, p), cnt in confusion.items():
            mat[g, p] += cnt
        for cid, cnt in missed_pc.items():
            mat[cid, c] += cnt
        for cid, cnt in fp_pc.items():
            mat[c, cid] += cnt

        # Row-normalized view: every row is a distribution over the columns, summing to 1.
        #   real-class row  -> divided by that class's GT total; cell = "fraction of
        #     forcep03 GT that ended up here". Diagonal = per-class class-aware recall,
        #     "(missed)" column = per-class miss rate.
        #   "(false positive)" row -> divided by the total false-positive count; cell =
        #     "fraction of all spurious predictions that carry this label".
        row_denom = np.maximum(mat.sum(axis=1), 1).astype(np.float64)
        norm = mat / row_denom[:, None]

        conf_csv = os.path.join(self._output_dir, "instance_matching_confusion.csv")
        with open(conf_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gt\\pred"] + col_names)
            for i in range(c + 1):
                w.writerow([row_names[i]] + mat[i].tolist())
        self._logger.info("[HungarianInstanceEvaluator] wrote %s", conf_csv)

        rownorm_csv = os.path.join(
            self._output_dir, "instance_matching_confusion_rownorm.csv"
        )
        with open(rownorm_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["gt\\pred"] + col_names)
            for i in range(c + 1):
                w.writerow([row_names[i]] + [f"{v:.4f}" for v in norm[i].tolist()])
        self._logger.info("[HungarianInstanceEvaluator] wrote %s", rownorm_csv)

        per_img_csv = os.path.join(self._output_dir, "instance_matching_per_image.csv")
        fields = [
            "image_id",
            "file_name",
            "num_gt",
            "num_pred",
            "num_matched",
            "num_correct",
            "num_missed",
            "num_false_pos",
            "num_misclassified",
            "sum_iou_matched",
        ]
        with open(per_img_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in records:
                w.writerow(r)
        self._logger.info("[HungarianInstanceEvaluator] wrote %s", per_img_csv)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            d = c + 1
            fig, ax = plt.subplots(figsize=(max(6, d * 0.7), max(5, d * 0.7)))
            im = ax.imshow(norm, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(d))
            ax.set_yticks(range(d))
            ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(row_names, fontsize=8)
            ax.set_xlabel("predicted class")
            ax.set_ylabel("ground-truth class")
            ax.set_title(
                "Hungarian matched-instance confusion (row-normalized, each row sums to 1)\n"
                "GT rows / GT total  ;  FP row / total FP  ;  cell = fraction / count"
            )
            for i in range(d):
                for j in range(d):
                    if mat[i, j]:
                        ax.text(
                            j,
                            i,
                            f"{norm[i, j] * 100:.0f}%\n{int(mat[i, j])}",
                            ha="center",
                            va="center",
                            color="w" if norm[i, j] < 0.6 else "k",
                            fontsize=7,
                        )
            fig.colorbar(im, ax=ax, label="row-normalized fraction")
            fig.tight_layout()
            png = os.path.join(self._output_dir, "instance_matching_confusion.png")
            fig.savefig(png, dpi=150)
            plt.close(fig)
            self._logger.info("[HungarianInstanceEvaluator] wrote %s", png)
        except Exception as e:  # noqa: BLE001 - plotting must never break evaluation
            self._logger.warning(
                "[HungarianInstanceEvaluator] confusion PNG skipped: %s", e
            )
