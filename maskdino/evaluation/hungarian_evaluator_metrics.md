# HungarianInstanceEvaluator — metric reference

`maskdino/evaluation/hungarian_instance_evaluation.py`. Enabled per config via
`MODEL.MaskDINO.TEST.HUNGARIAN_EVAL.ENABLED`. Runs alongside `COCOEvaluator`.

Per rendered frame: drop predictions with `score < SCORE_THRESH` (0.5), build the exact
pred×GT mask-IoU matrix, run Hungarian (`scipy.optimize.linear_sum_assignment`) with a
**lexicographic** objective — maximize the number of pairs with `IoU ≥ IOU_THRESH` (0.5)
first, then total IoU as a tie-break, **labels ignored** — then reject any matched pair
with `IoU < IOU_THRESH`: its prediction becomes a false positive, its GT a false negative.
Results are aggregated over the eval set and returned under the `instance_matching`
namespace (TensorBoard tags `instance_matching/<field>`).

## Config

`cfg.MODEL.MaskDINO.TEST.HUNGARIAN_EVAL`:

| key | default | meaning |
|---|---|---|
| `ENABLED` | `False` | attach the evaluator (surgical YAMLs set it `True`) |
| `SCORE_THRESH` | `0.5` | drop predictions below this confidence before matching |
| `IOU_THRESH` | `0.5` | min mask IoU for a match to be accepted |
| `BOX_PREFILTER` | `True` | bounding-box prune before the exact mask-IoU matmul (optimization only) |

GT visibility filtering is **not** a knob here — the evaluator reuses `cfg.INPUT.MIN_VISIBILITY`
(the training-time GT filter) so recall is measured against the same GT distribution the
model trained on. It only bites when frames were rendered with visibility computation on.

## Baseline quantities

A GT instance is **mask-matched** if Hungarian paired it to a prediction with
`IoU ≥ IOU_THRESH` (matching does not look at labels).

| symbol | meaning |
|---|---|
| `num_gt` | GT instances (after `iscrowd == 1` and `visibility_fraction < INPUT.MIN_VISIBILITY` skips) |
| `num_pred` | predictions kept (`score ≥ SCORE_THRESH`) |
| `num_mask_matched` | mask-matched, **any** predicted label |
| `num_correct_class` | mask-matched **and** predicted label == GT label |
| `num_false_neg` | GT with no accepted match (`num_mask_matched + num_false_neg = num_gt`) |
| `num_false_pos` | predictions with no accepted match (`num_mask_matched + num_false_pos = num_pred`) |
| `num_misclassified` | `num_mask_matched − num_correct_class` |
| per class `c` | `matched_c = gt_count[c] − false_neg[c]`, `correct_c = matched_c − misclassified[c]` |

Two axes for the derived metrics: **class-agnostic vs class-aware** (does a wrong label
disqualify the detection?) and **aggregate vs per-class**.

## Precision / recall / F1

Computed from raw counts, both conventions, aggregate:

| field prefix | TP | FP | FN |
|---|---|---|---|
| `classagnostic_` | `num_mask_matched` | `num_false_pos` | `num_false_neg` |
| `classaware_` | `num_correct_class` | `num_false_pos + num_misclassified` | `num_false_neg + num_misclassified` |

`precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `f1 = 2PR/(P+R)`. A misclassified match is
both a false positive (wrong label emitted) and a false negative (true class not recovered)
in the class-aware view. Fields: `classagnostic_precision/recall/f1`,
`classaware_precision/recall/f1`.

Identities (hold always):
- `classagnostic_recall == num_mask_matched_total / num_gt_total`
- `classaware_recall == num_correct_class_total / num_gt_total`
- `classagnostic_recall − classaware_recall == misclassified_total / num_gt_total`

## Per-class recall (flattened as `instance_matching/<group>/<classname>`)

| field | formula | breakdown of |
|---|---|---|
| `classagnostic_recall_per_class[c]` | `matched_c / gt_count[c]` | `classagnostic_recall` (GT-count-weighted mean) |
| `classaware_recall_per_class[c]` | `correct_c / gt_count[c]` | `classaware_recall` |
| `false_negatives_per_class[c]` | count | `false_negatives_total` |
| `false_negatives_rate_per_class[c]` | `false_neg[c] / gt_count[c]` | — |
| `misclassified_per_class[c]` | count (**multi-class only**) | `misclassified_total` |

Per class: `classagnostic_recall_per_class[c] − classaware_recall_per_class[c] = misclassified[c] / gt_count[c]`.
Class id 0 (the `"background"` slot) is skipped — it never has GT.

### Example

2 scalpel + 1 forceps GT; one scalpel matched + correct, the forceps matched but labeled
"clamp", the other scalpel a false negative:

```
num_gt=3  num_mask_matched=2  num_correct_class=1  num_false_neg=1  num_misclassified=1

classagnostic_recall              = 2/3 = 0.667
classagnostic_recall_per_class    = {scalpel: 0.5, forceps: 1.0}
classaware_recall                 = 1/3 = 0.333
classaware_recall_per_class       = {scalpel: 0.5, forceps: 0.0}
```
forceps: found it (`classagnostic_recall_per_class = 1.0`) but named it wrong
(`classaware_recall_per_class = 0.0`) — the gap is that one misclassification.

## Other emitted fields

| field | meaning |
|---|---|
| `false_negatives_total`, `false_negatives_per_image_{mean,median,max}` | false-negative counts; `*_per_image_mean` is the "per rendering" headline |
| `false_negatives_per_gt` | `false_negatives_total / num_gt_total` |
| `misclassified_total`, `misclassified_per_image_mean` | disabled (`0.0`) in single-class mode |
| `misclassified_per_match` | `misclassified_total / num_mask_matched_total` |
| `false_positives_total`, `false_positives_per_image_mean` | "hallucinated" predictions (matched no GT) |
| `false_positives_per_pred` | `false_positives_total / num_pred_total` |
| `mean_iou_matched` | mean IoU over accepted matches — segmentation quality of what was found |
| `num_images`, `num_gt_total`, `num_pred_total`, `num_mask_matched_total`, `num_correct_class_total` | raw totals |

**Single-class mode** (`NUM_CLASSES == 1`): `pred_classes` is always `0` while GT
`category_id ≥ 1`, so misclassification is disabled — `num_misclassified` stays `0`, the
`misclassified_per_class/*` group is dropped, and `classaware_*` equals `classagnostic_*`.

## Artifacts

Written to `<OUTPUT_DIR>/inference/` (only on the main process, when an `output_dir` is set):

| file | contents |
|---|---|
| `instance_matching_confusion.csv` | raw `(C+1)×(C+1)` count matrix — GT class rows, predicted class cols, plus a `(false negative)` column (GT with no accepted match) and a `(false positive)` row (predictions that matched nothing). Row sum = that class's GT count; column sum = predictions of that class. Diagonal of the `C×C` block = correct, off-diagonal = misclassified. |
| `instance_matching_confusion_rownorm.csv` | same, row-normalized so **every row sums to 1**: GT rows ÷ that class's GT total (diagonal = `classaware_recall_per_class`, `(false negative)` cell = false-negative rate); `(false positive)` row ÷ total false-positive count (cell = that label's share of all spurious predictions). |
| `instance_matching_confusion.png` | heatmap of the row-normalized matrix, each cell annotated `fraction% / count`. |
| `instance_matching_per_image.csv` | one row per frame: `image_id, file_name, num_gt, num_pred, num_mask_matched, num_correct_class, num_false_neg, num_false_pos, num_misclassified, sum_iou_matched`. |

Under `--eval-only` the scalar dict is only printed / JSON-logged (no TensorBoard); the
artifacts are still written.
