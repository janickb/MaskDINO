# Stage-1 Optuna objective: choosing a metric

**Setting:** a two-stage pipeline.
- **Stage 1:** single-class MaskDINO instance segmentation, tuned with Optuna.
- **Stage 2:** a classifier that labels each instance from its box and mask crop.

**Question:** which metric Optuna should optimize for stage 1: segm AP, PQ or F1.

## Definitions
These use single-class Hungarian matching with mask IoU ≥ 0.5, as in `maskdino/evaluation/hungarian_instance_evaluation.py`.

- **F1_seg** = TP / (TP + ½FP + ½FN). This equals PQ's recognition quality (RQ).
- **SQ** = mean IoU over matched pairs (the evaluator logs it as `mean_iou_matched`).
- **PQ** = SQ × RQ.
- **Acc_cls|matched** = correct / matched. This is stage 2's job (equal to `1 − misclassified_per_match`).

**Key identity:** **F1_classaware = Acc_cls|matched × F1_seg**. It is exact when two conditions hold:
- a misclassification counts as both an FP and an FN (the evaluator's convention), and
- stage 2 has no reject option.

So the end-to-end score splits cleanly into the two stages.

## Recommendations
1. **Don't use segm AP as the primary objective.** AP scores operating points that are never deployed, and its IoU 0.5:0.95 average overweights boundary precision.
2. **Don't score at a fixed score threshold such as 0.5.** The score then depends on calibration, which hyperparameters shift without changing the ranking: focal α/γ, class loss weight, learning rate, number of queries.
   - Use the **maximum over a score-threshold grid** instead.
   - The pattern already exists in `tools/scaling/sweep_focal_thresholds.py`: compute the IoU matrix once per image, then re-run the matcher on the predictions above each threshold.
3. **Choose the objective based on stage 2's design:**
   - **Without a reject class:** use max-over-threshold F1_seg.
   - **With a reject class:** favor recall. Use recall at a fixed low threshold with a cap on FPs per image, or use AP50. Tune the final threshold on the full pipeline.
   - In both cases, stage 2 cannot remove **duplicate** masks on the same instance, so monitor them.
4. **Use SQ or PQ only if stage-2 accuracy depends on mask quality.** Measure this in two ways:
   - Compare the classifier on GT masks with the classifier on predicted masks.
   - Plot classification accuracy against matched IoU, in bins.

   If accuracy is flat above an IoU of about 0.6–0.7, use F1. If it rises with IoU, use max-over-threshold PQ. For thin structures such as forceps tips, consider boundary IoU.
5. **Optuna pruning:** fixed-threshold metrics are noisy early in training. Prune on the max-over-threshold metric or on AP50.
6. **Log the parts separately:** F1, SQ, and FPs/duplicates per image. A product such as PQ can hide trade-offs; for example, dropping hard instances raises SQ while lowering F1.
7. **Prefer the metric with the best signal-to-noise ratio.** Compare the seed-to-seed σ from the noise-floor replicates with the spread between configurations.

## Suggested sequence
1. While stage 2 doesn't exist yet, run Optuna on **max-over-threshold F1_seg**, or on AP50 if stage 2 will have a reject class. Log SQ and FPs per image.
2. Once a first stage 2 exists, run the GT-mask vs predicted-mask check. Switch to PQ only if mask quality matters.
3. Re-rank the top stage-1 trials by **end-to-end class-aware F1** to confirm that the stage-1 metric picked the right trials.

## Open questions
- Will stage 2 have a reject (background) class?
- Does stage-2 accuracy depend on mask IoU or on boundary quality?
- Will the score threshold be retuned per model before deployment?

## Repo context
- The current multi-class study scores runs with `instance_matching/classaware_f1` at a score threshold of 0.5 and an IoU threshold of 0.5. See `OBJECTIVE_KEY` in `tools/compare_runs.py`.
- The evaluator's metrics are documented in `maskdino/evaluation/hungarian_evaluator_metrics.md`.
