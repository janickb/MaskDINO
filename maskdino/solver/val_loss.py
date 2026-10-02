"""Periodic validation loss: mirrors the "total_loss" that training already
writes to EventStorage every step, but averaged over the validation set instead
of the current training batch, so it shows up alongside "total_loss" in
TensorBoard / metrics.json / the console as a single "validation_loss" scalar.

Deliberately *only* the weighted total, not a "val_" copy of each loss
component - see the note in after_step().

This is the exact same loss function training uses - not a proxy metric. The
model only takes the loss branch (vs. its inference branch) when self.training
is True (see MaskDINO.forward), and that branch is untouched here: same
prepare_targets(), same self.criterion() Hungarian-matching loss, same
weight_dict scaling. The only difference from a training step is where the batch
comes from (the validation loader) and that no backward/optimizer step happens.

Because self.training gates the loss branch, this can't reuse
Trainer.build_test_loader()'s loader as-is: that one's mapper is built with
is_train=False, which drops "annotations"/"instances" entirely (inference
doesn't need GT). ValidationLossHook is handed a loader built the same way as
the *train* loader (is_train=True mapper) but pointed at DATASETS.TEST - see
Trainer.build_val_loss_loader() in train_net.py.

Runs at the same cadence as the regular eval (TEST.EVAL_PERIOD) - see build_hooks()
in train_net.py - not every training step, which would mean a full validation-set
forward pass per iteration.

Two known deviations from a "clean" held-out loss, both inherited from the
long-standing community LossEvalHook pattern this follows:

- Reuses the train-time mapper, so validation batches go through the same random
  augmentation pipeline as training (not a deterministic eval-time resize) -
  expect some extra run-to-run noise in the curve.
- Getting the loss branch requires flipping the model to .train() for the
  duration of the check. Under torch.no_grad() no backward/gradient update
  happens, but any BatchNorm layer still in train mode updates its running stats
  from validation batches during forward, and Dropout still samples. Harmless
  for this repo's reclassify configs (backbone BN stays frozen, DROPOUT=0.0 -
  see those configs' own comments) - worth checking before reuse elsewhere.
"""
import logging

import torch
from detectron2.engine.train_loop import HookBase
from detectron2.utils import comm

logger = logging.getLogger(__name__)


class ValidationLossHook(HookBase):
    """Every `period` iterations (and on the final iteration), runs the model's
    loss branch over `loader` (sharded per-rank like any detectron2 test loader)
    and writes the cross-rank average to EventStorage as "validation_loss".
    """

    def __init__(self, period, loader):
        """
        period: run every `period` iterations (and on the last iteration). <= 0
            disables the hook entirely.
        loader: iterable data loader over the validation set, built with an
            is_train=True mapper so batches carry GT "instances". Re-iterated
            (from the start) on every check.
        """
        self._period = period
        self._loader = loader
        self._warned_empty = False

    def _steps_this_rank(self):
        """Number of val batches EVERY rank must run - the per-rank minimum.

        The training-mode loss runs a torch.distributed.all_reduce over num_masks
        inside SetCriterion.forward (maskdino/modeling/criterion.py). So one forward
        pass == one collective, and every rank has to perform the SAME number of them
        or the collectives mismatch and the job deadlocks until gloo's 30-minute
        timeout.

        build_detection_test_loader shards the validation set with InferenceSampler,
        which hands the remainder to the lower ranks - so the shards are uneven whenever
        len(dataset) is not divisible by the world size (e.g. 3 images over 2 ranks ->
        2 and 1). Truncating every rank to the global minimum costs at most
        world_size - 1 batches of the validation average and keeps the collectives
        matched.
        """
        try:
            local = len(self._loader)
        except TypeError:  # loader without __len__; fall back to no truncation
            return None
        return min(comm.all_gather(local))

    def _local_loss_sums(self):
        model = self.trainer.model
        was_training = model.training
        model.train()
        sums: dict[str, float] = {}
        n_batches = 0
        n_steps = self._steps_this_rank()
        try:
            with torch.no_grad():
                for i, batch in enumerate(self._loader):
                    if n_steps is not None and i >= n_steps:
                        break
                    losses = model(batch)
                    for k, v in losses.items():
                        sums[k] = sums.get(k, 0.0) + float(v)
                    n_batches += 1
        finally:
            model.train(was_training)
        return sums, n_batches

    def after_step(self):
        if self._period <= 0:
            return
        next_iter = self.trainer.iter + 1
        is_last = next_iter >= self.trainer.max_iter
        if next_iter % self._period != 0 and not is_last:
            return

        local_sums, local_n = self._local_loss_sums()
        # Every rank holds a different shard of the validation set (loader was
        # built with the usual per-rank InferenceSampler), so gather every rank's
        # partial sums before averaging - same reasoning as PlateauLRHook
        # broadcasting rank 0's metric, just weighted-sum instead of pick-one.
        gathered = comm.all_gather((local_sums, local_n))
        if not comm.is_main_process():
            return

        total_n = sum(n for _, n in gathered)
        if total_n == 0:
            if not self._warned_empty:
                logger.warning(
                    "[ValidationLossHook] validation loader produced no batches "
                    "at iter %d - skipping",
                    next_iter,
                )
                self._warned_empty = True
            return

        totals: dict[str, float] = {}
        for sums, _ in gathered:
            for k, v in sums.items():
                totals[k] = totals.get(k, 0.0) + v
        means = {k: v / total_n for k, v in totals.items()}

        # The weighted total only, deliberately - no "val_<component>" scalar per loss
        # component. The components are already written per-step on the training side,
        # so the copies multiplied the TensorBoard series without adding a signal
        # anyone reads. This is the note the module docstring points at.
        self.trainer.storage.put_scalar("validation_loss", sum(means.values()))
