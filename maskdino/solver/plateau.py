"""ReduceLROnPlateau support: an adaptive alternative to the fixed
warmup/cosine/SGDR schedules in lr_scheduler.py. Unlike those - which are pure
functions of iter/MAX_ITER - this cuts the LR only when a monitored metric (by
default "total_loss") stops improving for SOLVER.PLATEAU.PATIENCE checks in a row.

Two things this needs that the fixed schedules don't:

- torch.optim.lr_scheduler.ReduceLROnPlateau isn't an LRScheduler: it has no
  get_lr(), and its .step(metric) takes a required argument, meant to be driven at
  whatever cadence you choose (here, every SOLVER.PLATEAU.CHECK_PERIOD iterations) -
  not detectron2's built-in hooks.LRScheduler, which calls `scheduler.step()` with
  no arguments on every training iteration. PlateauLRScheduler below satisfies that
  per-iteration no-arg call as a no-op, and exposes step_on_metric() for
  PlateauLRHook to drive instead.
- the monitored metric is written to EventStorage only on the main process (see
  detectron2's SimpleTrainer.write_metrics, which gathers+averages across ranks
  before writing "total_loss", only on rank 0). PlateauLRHook computes the check on
  rank 0 and broadcasts the result to every rank via comm.all_gather, so every
  rank's optimizer - and every DDP model replica - stays on the identical LR;
  otherwise ranks would silently diverge.
- ReduceLROnPlateau has no notion of a start-of-training warmup - the fixed
  schedules give every config a one-time linear ramp (SOLVER.WARMUP_ITERS /
  WARMUP_FACTOR) via WarmupParamScheduler for exactly this reason (freshly
  initialized optimizer state / just-loaded pretrained weights). PlateauLRScheduler
  reimplements that same ramp manually (it owns the per-iteration step() call
  regardless), and only hands control to the plateau mechanism once it ends.

Known limitation: unlike the fixed ParamScheduler-based schedules,
PlateauLRScheduler's state (the warmup counter and ReduceLROnPlateau's own
best-value/patience/cooldown counters) is NOT included in checkpoints - detectron2's
hooks.LRScheduler.state_dict() only serializes torch.optim.lr_scheduler.LRScheduler
instances, which this isn't. Resuming a run with --resume against a config using
this scheduler restarts warmup and plateau tracking from scratch rather than
picking up where it left off. Fine for this repo's actual pattern (each config
launch is typically a fresh run against a previous phase's model_final.pth, not a
literal --resume mid-run), but worth knowing if that ever changes.
"""
import logging

import torch
from detectron2.engine.train_loop import HookBase
from detectron2.utils import comm

logger = logging.getLogger(__name__)


class PlateauLRScheduler:
    """Wraps torch.optim.lr_scheduler.ReduceLROnPlateau so it can sit as
    Trainer.scheduler: satisfies detectron2's hooks.LRScheduler per-iteration,
    no-argument `step()` contract - which it uses to drive a one-time linear
    warmup ramp identical in spirit to the fixed schedules' WARMUP_ITERS /
    WARMUP_FACTOR - and exposes `step_on_metric`, called by PlateauLRHook at its
    own cadence, for everything after warmup ends.

    `min_lr_fraction` is a *fraction of each param group's own starting LR*, not
    an absolute value - ReduceLROnPlateau's stock `min_lr` is a flat number (or a
    list, one per group) applied identically to every group regardless of its own
    base LR. A flat floor would clamp a low-LR group (e.g. the backbone, at
    BACKBONE_MULTIPLIER x BASE_LR) many reductions before a high-LR group hits the
    same number, silently drifting their ratio apart - the equivalent bug the
    cosine scheduler avoids by expressing BASE_LR_END as a ratio of BASE_LR,
    applied identically (via LRMultiplier) to every group's own base. Passing a
    fraction here reproduces that same ratio-preserving floor.
    """

    def __init__(
        self,
        optimizer,
        *,
        mode,
        factor,
        patience,
        threshold,
        cooldown,
        min_lr_fraction,
        warmup_iters=0,
        warmup_factor=1.0,
    ):
        self._optimizer = optimizer
        self._warmup_iters = max(int(warmup_iters), 0)
        self._warmup_factor = warmup_factor
        # Each group's target LR (BASE_LR, BASE_LR*BACKBONE_MULTIPLIER, ...) as set
        # by Trainer.build_optimizer - ramp toward these, then let ReduceLROnPlateau
        # take it from here.
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._iter = 0
        if self._warmup_iters > 0:
            for g, base_lr in zip(optimizer.param_groups, self._base_lrs):
                g["lr"] = base_lr * warmup_factor
        min_lrs = [b * min_lr_fraction for b in self._base_lrs]
        self._sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lrs,
        )

    def step(self):
        """Per-iteration, called by detectron2's stock hooks.LRScheduler: drives
        the one-time warmup ramp only. Once warmup ends this does nothing - LR changes come
        exclusively from step_on_metric() from then on."""
        self._iter += 1
        if self._warmup_iters <= 0 or self._iter > self._warmup_iters:
            return
        t = self._iter / self._warmup_iters
        scale = self._warmup_factor + (1.0 - self._warmup_factor) * t
        for g, base_lr in zip(self._optimizer.param_groups, self._base_lrs):
            g["lr"] = base_lr * scale

    def step_on_metric(self, value):
        if self._iter < self._warmup_iters:
            return  # still ramping up - ignore plateau checks until warmup ends
        self._sched.step(value)

    def state_dict(self):
        # See module docstring: plateau/warmup state is intentionally not
        # checkpointed (hooks.LRScheduler only serializes real LRScheduler
        # instances, which this isn't).
        return {}

    def load_state_dict(self, state_dict):
        pass


class PlateauLRHook(HookBase):
    """Every SOLVER.PLATEAU.CHECK_PERIOD iterations (and on the final iteration),
    reads the average of SOLVER.PLATEAU.METRIC over that window from EventStorage
    - which only holds real values on the main process - and broadcasts it to
    every rank so PlateauLRScheduler.step_on_metric() runs identically everywhere.
    """

    def __init__(self, scheduler, metric_key, check_period):
        self._scheduler = scheduler
        self._metric_key = metric_key
        self._check_period = check_period
        self._warned_missing = False

    def after_step(self):
        if self._check_period <= 0:
            return
        next_iter = self.trainer.iter + 1
        is_last = next_iter >= self.trainer.max_iter
        if next_iter % self._check_period != 0 and not is_last:
            return

        value = None
        if comm.is_main_process():
            storage = self.trainer.storage
            if self._metric_key in storage.histories():
                hist = storage.history(self._metric_key)
                window = min(self._check_period, len(hist.values()))
                if window > 0:
                    value = float(hist.avg(window))
            if value is None and not self._warned_missing:
                logger.warning(
                    "[PlateauLRHook] no '%s' in EventStorage yet at iter %d - skipping check",
                    self._metric_key,
                    next_iter,
                )
                self._warned_missing = True

        # Broadcast rank 0's value (or None) to every rank so ReduceLROnPlateau's
        # internal patience/cooldown counters - and therefore every rank's
        # optimizer LR - stay identical across the DDP group. No-op / returns
        # [value] unchanged when running on a single process.
        value = comm.all_gather(value)[0]
        if value is not None:
            self._scheduler.step_on_metric(value)
