"""SGDR-style cosine annealing with warm restarts (Loshchilov & Hutter,
https://arxiv.org/abs/1608.03983), wired into detectron2's ParamScheduler /
LRMultiplier machinery the same way detectron2.solver.build.build_lr_scheduler wires
up its single-cycle WarmupCosineLR (see that function's "WarmupCosineLR" branch).
fvcore's own CosineParamScheduler docstring notes it implements only the annealing
half of SGDR, not the restarts - this module adds the restart half, without changing
anything upstream (WarmupParamScheduler / LRMultiplier are reused unchanged).

Restarts are "warm" by default (SGDR's own definition: snap straight back to
start_value, no re-ramp) but can optionally be softened via restart_warmup_factor /
restart_warmup_frac - a gentler ramp at the start of every cycle after the first.
This is a deliberate extension beyond the SGDR paper, not the same as start-of-training
warmup (SOLVER.WARMUP_*, handled by the outer WarmupParamScheduler in
build_warmup_cosine_restarts_lr_scheduler, which only ever fires once, before cycle 0).
"""
import math

from detectron2.solver import LRMultiplier, WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler, ParamScheduler


class CosineWithRestartsParamScheduler(ParamScheduler):
    """SGDR: repeating cosine decay cycles with warm restarts.

    Cycle 0 spans the first ``t_0`` of the ``[0, 1)`` ``where`` domain (``where`` is
    the fraction of training elapsed). Each following cycle is ``t_mult`` times
    longer than the one before it (``t_mult=1.0`` -> fixed-length repeating cycles;
    ``t_mult=2.0`` is the canonical SGDR setting). Within a cycle the value
    cosine-decays from ``start_value`` to ``end_value``.

    By default every restart (the start of cycle 1, 2, 3, ...) snaps straight back
    to ``start_value`` - a *warm* restart, the SGDR paper's own definition, no
    re-ramp. Passing ``restart_warmup_frac > 0`` softens that: the first
    ``restart_warmup_frac`` of *every* cycle after cycle 0 linearly ramps from
    ``restart_warmup_factor * start_value`` up to ``start_value`` instead of jumping
    there instantly, and the cosine decay itself then plays out fully across the
    remainder of that cycle (mirroring how the one-time ``WarmupParamScheduler`` at
    the very start of training works, just re-applied at every restart). Cycle 0 is
    deliberately left alone here - it's expected to go through a *separate*,
    one-time ``WarmupParamScheduler`` wrap for the true start of training (see
    ``build_warmup_cosine_restarts_lr_scheduler``), which is a different situation
    (freshly-initialized optimizer state / just-loaded pretrained weights) from a
    mid-training restart.

    If training ends mid-cycle, that cycle is simply left unfinished, same as
    stopping ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`` early.

    Cycle-boundary math mirrors ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts``
    (the ``T_mult`` branch of its ``step()``), translated from an absolute iteration
    count to a fractional ``where``: cycle ``n``'s cumulative start is the geometric
    sum ``t_0 * (t_mult**n - 1) / (t_mult - 1)``, inverted via a log to find which
    cycle a given ``where`` falls in.
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        t_0: float,  # cycle length as a fraction of train
        t_mult: float = 2.0,  # length increase per cycle
        restart_warmup_factor: float = 0.0,  # start-of-restart value, as a fraction of start_value
        restart_warmup_frac: float = 0.0,  # fraction of each restart cycle spent ramping; 0 = warm (no ramp)
    ) -> None:
        if not 0.0 < t_0 <= 1.0:
            raise ValueError(
                f"t_0 (first cycle length, as a fraction of training) must be in "
                f"(0, 1], got {t_0}"
            )
        if t_mult < 1.0:
            raise ValueError(f"t_mult must be >= 1.0, got {t_mult}")
        if not 0.0 <= restart_warmup_frac < 1.0:
            raise ValueError(
                f"restart_warmup_frac must be in [0, 1), got {restart_warmup_frac}"
            )
        self._start_value = start_value
        self._end_value = end_value
        self._t_0 = t_0
        self._t_mult = t_mult
        self._plain_cycle = CosineParamScheduler(start_value, end_value)
        self._restart_cycle = None
        if restart_warmup_frac > 0.0:
            # rescale_interval=True: after the ramp, the cosine decay is replayed in
            # full (start_value -> end_value) across the *remaining* fraction of the
            # cycle, so every softened restart still bottoms out at end_value by the
            # cycle's own end - only the ramp's own share of the cycle changes.
            self._restart_cycle = WarmupParamScheduler(
                self._plain_cycle,
                restart_warmup_factor,
                restart_warmup_frac,
                "linear",
                True,
            )

    def __call__(self, where: float) -> float:
        if self._t_mult == 1.0:
            n = math.floor(where / self._t_0)
            cycle_start = n * self._t_0
            cycle_len = self._t_0
        else:
            x = where / self._t_0 * (self._t_mult - 1.0) + 1.0
            n = math.floor(math.log(x, self._t_mult))
            cycle_start = self._t_0 * (self._t_mult**n - 1.0) / (self._t_mult - 1.0)
            cycle_len = self._t_0 * (self._t_mult**n)
        progress = (where - cycle_start) / cycle_len
        progress = min(max(progress, 0.0), 1.0)  # guard float edge cases at cycle boundaries
        if n == 0 or self._restart_cycle is None:
            return self._plain_cycle(progress)
        return self._restart_cycle(progress)


def build_warmup_cosine_restarts_lr_scheduler(cfg, optimizer):
    """Builds the SOLVER.LR_SCHEDULER_NAME == "WarmupCosineRestartsLR" scheduler:
    one linear warmup (SOLVER.WARMUP_*, same semantics as detectron2's stock
    WarmupCosineLR - fires once, before cycle 0) followed by SGDR restarts
    (SOLVER.COSINE_RESTARTS.*) decaying each cycle from BASE_LR to BASE_LR_END, with
    every restart after cycle 0 optionally softened by
    SOLVER.COSINE_RESTARTS.RESTART_WARMUP_FACTOR / RESTART_WARMUP_FRACTION (both
    default 0.0 = warm restarts, unchanged from the plain SGDR behavior). Mirrors
    detectron2.solver.build.build_lr_scheduler's WarmupCosineLR branch, swapping
    CosineParamScheduler for CosineWithRestartsParamScheduler; see
    train_net.Trainer.build_lr_scheduler for the LR_SCHEDULER_NAME dispatch.
    """
    end_value = cfg.SOLVER.BASE_LR_END / cfg.SOLVER.BASE_LR
    assert 0.0 <= end_value <= 1.0, end_value
    t_0 = cfg.SOLVER.COSINE_RESTARTS.T_0 / cfg.SOLVER.MAX_ITER
    sched = CosineWithRestartsParamScheduler(
        start_value=1.0,
        end_value=end_value,
        t_0=t_0,
        t_mult=cfg.SOLVER.COSINE_RESTARTS.T_MULT,
        restart_warmup_factor=cfg.SOLVER.COSINE_RESTARTS.RESTART_WARMUP_FACTOR,
        restart_warmup_frac=cfg.SOLVER.COSINE_RESTARTS.RESTART_WARMUP_FRACTION,
    )
    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
        cfg.SOLVER.RESCALE_INTERVAL,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)
