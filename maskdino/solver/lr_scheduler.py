"""SGDR-style cosine annealing with warm restarts (Loshchilov & Hutter,
https://arxiv.org/abs/1608.03983), wired into detectron2's ParamScheduler /
LRMultiplier machinery the same way detectron2.solver.build.build_lr_scheduler wires
up its single-cycle WarmupCosineLR (see that function's "WarmupCosineLR" branch).
fvcore's own CosineParamScheduler docstring notes it implements only the annealing
half of SGDR, not the restarts - this module adds the restart half, without changing
anything upstream (WarmupParamScheduler / LRMultiplier are reused unchanged).
"""
import math

from detectron2.solver import LRMultiplier, WarmupParamScheduler
from fvcore.common.param_scheduler import ParamScheduler


class CosineWithRestartsParamScheduler(ParamScheduler):
    """SGDR: repeating cosine decay cycles with warm restarts.
    """

    def __init__(
        self,
        start_value: float,
        end_value: float,
        t_0: float,  #cylce length as a fraction of train
        t_mult: float = 2.0, #length increase per cycle
    ) -> None:
        if not 0.0 < t_0 <= 1.0:
            raise ValueError(
                f"t_0 (first cycle length, as a fraction of training) must be in "
                f"(0, 1], got {t_0}"
            )
        if t_mult < 1.0:
            raise ValueError(f"t_mult must be >= 1.0, got {t_mult}")
        self._start_value = start_value
        self._end_value = end_value
        self._t_0 = t_0
        self._t_mult = t_mult

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
        return self._end_value + 0.5 * (self._start_value - self._end_value) * (
            1.0 + math.cos(math.pi * progress)
        )


def build_warmup_cosine_restarts_lr_scheduler(cfg, optimizer):
    """Builds the SOLVER.LR_SCHEDULER_NAME == "WarmupCosineRestartsLR" scheduler:
    one linear warmup (SOLVER.WARMUP_*, same semantics as detectron2's stock
    WarmupCosineLR) followed by SGDR restarts (SOLVER.COSINE_RESTARTS.*) decaying
    each cycle from BASE_LR to BASE_LR_END. Mirrors
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
    )
    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
        cfg.SOLVER.RESCALE_INTERVAL,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)
