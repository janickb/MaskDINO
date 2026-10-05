"""Regression test for the multi-GPU deadlock in Trainer.build_hooks().

ValidationLossHook and PlateauLRHook both run a comm.all_gather in after_step. Their
relative order therefore has to be identical on every rank: if rank 0 enters one
collective while rank 1 enters the other, the ranks mismatch and the job hangs until
gloo's 30-minute timeout, then dies with

    RuntimeError: Timed out waiting 1800000ms for send operation to complete

The original code anchored ValidationLossHook on hooks.PeriodicWriter, which detectron2
adds ONLY on the main process, so the two hooks ended up in opposite order on rank 0 vs
the rest. They first collide at lcm(TEST.EVAL_PERIOD, SOLVER.PLATEAU.CHECK_PERIOD) -
iteration 3000 for the surgical configs - which is exactly where a 2-GPU run died.
"""
import math

import pytest
from detectron2.engine import hooks

from train_net import insert_collective_hooks


class _FakeValLossHook(hooks.HookBase):
    pass


class _FakePlateauHook(hooks.HookBase):
    pass


def _base_hooks(is_main_process, eval_periods=(1000, 4000)):
    """What DefaultTrainer.build_hooks() produces, plus train_net's EvalHook split.

    Mirrors detectron2: PeriodicCheckpointer and PeriodicWriter are main-process only,
    EvalHook is added on every rank.
    """
    ret = [hooks.IterationTimer(), hooks.LRScheduler()]
    if is_main_process:
        ret.append(hooks.PeriodicCheckpointer.__new__(hooks.PeriodicCheckpointer))
    ret.extend(hooks.EvalHook(p, lambda: None) for p in eval_periods)
    if is_main_process:
        ret.append(hooks.PeriodicWriter.__new__(hooks.PeriodicWriter))
    return ret


def _collective_order(is_main_process):
    """Relative order of the two all_gather-running hooks, as assembled for one rank."""
    ret = insert_collective_hooks(_base_hooks(is_main_process), _FakeValLossHook())
    ret.append(_FakePlateauHook())  # build_hooks appends this last, on every rank
    return [
        type(h).__name__
        for h in ret
        if isinstance(h, (_FakeValLossHook, _FakePlateauHook, hooks.EvalHook))
    ]


def test_collective_hook_order_is_identical_across_ranks():
    main = _collective_order(is_main_process=True)
    worker = _collective_order(is_main_process=False)
    assert main == worker, (
        "collective hooks are ordered differently on rank 0 vs the other ranks; "
        f"rank0={main} other={worker}. This deadlocks DDP at "
        "lcm(TEST.EVAL_PERIOD, SOLVER.PLATEAU.CHECK_PERIOD)."
    )


def test_val_loss_hook_runs_before_the_plateau_hook():
    """PlateauLRHook reads the metric window from EventStorage, so the validation-loss
    values must already be written when it runs."""
    order = _collective_order(is_main_process=True)
    assert order.index("_FakeValLossHook") < order.index("_FakePlateauHook")


def test_val_loss_hook_lands_after_every_eval_hook():
    """Anchored on the last EvalHook - present on every rank - not on PeriodicWriter,
    which exists only on rank 0."""
    for is_main in (True, False):
        ret = insert_collective_hooks(_base_hooks(is_main), _FakeValLossHook())
        idx_val = next(i for i, h in enumerate(ret) if isinstance(h, _FakeValLossHook))
        last_eval = max(i for i, h in enumerate(ret) if isinstance(h, hooks.EvalHook))
        assert idx_val == last_eval + 1


def test_no_eval_hook_still_inserts():
    ret = insert_collective_hooks([hooks.IterationTimer()], _FakeValLossHook())
    assert isinstance(ret[-1], _FakeValLossHook)


def test_the_colliding_iteration_is_what_we_think():
    """Documents why the failure looked intermittent: the two hooks only fire together
    at the lcm of their periods."""
    assert math.lcm(1000, 300) == 3000
