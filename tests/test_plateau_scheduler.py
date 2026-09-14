"""Unit tests for maskdino/solver/plateau.py's PlateauLRScheduler / PlateauLRHook.

The module is loaded by file path (like tests/test_cosine_restarts_scheduler.py) so
the tests don't trigger `import maskdino` (which registers datasets from absolute
data paths). plateau.py only imports torch/detectron2, so this is safe either way,
but file-path loading keeps the pattern consistent with the rest of this suite.
"""
import importlib.util
import os
import sys

import pytest
import torch
from detectron2.utils.events import EventStorage

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "maskdino", "solver", "plateau.py")
_spec = importlib.util.spec_from_file_location("plateau", _MOD_PATH)
plateau_mod = importlib.util.module_from_spec(_spec)
sys.modules["plateau"] = plateau_mod
_spec.loader.exec_module(plateau_mod)

PlateauLRScheduler = plateau_mod.PlateauLRScheduler
PlateauLRHook = plateau_mod.PlateauLRHook

BACKBONE_MULT = 0.1


def _make_optimizer(base_lr=0.1):
    p_head = torch.nn.Parameter(torch.zeros(1))
    p_backbone = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW(
        [
            {"params": [p_head], "lr": base_lr},
            {"params": [p_backbone], "lr": base_lr * BACKBONE_MULT},
        ]
    )


def _lrs(opt):
    return opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]


def _ratio_ok(opt):
    head, backbone = _lrs(opt)
    assert backbone == pytest.approx(head * BACKBONE_MULT, rel=1e-6)


def test_warmup_ramps_then_holds():
    opt = _make_optimizer(base_lr=0.1)
    sched = PlateauLRScheduler(
        opt, mode="min", factor=0.5, patience=2, threshold=1e-4, cooldown=0,
        min_lr_fraction=0.0, warmup_iters=10, warmup_factor=0.1,
    )
    head, _ = _lrs(opt)
    assert head == pytest.approx(0.01)  # 0.1 * warmup_factor, applied at construction
    _ratio_ok(opt)

    for _ in range(5):
        sched.step()
    head, _ = _lrs(opt)
    assert 0.01 < head < 0.1
    _ratio_ok(opt)

    for _ in range(5):
        sched.step()
    head, _ = _lrs(opt)
    assert head == pytest.approx(0.1)  # fully ramped to base_lr after warmup_iters steps
    _ratio_ok(opt)

    # holds after warmup ends - no further per-iteration drift
    for _ in range(20):
        sched.step()
    head, _ = _lrs(opt)
    assert head == pytest.approx(0.1)


def test_step_on_metric_ignored_during_warmup():
    opt = _make_optimizer(base_lr=0.1)
    sched = PlateauLRScheduler(
        opt, mode="min", factor=0.1, patience=0, threshold=1e-4, cooldown=0,
        min_lr_fraction=0.0, warmup_iters=10, warmup_factor=0.1,
    )
    for _ in range(5):
        sched.step()
    before = _lrs(opt)
    # even a terrible metric shouldn't trigger a reduction mid-warmup
    for _ in range(5):
        sched.step_on_metric(1e9)
    assert _lrs(opt) == before


def test_plateau_reduces_after_patience_checks_with_no_improvement():
    opt = _make_optimizer(base_lr=0.1)
    sched = PlateauLRScheduler(
        opt, mode="min", factor=0.5, patience=2, threshold=1e-4, cooldown=0,
        min_lr_fraction=0.0, warmup_iters=0, warmup_factor=1.0,
    )
    _ratio_ok(opt)
    head0, _ = _lrs(opt)
    assert head0 == pytest.approx(0.1)

    sched.step_on_metric(1.0)  # first value: always "best" so far -> num_bad_epochs=0
    assert _lrs(opt)[0] == pytest.approx(0.1)
    sched.step_on_metric(1.0)  # no improvement -> num_bad_epochs=1 (1 > patience=2? no)
    assert _lrs(opt)[0] == pytest.approx(0.1)
    sched.step_on_metric(1.0)  # no improvement -> num_bad_epochs=2 (2 > patience=2? no)
    assert _lrs(opt)[0] == pytest.approx(0.1)
    sched.step_on_metric(1.0)  # no improvement -> num_bad_epochs=3 (3 > patience=2? yes) -> reduce
    head, _ = _lrs(opt)
    assert head == pytest.approx(0.05)  # 0.1 * factor
    _ratio_ok(opt)  # backbone group reduced in lockstep, ratio preserved


def test_plateau_does_not_reduce_while_improving():
    opt = _make_optimizer(base_lr=0.1)
    sched = PlateauLRScheduler(
        opt, mode="min", factor=0.5, patience=1, threshold=1e-4, cooldown=0,
        min_lr_fraction=0.0, warmup_iters=0, warmup_factor=1.0,
    )
    for value in [1.0, 0.9, 0.8, 0.7, 0.6]:
        sched.step_on_metric(value)
    assert _lrs(opt)[0] == pytest.approx(0.1)


def test_hook_noop_when_check_period_disabled():
    calls = []
    hook = PlateauLRHook(
        scheduler=type("S", (), {"step_on_metric": lambda self, v: calls.append(v)})(),
        metric_key="total_loss",
        check_period=0,
    )
    hook.trainer = type("T", (), {"iter": 0, "max_iter": 100, "storage": EventStorage()})()
    for i in range(50):
        hook.trainer.iter = i
        hook.after_step()
    assert calls == []


def test_hook_fires_at_check_period_and_final_iter_with_windowed_average():
    calls = []
    scheduler = type("S", (), {"step_on_metric": lambda self, v: calls.append(v)})()
    hook = PlateauLRHook(scheduler=scheduler, metric_key="total_loss", check_period=5)
    storage = EventStorage()
    trainer = type("T", (), {"iter": 0, "max_iter": 12, "storage": storage})()
    hook.trainer = trainer

    for i in range(12):
        storage.put_scalar("total_loss", float(i), cur_iter=i)  # values 0..11
        trainer.iter = i
        hook.after_step()

    # checks fire at next_iter=5, next_iter=10 (both multiples of 5), and at the
    # final iteration (next_iter=12, trainer.max_iter=12) even though it isn't a
    # multiple of check_period.
    assert len(calls) == 3
    assert calls[0] == pytest.approx(sum(range(0, 5)) / 5)  # avg of values 0..4
    assert calls[1] == pytest.approx(sum(range(5, 10)) / 5)  # avg of values 5..9
    # final check's window is min(check_period, len(history)) = 5 most recent: 7..11
    assert calls[2] == pytest.approx(sum(range(7, 12)) / 5)


def test_hook_skips_silently_when_metric_never_written():
    calls = []
    scheduler = type("S", (), {"step_on_metric": lambda self, v: calls.append(v)})()
    hook = PlateauLRHook(scheduler=scheduler, metric_key="total_loss", check_period=5)
    storage = EventStorage()  # nothing ever put into it
    trainer = type("T", (), {"iter": 4, "max_iter": 100, "storage": storage})()
    hook.trainer = trainer
    hook.after_step()  # next_iter=5, a check boundary, but no history yet
    assert calls == []
