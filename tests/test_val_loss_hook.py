"""Unit tests for maskdino/solver/val_loss.py's ValidationLossHook.

Loaded by file path (like tests/test_plateau_scheduler.py) so the tests don't
trigger `import maskdino` (which registers datasets from absolute data paths).
val_loss.py only imports torch/detectron2, so this is safe either way, but
file-path loading keeps the pattern consistent with the rest of this suite.
"""
import importlib.util
import os
import sys

import torch
from detectron2.utils.events import EventStorage

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "maskdino", "solver", "val_loss.py")
_spec = importlib.util.spec_from_file_location("val_loss", _MOD_PATH)
val_loss_mod = importlib.util.module_from_spec(_spec)
sys.modules["val_loss"] = val_loss_mod
_spec.loader.exec_module(val_loss_mod)

ValidationLossHook = val_loss_mod.ValidationLossHook


class _DummyModel(torch.nn.Module):
    """Returns a fixed loss dict per batch when in training mode, like MaskDINO's
    forward() does - the hook flips the model into .train() to reach this branch.
    """

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        assert self.training, "hook must call the model in train() mode"
        # one "batch" == one scalar loss contribution, keyed by its value so the
        # test can check the exact mean.
        return {"loss_ce": torch.tensor(float(batch)), "loss_mask": torch.tensor(float(batch) * 2)}


class _FakeTrainer:
    def __init__(self, model, storage, iter_, max_iter):
        self.model = model
        self.storage = storage
        self.iter = iter_
        self.max_iter = max_iter


def _make_hook(loader, period=5):
    hook = ValidationLossHook(period, loader)
    return hook


def test_skips_until_period_boundary():
    model = _DummyModel()
    model.eval()
    storage = EventStorage()
    hook = _make_hook(loader=[1, 2, 3], period=5)
    with storage:
        for it in range(4):  # next_iter = 1..4, none hit period 5 and none is last (max_iter=100)
            hook.trainer = _FakeTrainer(model, storage, it, 100)
            hook.after_step()
    assert "validation_loss" not in storage.histories()
    assert model.training is False  # untouched - hook never ran


def test_computes_mean_and_restores_eval_mode():
    model = _DummyModel()
    model.eval()
    storage = EventStorage()
    loader = [1.0, 2.0, 3.0]  # loss_ce mean = 2.0, loss_mask mean = 4.0
    hook = _make_hook(loader=loader, period=5)
    with storage:
        hook.trainer = _FakeTrainer(model, storage, 4, 100)  # next_iter = 5 -> boundary
        hook.after_step()

    assert storage.history("val_loss_ce").latest() == 2.0
    assert storage.history("val_loss_mask").latest() == 4.0
    assert storage.history("validation_loss").latest() == 6.0
    # model was .eval() before the check and must be restored to .eval() after
    assert model.training is False


def test_runs_on_final_iteration_even_off_period():
    model = _DummyModel()
    model.train()
    storage = EventStorage()
    loader = [10.0]
    hook = _make_hook(loader=loader, period=1000)
    with storage:
        hook.trainer = _FakeTrainer(model, storage, 98, 99)  # next_iter = 99 == max_iter
        hook.after_step()
    assert storage.history("validation_loss").latest() == 30.0
    # model was .train() before the check and must be restored to .train() after
    assert model.training is True


def test_disabled_when_period_non_positive():
    model = _DummyModel()
    storage = EventStorage()
    hook = _make_hook(loader=[1.0], period=0)
    with storage:
        hook.trainer = _FakeTrainer(model, storage, 999, 1000)
        hook.after_step()
    assert "validation_loss" not in storage.histories()


def test_empty_loader_warns_and_skips():
    model = _DummyModel()
    storage = EventStorage()
    hook = _make_hook(loader=[], period=1)
    with storage:
        hook.trainer = _FakeTrainer(model, storage, 0, 100)
        hook.after_step()
    assert "validation_loss" not in storage.histories()
