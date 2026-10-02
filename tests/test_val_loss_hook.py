"""Unit tests for maskdino/solver/val_loss.py's ValidationLossHook.

Loaded by file path (like tests/test_plateau_scheduler.py) so the tests don't
trigger `import maskdino` (which registers datasets from absolute data paths).
val_loss.py only imports torch/detectron2, so this is safe either way, but
file-path loading keeps the pattern consistent with the rest of this suite.
"""
import ast
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


    assert storage.history("validation_loss").latest() == 6.0
    assert [k for k in storage.histories() if k.startswith("val_")] == []
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


# ---------------------------------------------------------------------------
# DDP: the training-mode loss runs torch.distributed.all_reduce(num_masks) inside
# SetCriterion.forward, so one forward pass == one collective.
# Every rank must therefore run the SAME number of val batches, or the collectives
# mismatch and the job deadlocks until gloo's 30-minute timeout.
#
# build_detection_test_loader shards the val set with InferenceSampler, which gives the
# remainder to the lower ranks - so shards are uneven whenever len(dataset) is not
# divisible by the world size (3 images over 2 ranks -> 2 and 1). Observed in practice
# as: rank 0 pinned at 100% GPU inside all_reduce, rank 1 idle, both logs frozen.
# ---------------------------------------------------------------------------


class _CountingLoader:
    """Loader of `n` dummy batches that records how many were consumed."""

    def __init__(self, n):
        self._n = n
        self.consumed = 0

    def __len__(self):
        return self._n

    def __iter__(self):
        for _ in range(self._n):
            self.consumed += 1
            yield [{}]


def _hook_with(loader, gathered_lengths, monkeypatch):
    # val_loss_mod is the file-path-loaded module from the top of this file; importing
    # maskdino.solver.val_loss instead would pull in the maskdino package.
    hook = val_loss_mod.ValidationLossHook(period=1, loader=loader)
    monkeypatch.setattr(val_loss_mod.comm, "all_gather", lambda v: list(gathered_lengths))
    return hook


def test_every_rank_runs_the_global_minimum_batch_count(monkeypatch):
    """This rank has 2 batches, the other has 1 -> both must run 1."""
    loader = _CountingLoader(2)
    hook = _hook_with(loader, [2, 1], monkeypatch)
    assert hook._steps_this_rank() == 1


def test_even_shards_are_not_truncated(monkeypatch):
    loader = _CountingLoader(128)
    hook = _hook_with(loader, [128, 128], monkeypatch)
    assert hook._steps_this_rank() == 128


def test_single_process_is_unaffected(monkeypatch):
    loader = _CountingLoader(7)
    hook = _hook_with(loader, [7], monkeypatch)
    assert hook._steps_this_rank() == 7


def test_loader_without_len_disables_truncation(monkeypatch):
    class _NoLen:
        def __iter__(self):
            return iter([])

    hook = val_loss_mod.ValidationLossHook(period=1, loader=_NoLen())
    assert hook._steps_this_rank() is None


def test_criterion_really_contains_a_collective():
    """If this ever stops being true the truncation above is no longer needed - but
    while it holds, unequal batch counts across ranks are a hard deadlock.

    Parsed from source rather than imported: maskdino/modeling/criterion.py uses
    relative imports and pulls in maskdino.utils, so reaching SetCriterion means
    `import maskdino` - which registers datasets from absolute data paths and is what
    this suite's file-path loading exists to avoid (see the module docstring).
    """
    src = os.path.join(_HERE, "..", "maskdino", "modeling", "criterion.py")
    with open(src) as fh:
        tree = ast.parse(fh.read(), filename=src)

    forward = next(
        (
            fn
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "SetCriterion"
            for fn in cls.body
            if isinstance(fn, ast.FunctionDef) and fn.name == "forward"
        ),
        None,
    )
    assert forward is not None, f"SetCriterion.forward not found in {src}"
    calls = {
        ast.unparse(node.func)
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
    }
    assert any("all_reduce" in c for c in calls), (
        "SetCriterion.forward no longer runs a collective, so ValidationLossHook's "
        "per-rank truncation (_steps_this_rank) may no longer be needed"
    )
