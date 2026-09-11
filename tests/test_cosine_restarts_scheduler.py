"""Unit tests for maskdino/solver/lr_scheduler.py's CosineWithRestartsParamScheduler.

The module is loaded by file path (like tests/test_class_mapping.py) so the tests
don't trigger `import maskdino` (which registers datasets from absolute data paths).
lr_scheduler.py only imports detectron2/fvcore/math, so this is safe either way, but
file-path loading keeps the pattern consistent with the rest of this test suite.
"""
import importlib.util
import math
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "maskdino", "solver", "lr_scheduler.py")
_spec = importlib.util.spec_from_file_location("lr_scheduler", _MOD_PATH)
lr_sched_mod = importlib.util.module_from_spec(_spec)
sys.modules["lr_scheduler"] = lr_sched_mod
_spec.loader.exec_module(lr_sched_mod)

CosineWithRestartsParamScheduler = lr_sched_mod.CosineWithRestartsParamScheduler


def _cosine(start, end, progress):
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def test_growing_cycles_boundaries_match_geometric_series():
    # t_0=0.1, t_mult=2.0 -> cumulative cycle starts 0, 0.1, 0.3, 0.7, 1.5, ...
    sched = CosineWithRestartsParamScheduler(start_value=1.0, end_value=0.0, t_0=0.1, t_mult=2.0)
    boundaries = [0.0, 0.1, 0.3, 0.7]
    for b in boundaries:
        # just after a boundary, value should be back near start_value (a restart)
        assert sched(b + 1e-9) == pytest.approx(1.0, abs=1e-4)
    # just before the next boundary, value should be near end_value (bottom of the cycle)
    for b in boundaries[1:]:
        assert sched(b - 1e-9) == pytest.approx(0.0, abs=1e-4)


def test_growing_cycles_midpoint_matches_formula():
    sched = CosineWithRestartsParamScheduler(start_value=1.0, end_value=0.0, t_0=0.1, t_mult=2.0)
    # cycle 1 spans [0.1, 0.3), length 0.2; its midpoint is where=0.2, local progress=0.5
    assert sched(0.2) == pytest.approx(_cosine(1.0, 0.0, 0.5), abs=1e-9)
    # cycle 2 spans [0.3, 0.7), length 0.4; its midpoint is where=0.5, local progress=0.5
    assert sched(0.5) == pytest.approx(_cosine(1.0, 0.0, 0.5), abs=1e-9)


def test_fixed_length_cycles_when_t_mult_is_one():
    sched = CosineWithRestartsParamScheduler(start_value=1.0, end_value=0.2, t_0=0.25, t_mult=1.0)
    for k in range(4):
        start = k * 0.25
        assert sched(start + 1e-9) == pytest.approx(1.0, abs=1e-4)
        assert sched(start + 0.25 - 1e-9) == pytest.approx(0.2, abs=1e-4)
        # local progress 0.5 inside cycle k
        assert sched(start + 0.125) == pytest.approx(_cosine(1.0, 0.2, 0.5), abs=1e-9)


def test_single_cycle_matches_plain_cosine_when_t_0_is_1():
    # t_0=1.0 -> only cycle 0 ever runs within [0, 1), i.e. plain CosineParamScheduler
    sched = CosineWithRestartsParamScheduler(start_value=1.0, end_value=0.01, t_0=1.0, t_mult=2.0)
    for where in (0.0, 0.1, 0.5, 0.9, 0.999):
        assert sched(where) == pytest.approx(_cosine(1.0, 0.01, where), abs=1e-9)


def test_invalid_t_0_and_t_mult_raise():
    with pytest.raises(ValueError):
        CosineWithRestartsParamScheduler(start_value=1.0, end_value=0.0, t_0=0.0, t_mult=2.0)
    with pytest.raises(ValueError):
        CosineWithRestartsParamScheduler(start_value=1.0, end_value=0.0, t_0=1.5, t_mult=2.0)
    with pytest.raises(ValueError):
        CosineWithRestartsParamScheduler(start_value=1.0, end_value=0.0, t_0=0.1, t_mult=0.5)
