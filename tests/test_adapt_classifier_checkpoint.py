"""Unit tests for tools/adapt_classifier_checkpoint.py::adapt_class_head.

Both modules are loaded by file path so the tests don't trigger
``import maskdino``. Only ``adapt_class_head`` (pure torch) is exercised;
``main()``'s dataset-name resolution is not.
"""
import importlib.util
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(__file__)


def _load(name, relpath):
    path = os.path.join(_HERE, "..", *relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm_mod = _load("class_mapping", ["maskdino", "data", "class_mapping.py"])
adapt_mod = _load("adapt_classifier_checkpoint", ["tools", "adapt_classifier_checkpoint.py"])

adapt_class_head = adapt_mod.adapt_class_head
derive_class_mapping = cm_mod.derive_class_mapping

_W = "sem_seg_head.predictor.class_embed.weight"
_B = "sem_seg_head.predictor.class_embed.bias"
_E = "sem_seg_head.predictor.label_enc.weight"
_EW = "criterion.empty_weight"

SETB = ["background"] + [f"unused_{i}" for i in range(1, 11)] + [
    "adapter-11", "clamp-11-fusion", "forcep-11-fusion", "hammer-11",
    "scalpel-11-fusion", "scissor-11-fusion", "scissor-12-fusion", "tweezer-11-fusion",
]
SETAB = [
    "background", "forcep01", "forcep02", "forcep03", "forcep04", "scalpel01",
    "hammer01", "unused_7", "sharpspoon01", "scarstick01", "unused_10",
    "adapter-11", "clamp-11-fusion", "forcep-11-fusion", "hammer-11",
    "scalpel-11-fusion", "scissor-11-fusion", "scissor-12-fusion", "tweezer-11-fusion",
]


def _synthetic_sd(n=19, hidden=4):
    """A 19-wide source head: canonical category_id i lives at physical row i.
    Rows are made identifiable (row i is all i.0)."""
    return {
        _W: torch.arange(n, dtype=torch.float32)[:, None].repeat(1, hidden).clone(),
        _B: torch.arange(n, dtype=torch.float32).clone(),
        _E: (torch.arange(n, dtype=torch.float32)[:, None].repeat(1, hidden) + 0.5).clone(),
        _EW: torch.ones(n + 1),
        "backbone.stem.conv1.weight": torch.randn(2, 2),  # unrelated, must survive
    }


def test_to_setab_shapes_and_carry():
    sd = _synthetic_sd()
    unrelated_before = sd["backbone.stem.conv1.weight"].clone()
    cm_t = derive_class_mapping(SETAB)  # N=16
    src_c2r = {c: c for c in range(11, 19)}  # source head: canonical id i lives at row i

    carried, fresh = adapt_class_head(sd, src_c2r, cm_t, seed=0, eos_coef=0.1)

    assert carried == [11, 12, 13, 14, 15, 16, 17, 18]
    assert fresh == [1, 2, 3, 4, 5, 6, 8, 9]
    assert tuple(sd[_W].shape) == (16, 4)
    assert tuple(sd[_B].shape) == (16,)
    assert tuple(sd[_E].shape) == (16, 4)
    assert tuple(sd[_EW].shape) == (17,)
    assert sd[_EW][-1] == pytest.approx(0.1)
    assert sd[_EW][:-1].eq(1).all()
    assert torch.equal(sd["backbone.stem.conv1.weight"], unrelated_before)


def test_carried_rows_are_bit_identical():
    sd = _synthetic_sd()
    cm_t = derive_class_mapping(SETAB)
    src_c2r = {c: c for c in range(11, 19)}
    adapt_class_head(sd, src_c2r, cm_t)

    # target contiguous 8..15 <- canonical 11..18 <- source rows 11..18
    for j, canonical in zip(range(8, 16), range(11, 19)):
        assert sd[_W][j].eq(float(canonical)).all()
        assert sd[_B][j] == float(canonical)
        assert sd[_E][j].eq(canonical + 0.5).all()


def test_fresh_rows_differ_from_every_source_row():
    sd = _synthetic_sd()
    src_rows_before = sd[_W].clone()
    cm_t = derive_class_mapping(SETAB)
    adapt_class_head(sd, {c: c for c in range(11, 19)}, cm_t, seed=0)
    # set-A rows are contiguous 0..7; each source row is a constant vector [i,i,i,i]
    for j in range(8):
        row = sd[_W][j]
        assert not torch.equal(row, row[0].repeat(row.numel()))  # not constant -> not copied
        assert not any(torch.equal(row, src_rows_before[k]) for k in range(19))


def test_seed_determinism():
    a, b = _synthetic_sd(), _synthetic_sd()
    cm_t = derive_class_mapping(SETAB)
    adapt_class_head(a, {c: c for c in range(11, 19)}, cm_t, seed=7)
    adapt_class_head(b, {c: c for c in range(11, 19)}, cm_t, seed=7)
    assert torch.equal(a[_W], b[_W])
    assert torch.equal(a[_E], b[_E])


def test_native_compact_source_8_to_16():
    """Source is itself compact (N=8, rows 0..7 == set-B), like a future phase-1."""
    sd = _synthetic_sd(n=8)  # rows now mean the compact contiguous ids 0..7
    cm_src = derive_class_mapping(SETB)  # {11:0, ..., 18:7}
    cm_t = derive_class_mapping(SETAB)
    carried, fresh = adapt_class_head(
        sd, dict(cm_src.thing_dataset_id_to_contiguous_id), cm_t
    )
    assert carried == [11, 12, 13, 14, 15, 16, 17, 18]
    assert fresh == [1, 2, 3, 4, 5, 6, 8, 9]
    # target 8..15 <- source rows 0..7
    for j, row in zip(range(8, 16), range(8)):
        assert sd[_W][j].eq(float(row)).all()


def test_missing_key_raises():
    sd = _synthetic_sd()
    del sd[_E]
    with pytest.raises(KeyError):
        adapt_class_head(sd, {11: 11}, derive_class_mapping(SETAB))


def test_row_out_of_range_raises():
    sd = _synthetic_sd(n=8)
    with pytest.raises(ValueError):
        adapt_class_head(sd, {11: 11}, derive_class_mapping(SETAB))  # row 11 >= 8
