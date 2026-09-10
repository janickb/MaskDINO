"""Unit tests for tools/summarize_reclass.py - no GPU, no model, synthetic
confusion matrices written to tmp_path in the exact format
HungarianInstanceEvaluator._write_artifacts produces.

Class ids are the compact 0..N-1 space (background / unused_* already dropped at
dataset registration), so id 0 is a real class.
"""
import csv
import importlib.util
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "tools", "summarize_reclass.py")
_spec = importlib.util.spec_from_file_location("summarize_reclass", _MOD_PATH)
sr = importlib.util.module_from_spec(_spec)
sys.modules["summarize_reclass"] = sr
_spec.loader.exec_module(sr)

FN_COL = "(false negative)"
FP_ROW = "(false positive)"


def write_confusion(path, names, mat):
    """names: C class names. mat: (C+1, C+1) int array (rows GT+FP, cols pred+FN)."""
    c = len(names)
    assert mat.shape == (c + 1, c + 1)
    col_names = names + [FN_COL]
    row_names = names + [FP_ROW]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gt\\pred"] + col_names)
        for i in range(c + 1):
            w.writerow([row_names[i]] + mat[i].tolist())


# compact ids: newA=0, newB=1, newC=2, oldX=3, oldY=4
NAMES = ["newA", "newB", "newC", "oldX", "oldY"]
EXISTING = [3, 4]


def _messy_mat():
    c = len(NAMES)
    m = np.zeros((c + 1, c + 1), dtype=np.int64)
    m[0, 0] = 10                     # newA: perfectly clean
    m[1, 1] = 3; m[1, 2] = 7        # newB: 70% leaks to newC  -> NOT distinct
    m[2, 2] = 6; m[2, 3] = 4        # newC: 40% leaks to oldX  -> collapsed onto existing
    m[3, 3] = 9; m[4, 4] = 9        # existing classes fine
    return m


def _clean_mat():
    c = len(NAMES)
    m = np.zeros((c + 1, c + 1), dtype=np.int64)
    for i in range(c):
        m[i, i] = 10
    return m


def test_load_confusion_roundtrip(tmp_path):
    p = tmp_path / "instance_matching_confusion.csv"
    write_confusion(p, NAMES, _messy_mat())
    mat, names = sr.load_confusion(str(p))
    assert names == NAMES
    assert mat.shape == (len(NAMES) + 1, len(NAMES) + 1)
    assert mat[1, 2] == 7


def test_infer_new_ids_includes_id0_skips_existing_unused_and_empty():
    names = ["newA", "unused_1", "newC", "oldX"]
    c = len(names)
    m = np.zeros((c + 1, c + 1), dtype=np.int64)
    m[0, 0] = 5          # newA (id 0) has GT -> new, no longer skipped
    m[1, 1] = 0          # unused_1, no GT anyway
    m[2, 2] = 5          # newC has GT
    m[3, 3] = 5          # oldX existing
    assert sr.infer_new_ids(m, names, {3}) == [0, 2]


def test_messy_flags_not_distinct_and_collapse(tmp_path):
    p = tmp_path / "instance_matching_confusion.csv"
    write_confusion(p, NAMES, _messy_mat())
    report, not_distinct = sr.summarize(str(p), None, EXISTING, confuse_thresh=0.10)

    assert not_distinct == [("newB", "newC", pytest.approx(0.7), 7)]
    assert "VERDICT: new classes are NOT DISTINCT" in report
    assert "new classes whose dominant confusion is a pre-existing class" in report
    assert "newC" in report and "oldX" in report


def test_clean_is_distinct(tmp_path):
    p = tmp_path / "instance_matching_confusion.csv"
    write_confusion(p, NAMES, _clean_mat())
    report, not_distinct = sr.summarize(str(p), None, EXISTING, confuse_thresh=0.10)
    assert not_distinct == []
    assert "VERDICT: new classes are DISTINCT" in report


def test_thresh_controls_flagging(tmp_path):
    p = tmp_path / "instance_matching_confusion.csv"
    write_confusion(p, NAMES, _messy_mat())
    _, nd_hi = sr.summarize(str(p), None, EXISTING, confuse_thresh=0.80)
    assert nd_hi == []


def test_explicit_new_ids_override(tmp_path):
    p = tmp_path / "instance_matching_confusion.csv"
    write_confusion(p, NAMES, _messy_mat())
    report, nd = sr.summarize(str(p), [0], EXISTING, confuse_thresh=0.10)
    assert nd == []
    assert "['newA']" in report


def test_existing_names_marks_pre_existing(tmp_path):
    p = tmp_path / "instance_matching_confusion.csv"
    write_confusion(p, NAMES, _clean_mat())
    # phase-1 knew oldX/oldY -> treat them as existing without passing ids
    report, nd = sr.summarize(
        str(p), None, [], confuse_thresh=0.10, existing_names={"oldX", "oldY"}
    )
    assert "new (adapted) classes: ['newA', 'newB', 'newC']" in report
    assert nd == []


def test_resolve_csv_prefers_dataset_subdir(tmp_path):
    run = tmp_path / "runs" / "reclassify_v1"
    (run / "inference" / "reclass_val_pile").mkdir(parents=True)
    (run / "inference" / "reclass_val_sparse").mkdir(parents=True)
    for sub in ("reclass_val_pile", "reclass_val_sparse"):
        write_confusion(
            run / "inference" / sub / "instance_matching_confusion.csv",
            NAMES, _clean_mat(),
        )

    class A:
        csv = None
        run_dir = str(run)
        dataset = "reclass_val_pile"

    got = sr.resolve_csv(A())
    assert got.endswith(os.path.join("reclass_val_pile", "instance_matching_confusion.csv"))
