"""Unit tests for maskdino/data/datasets/images_root.py.

Loaded by file path (like tests/test_class_mapping.py) so the tests don't trigger
``import maskdino`` (which registers datasets). HOME is pointed at tmp_path, so
the real ~/training/images_* dirs never leak in.
"""
import importlib.util
import os

import pytest

_MOD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "maskdino", "data", "datasets", "images_root.py"
)
_spec = importlib.util.spec_from_file_location("images_root", _MOD_PATH)
images_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(images_root)

resolve_images_root = images_root.resolve_images_root
ENV_VAR = images_root.ENV_VAR


@pytest.fixture
def training(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(ENV_VAR, raising=False)
    d = tmp_path / "training"
    d.mkdir()
    return d


def test_single_dir_is_picked(training):
    (training / "images_dgx-a100").mkdir()
    assert resolve_images_root() == str(training / "images_dgx-a100")


def test_no_dir_returns_none(training):
    assert resolve_images_root() is None


def test_several_dirs_without_override_raise(training):
    (training / "images_dgx-a100").mkdir()
    (training / "images_babylambda1-2080").mkdir()
    with pytest.raises(RuntimeError) as exc:
        resolve_images_root()
    assert "images_dgx-a100" in str(exc.value)
    assert "images_babylambda1-2080" in str(exc.value)
    assert ENV_VAR in str(exc.value)


def test_override_wins_over_several_dirs(training, monkeypatch):
    (training / "images_dgx-a100").mkdir()
    (training / "images_babylambda1-2080").mkdir()
    monkeypatch.setenv(ENV_VAR, str(training / "images_babylambda1-2080"))
    assert resolve_images_root() == str(training / "images_babylambda1-2080")


def test_override_to_missing_dir_raises(training, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(training / "images_typo"))
    with pytest.raises(FileNotFoundError):
        resolve_images_root()


def test_plain_file_is_not_a_candidate(training):
    (training / "images_dgx-a100").mkdir()
    (training / "images_notes.txt").write_text("")
    assert resolve_images_root() == str(training / "images_dgx-a100")
