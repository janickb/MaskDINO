"""Step-0 verification: the classifier config loads and rejects bad values."""

import dataclasses

import pytest
import yaml

from cropcls.config import PIXEL_MEAN, PIXEL_STD, CropClsConfig, load_config

CONFIG = "configs/cropcls/phase1_seta.yaml"


def test_shipped_config_loads_and_paths_exist():
    cfg = load_config(CONFIG)
    assert cfg.crop_size == 224
    assert cfg.train_dataset == "train_seta_mm"
    assert cfg.val_dataset == "val_seta_mm"
    assert cfg.iou_floor == 0.5
    assert cfg.runner_up_ceiling == 0.2
    assert cfg.rotations == [0, 90, 180, 270]


def test_detector_config_defaults_to_the_runs_own_config_yaml():
    cfg = load_config(CONFIG)
    assert cfg.detector_config.endswith("/config.yaml")
    assert "20261007130733_single_class_v2" in cfg.detector_config


def test_pixel_statistics_are_maskdinos():
    cfg = load_config(CONFIG)
    assert cfg.pixel_mean == PIXEL_MEAN
    assert cfg.pixel_std == PIXEL_STD


def test_unknown_key_is_an_error_not_silently_ignored(tmp_path):
    p = tmp_path / "bad.yaml"
    raw = yaml.safe_load(open(CONFIG))
    raw["crop_siz"] = 224  # typo
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(p)


def test_class_count_is_not_a_config_field():
    """It is len(thing_classes) of train_dataset; stating it would be a second copy."""
    assert "num_classes" not in {f.name for f in dataclasses.fields(CropClsConfig)}
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(CONFIG, {"num_classes": 8})


def test_non_90_rotation_is_rejected():
    cfg = load_config(CONFIG)
    cfg.rotations = [0, 45]
    with pytest.raises(ValueError, match="multiples of 90"):
        cfg.validate()


def test_missing_checkpoint_is_reported_at_startup():
    cfg = CropClsConfig(detector_checkpoint="/nope/model_final.pth")
    with pytest.raises(FileNotFoundError, match="detector_checkpoint"):
        cfg.validate()


def test_classifier_weights_are_empty_by_default_and_validate_when_set():
    """Empty is the training default; a set-but-wrong path must fail at startup."""
    cfg = load_config(CONFIG)
    assert cfg.weights == ""
    cfg.validate()  # empty is not a missing file

    cfg.weights = "/nope/model_0002000.pth"
    with pytest.raises(FileNotFoundError, match="weights"):
        cfg.validate()
