"""Unit tests for maskdino/data/class_mapping.py.

The module is loaded by file path (like tests/test_summarize_reclass.py) so the
tests don't trigger ``import maskdino`` (which registers datasets from absolute
data paths). detectron2's MetadataCatalog is a hard dep and is used directly with
throwaway dataset names.
"""
import importlib.util
import json
import os
import sys
import types
import uuid

import pytest

_HERE = os.path.dirname(__file__)
_MOD_PATH = os.path.join(_HERE, "..", "maskdino", "data", "class_mapping.py")
_spec = importlib.util.spec_from_file_location("class_mapping", _MOD_PATH)
cm_mod = importlib.util.module_from_spec(_spec)
sys.modules["class_mapping"] = cm_mod
_spec.loader.exec_module(cm_mod)

derive_class_mapping = cm_mod.derive_class_mapping
ClassMapping = cm_mod.ClassMapping
remap_gt_category_ids = cm_mod.remap_gt_category_ids

# the phase-1 (set-B only) and reclass (set A + B) instrument_classes lists
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


def test_derive_drops_background_and_unused():
    cm = derive_class_mapping(SETB)
    assert cm.num_classes == 8
    assert cm.thing_classes == SETB[11:]
    assert cm.thing_dataset_id_to_contiguous_id == {c: c - 11 for c in range(11, 19)}
    assert cm.contiguous_to_canonical[0] == {"category_id": 11, "name": "adapter-11"}


def test_derive_setab_ordered_by_canonical_id():
    cm = derive_class_mapping(SETAB)
    assert cm.num_classes == 16
    assert cm.thing_dataset_id_to_contiguous_id == {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: 7,
        11: 8, 12: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15,
    }
    # the COCOEvaluator invariant: contiguous ids are exactly 0..N-1
    assert set(cm.thing_dataset_id_to_contiguous_id.values()) == set(range(16))


def test_reverse_lookups_and_keyerror():
    cm = derive_class_mapping(SETAB)
    assert cm.canonical_id(8) == 11
    assert cm.canonical_name(8) == "adapter-11"
    assert cm.contiguous_id(11) == 8
    with pytest.raises(KeyError):
        cm.contiguous_id(7)  # unused_7 has no slot
    with pytest.raises(KeyError):
        cm.canonical_id(16)  # out of range


def test_derive_rejects_all_placeholder():
    with pytest.raises(ValueError):
        derive_class_mapping(["background", "unused_1", "unused_2"])


def test_derive_warns_on_missing_background(caplog):
    cm = derive_class_mapping(["forcep01", "scalpel01"])  # index 0 not "background"
    # index 0 dropped regardless -> only "scalpel01" (canonical id 1) survives
    assert cm.thing_classes == ["scalpel01"]
    assert cm.thing_dataset_id_to_contiguous_id == {1: 0}


def test_state_dict_roundtrip_int_keys():
    cm = derive_class_mapping(SETAB)
    state = cm.state_dict()
    assert list(state["entries"].keys()) == [str(i) for i in range(16)]  # str keys out
    back = ClassMapping.from_state_dict(state)
    assert back == cm
    assert all(isinstance(k, int) for k in back.contiguous_to_canonical)  # int keys back
    assert back.thing_dataset_id_to_contiguous_id == cm.thing_dataset_id_to_contiguous_id


def test_json_roundtrip():
    cm = derive_class_mapping(SETB)
    back = ClassMapping.from_json_dict(json.loads(json.dumps(cm.to_json_dict())))
    assert back == cm


def test_load_state_dict_in_place():
    cm = derive_class_mapping(SETB)
    other = derive_class_mapping(SETAB)
    cm.load_state_dict(other.state_dict())
    assert cm == other
    assert cm.num_classes == 16


def test_remap_gt_category_ids_inplace_and_raises():
    cm = derive_class_mapping(SETAB)
    anns = [{"category_id": 11, "bbox": [0, 0, 1, 1]}, {"category_id": 1}]
    remap_gt_category_ids(anns, cm)
    assert [a["category_id"] for a in anns] == [8, 0]
    assert anns[0]["bbox"] == [0, 0, 1, 1]  # other keys untouched
    with pytest.raises(ValueError):
        remap_gt_category_ids([{"category_id": 7}], cm)  # unused_7 -> no slot


def test_from_metadata_with_and_without_id_map():
    md = types.SimpleNamespace(
        thing_classes=["adapter-11", "hammer-11"],
        thing_dataset_id_to_contiguous_id={11: 0, 14: 1},
    )
    cm = ClassMapping.from_metadata(md)
    assert cm.contiguous_to_canonical[1] == {"category_id": 14, "name": "hammer-11"}

    legacy = types.SimpleNamespace(thing_classes=["a", "b", "c"])
    cm2 = ClassMapping.from_metadata(legacy)
    assert cm2.thing_dataset_id_to_contiguous_id == {0: 0, 1: 1, 2: 2}


def _fake_cfg(num_classes):
    return types.SimpleNamespace(
        MODEL=types.SimpleNamespace(SEM_SEG_HEAD=types.SimpleNamespace(NUM_CLASSES=num_classes))
    )


@pytest.fixture
def registered_dataset():
    from detectron2.data import MetadataCatalog

    name = f"_test_cm_{uuid.uuid4().hex}"
    cm = derive_class_mapping(SETB)
    cm_mod.apply_class_mapping_to_metadata(name, cm)
    yield name, cm
    MetadataCatalog.remove(name)


def test_apply_class_mapping_to_metadata(registered_dataset):
    from detectron2.data import MetadataCatalog

    name, cm = registered_dataset
    md = MetadataCatalog.get(name)
    assert md.thing_classes == cm.thing_classes
    assert md.thing_dataset_id_to_contiguous_id == cm.thing_dataset_id_to_contiguous_id
    assert md.evaluator_type == "coco"
    assert ClassMapping.from_state_dict(md.class_mapping_entries) == cm


def test_set_num_classes_sentinel_fills_in(registered_dataset):
    name, cm = registered_dataset
    cfg = _fake_cfg(-1)
    cm_mod.set_num_classes_from_metadata(cfg, name)
    assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 8


def test_set_num_classes_match_is_noop(registered_dataset):
    name, _ = registered_dataset
    cfg = _fake_cfg(8)
    cm_mod.set_num_classes_from_metadata(cfg, name)
    assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 8


def test_set_num_classes_pinned_one_warns_not_override(registered_dataset):
    name, _ = registered_dataset
    cfg = _fake_cfg(1)
    cm_mod.set_num_classes_from_metadata(cfg, name)
    assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 1


def test_set_num_classes_mismatch_raises(registered_dataset):
    name, _ = registered_dataset
    cfg = _fake_cfg(19)
    with pytest.raises(ValueError):
        cm_mod.set_num_classes_from_metadata(cfg, name)


def test_set_num_classes_no_marker_is_noop():
    from detectron2.data import MetadataCatalog

    name = f"_test_nomarker_{uuid.uuid4().hex}"
    MetadataCatalog.get(name).set(thing_classes=["a", "b", "c"])  # no class_mapping_entries
    try:
        cfg = _fake_cfg(80)
        cm_mod.set_num_classes_from_metadata(cfg, name)
        assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 80
    finally:
        MetadataCatalog.remove(name)


def test_write_and_load_sidecar(tmp_path):
    cm = derive_class_mapping(SETAB)
    path = cm_mod.write_class_mapping_sidecar(str(tmp_path), cm)
    assert path.endswith("class_mapping.json")
    assert cm_mod.load_class_mapping(str(tmp_path)) == cm  # dir
    assert cm_mod.load_class_mapping(path) == cm  # file
