"""Static crop cache: provenance refusal, load-time filters, equivalence to the live path.

Every assertion here is the negative branch of something that would otherwise train happily
on the wrong pixels (spec_static_crops.md section 4). No detector and no GPU: the claim
"a cached crop equals the live crop" is about cropping a stored patch versus cropping the
whole frame, which is synthesizable.
"""

from __future__ import annotations

import json

import pytest
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog

from cropcls.cache import (
    MANIFEST,
    RECORDS,
    CachedCropDataset,
    CacheMismatch,
    _patch_bbox,
    _save_patch,
    build_loader,
)
from cropcls.config import CropClsConfig
from cropcls.crops import letterbox_crop, mask_tight_bbox, normalize_crop

CLASSES = ["forcep01", "scalpel01", "hammer01"]
FRAME = (3, 256, 256)


def _register(name: str, classes=CLASSES) -> str:
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
    DatasetCatalog.register(name, list)
    MetadataCatalog.get(name).thing_classes = list(classes)
    return name


def _instance(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A frame and one blob mask, well inside the frame so margin is never clipped."""
    g = torch.Generator().manual_seed(seed)
    frame = torch.randint(0, 256, FRAME, dtype=torch.uint8, generator=g)
    mask = torch.zeros(FRAME[1:], dtype=torch.bool)
    x, y = 90 + seed * 3, 100 + seed * 2
    mask[y : y + 34, x : x + 58] = True
    return frame, mask


def _write_cache(
    tmp_path, n=3, margin=32, iou_floor=0.5, runner_up_ceiling=0.2,
    ckpt="/fake/det.pth", classes=CLASSES, dropped=1,
):
    """Build a minimal valid cache by hand. Returns (root, [(frame, mask, label)])."""
    root = tmp_path / "cache"
    (root / "patches").mkdir(parents=True)
    truth, rows = [], []
    for i in range(n):
        frame, mask = _instance(i)
        label = i % len(classes)
        box = _patch_bbox(mask, margin)
        x0, y0, x1, y1 = box
        rel = f"patches/f{i:03d}_p000.png"
        _save_patch(root / rel, frame[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
        rows.append({
            "frame_index": i, "file_name": f"f{i:03d}.hdf5", "pred_index": 0,
            "label": label, "class_name": classes[label], "best_iou": 0.9,
            "runner_up_iou": 0.05, "verdict": "assigned", "score": 0.8, "patch": rel,
        })
        truth.append((frame, mask, label))
    for j in range(dropped):          # metadata row, no patch
        rows.append({
            "frame_index": 900 + j, "file_name": f"d{j:03d}.hdf5", "pred_index": 0,
            "label": -1, "class_name": "", "best_iou": 0.55,
            "runner_up_iou": 0.38, "verdict": "dropped_contaminated", "score": 0.7,
            "patch": None,
        })
    (root / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / MANIFEST).write_text(json.dumps({
        "dataset": "cachetest", "pool_realpath": ["/fake/pool"],
        "frame_digest": "deadbeef", "n_frames": n, "thing_classes": list(classes),
        "detector_checkpoint": ckpt, "detector_config": "/fake/config.yaml",
        "detector_score_thresh": 0.5, "margin": margin, "iou_floor": iou_floor,
        "runner_up_ceiling": runner_up_ceiling, "n_frames_extracted": n,
        "n_patches": n, "n_rows": len(rows), "verdict_counts": {},
    }, indent=2))
    return root, truth


def _cfg(root, **kw) -> CropClsConfig:
    base = {
        "detector_checkpoint": "/fake/det.pth", "detector_config": "/fake/config.yaml",
        "crop_cache": str(root), "margin": 32, "crop_size": 64, "photometric": False,
        "rotations": [0], "crops_per_batch": 2,
    }
    base.update(kw)
    return CropClsConfig(**base)


# --- equivalence to the live path ----------------------------------------------


def test_cached_crop_is_bit_identical_to_live_crop(tmp_path):
    """The headline comparison rests on this: a cache crop must equal a frame crop.

    If the stored patch carried too little context, or the load path cropped to a different
    box than data.py does, training would proceed on subtly different pixels and only the
    final f1 would disagree - with no way to tell why.
    """
    name = _register("cachetest")
    root, truth = _write_cache(tmp_path)
    cfg = _cfg(root)
    ds = CachedCropDataset(cfg, name, is_train=False)
    assert len(ds) == len(truth)

    for i in range(len(ds)):
        got, label = ds[i]
        frame, mask, want_label = truth[i]
        # box= the UNDILATED tight box, because that is what data.py:218-225 passes.
        # Using box=None here would take letterbox_crop's *dilated* box instead - which is
        # what crops.py:104-105 documents, and the two disagree. The live path wins.
        live = letterbox_crop(frame.float(), mask, cfg.crop_size, box=mask_tight_bbox(mask))
        want = normalize_crop(live[0], cfg.pixel_mean, cfg.pixel_std)
        assert label == want_label
        assert torch.equal(got, want), f"row {i} differs from the live crop"


@pytest.mark.parametrize("dilate", [0, 4, 16])
def test_cached_crop_matches_live_crop_under_dilation(tmp_path, dilate):
    """margin must cover the dilation, or the patch edge truncates the grown mask."""
    name = _register("cachetest")
    root, truth = _write_cache(tmp_path, margin=32)
    cfg = _cfg(root, mask_dilate=dilate)
    ds = CachedCropDataset(cfg, name, is_train=False)
    for i in range(len(ds)):
        got, _ = ds[i]
        frame, mask, _ = truth[i]
        live = letterbox_crop(
            frame.float(), mask, cfg.crop_size, box=mask_tight_bbox(mask), dilate=dilate
        )
        want = normalize_crop(live[0], cfg.pixel_mean, cfg.pixel_std)
        assert torch.equal(got, want), f"dilate={dilate} row {i} differs"


# --- the cache cannot be silently wrong ----------------------------------------


def test_mismatched_detector_checkpoint_raises_not_warns(tmp_path):
    """Masks from another detector are different data; training would not notice."""
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, ckpt="/fake/det_A.pth")
    with pytest.raises(CacheMismatch, match="different detector"):
        CachedCropDataset(_cfg(root, detector_checkpoint="/fake/det_B.pth"), name)


def test_reordered_thing_classes_raises_not_remaps(tmp_path):
    """A permutation of the same names has the same length - length alone cannot catch it."""
    root, _ = _write_cache(tmp_path, classes=CLASSES)
    name = _register("cachetest", classes=[CLASSES[1], CLASSES[0], CLASSES[2]])
    with pytest.raises(CacheMismatch, match="label space"):
        CachedCropDataset(_cfg(root), name)


def test_mask_dilate_above_margin_raises_not_clips(tmp_path):
    """mask_dilate is the largest measured effect; silent clipping would destroy the sweep."""
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, margin=8)
    with pytest.raises(CacheMismatch, match="margin"):
        CachedCropDataset(_cfg(root, mask_dilate=16), name)


def test_margin_check_reads_the_manifest_not_the_config(tmp_path):
    """A config claiming a wide margin must not license dilation the patches cannot support."""
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, margin=4)
    with pytest.raises(CacheMismatch, match="margin=4"):
        CachedCropDataset(_cfg(root, margin=64, mask_dilate=16), name)


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(CacheMismatch, match="no cache manifest"):
        CachedCropDataset(_cfg(tmp_path / "empty"), _register("cachetest"))


def test_empty_mask_in_cache_is_rejected_not_silently_served(tmp_path):
    """An all-zero alpha channel would letterbox to an all-black crop and train as a class."""
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, n=1, dropped=0)
    rel = json.loads((root / RECORDS).read_text().splitlines()[0])["patch"]
    _save_patch(root / rel, torch.zeros((3, 40, 40), dtype=torch.uint8),
                torch.zeros((40, 40), dtype=torch.bool))
    ds = CachedCropDataset(_cfg(root), name)
    with pytest.raises(CacheMismatch, match="empty mask"):
        ds[0]


def test_row_count_matches_patches_on_disk(tmp_path):
    """Manifest arithmetic must agree with the filesystem, independently counted."""
    _register("cachetest")
    root, _ = _write_cache(tmp_path, n=4, dropped=2)
    man = json.loads((root / MANIFEST).read_text())
    rows = [json.loads(x) for x in (root / RECORDS).read_text().splitlines() if x.strip()]
    on_disk = list((root / "patches").glob("*.png"))
    assert man["n_rows"] == len(rows) == 6
    assert man["n_patches"] == len(on_disk) == 4
    assert sum(1 for r in rows if r["patch"] is None) == 2


# --- load-time filters ----------------------------------------------------------


def test_dropped_detections_have_a_row_but_no_patch(tmp_path):
    """spec_cropcls.md:54 drops contaminated samples; the row stays for the rate monitor."""
    name = _register("cachetest")
    root, truth = _write_cache(tmp_path, n=3, dropped=2)
    ds = CachedCropDataset(_cfg(root), name)
    assert len(ds) == len(truth)        # the two dropped rows are not trainable
    rows = [json.loads(x) for x in (root / RECORDS).read_text().splitlines() if x.strip()]
    assert sum(1 for r in rows if r["verdict"] == "dropped_contaminated") == 2


def test_tightening_a_threshold_shrinks_the_dataset_without_re_extraction(tmp_path):
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, n=3, iou_floor=0.5)
    wide = CachedCropDataset(_cfg(root, iou_floor=0.5), name)
    tight = CachedCropDataset(_cfg(root, iou_floor=0.95), name)
    assert len(tight) < len(wide)


def test_loosening_a_threshold_raises_not_silently_short(tmp_path):
    """The patches for looser thresholds were never written; returning the old set is a lie."""
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, iou_floor=0.5, runner_up_ceiling=0.2)
    with pytest.raises(CacheMismatch, match="cannot loosen"):
        CachedCropDataset(_cfg(root, iou_floor=0.3), name)
    with pytest.raises(CacheMismatch, match="cannot loosen"):
        CachedCropDataset(_cfg(root, runner_up_ceiling=0.4), name)


def test_label_is_derived_from_the_name_not_read_as_an_int(tmp_path):
    """Stored ints would survive a relabelling; names cannot."""
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, n=3)
    lines = (root / RECORDS).read_text().splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    for r in rows:
        if r["patch"]:
            r["label"] = 999            # poison the stored int
    (root / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = CachedCropDataset(_cfg(root), name)
    for i in range(len(ds)):
        _, label = ds[i]
        assert label == CLASSES.index(ds.rows[i]["class_name"])


# --- the cache is used, not bypassed -------------------------------------------


def test_loader_runs_no_detector_forward(tmp_path, monkeypatch):
    """The whole point of the change. If the forward still runs, every other test passes."""
    import cropcls.detector as det_mod

    def explode(*a, **k):
        raise AssertionError("the detector was called while training from the cache")

    monkeypatch.setattr(det_mod.FrozenSingleClassDetector, "detect", explode)
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, n=4)
    loader = build_loader(_cfg(root, crops_per_batch=2), name, is_train=True, seed=0)
    seen = sum(x.shape[0] for x, _ in loader)
    assert seen > 0
