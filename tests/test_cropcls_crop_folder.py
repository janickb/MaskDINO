"""Crops on disk: a folder crop must equal the crop the live pipeline builds.

The fake mask here is deliberately FRAGMENTED - dashed bars plus a blob, like the predicted
masks `crops.py:52-54` describes. An earlier version of these tests used a solid rectangle,
which made every dilation assertion vacuous: a solid mask equals its own tight box, and with
`box=tight` the crop region never grows, so `mask_dilate=16` produced a crop bit-identical to
`mask_dilate=0`. The test passed with dilation deleted.
"""

from __future__ import annotations

import csv
import json

import pytest
import torch

from cropcls.assign import assign_gt_classes
from cropcls.config import CropClsConfig
from cropcls.crop_folder import (
    CropFolder,
    build_loader,
    load_crop,
    patch_bbox,
    save_crop,
    write_frame_crops,
)
from cropcls.crops import (
    letterbox_crop,
    mask_tight_bbox,
    normalize_crop,
    to_model_input,
)

# Deliberately NOT sorted: thing_classes puts scalpel before hammer, sorted() does the
# reverse. A fixture in sorted order would make the real label-order bug invisible.
CLASSES = ["forcep01_mm", "scalpel01_mm", "hammer01_mm"]
assert CLASSES != sorted(CLASSES), "the fixture must exercise the order mismatch"
H = W = 256


def _frame(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (3, H, W), dtype=torch.uint8, generator=g)


def _fragmented(seed: int) -> torch.Tensor:
    """A dashed shaft plus a jaw blob: holes that dilation closes and erosion reopens."""
    m = torch.zeros((H, W), dtype=torch.bool)
    y, x = 100 + (seed % 3), 60 + (seed % 5)
    for i in range(0, 110, 14):                     # dashed horizontal shaft
        m[y : y + 6, x + i : x + i + 9] = True
    m[y - 18 : y + 24, x + 46 : x + 52] = True      # crossbar
    m[y - 8 : y + 14, x + 104 : x + 126] = True     # jaws
    return m


def _cfg(**kw) -> CropClsConfig:
    base = {
        "detector_checkpoint": "/fake/det.pth", "detector_config": "/fake/cfg.yaml",
        "crop_size": 64, "photometric": False, "rotations": [0],
        "crops_per_batch": 2, "margin": 16, "num_workers": 0,
    }
    base.update(kw)
    return CropClsConfig(**base)


def _folder(tmp_path, n=5, margin=16):
    """Write n crops the way the extraction script does. Returns (root, truth)."""
    root = tmp_path / "crops"
    root.mkdir()
    truth = []
    for i in range(n):
        frame, mask = _frame(i), _fragmented(i)
        cls = CLASSES[i % len(CLASSES)]
        x0, y0, x1, y1 = patch_bbox(mask, margin)
        (root / cls).mkdir(exist_ok=True)
        save_crop(root / cls / f"f{i:03d}_p000.png", frame[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
        truth.append((f"f{i:03d}_p000.png", cls, frame, mask))
    (root / "meta.json").write_text(
        json.dumps({"margin": margin, "thing_classes": CLASSES})
    )
    return root, truth


def _live(frame, mask, cfg, photometric=None):
    """What `data.py` builds: letterbox_crop with box= the UNDILATED tight box."""
    out = letterbox_crop(
        frame.float(), mask, cfg.crop_size, box=mask_tight_bbox(mask),
        photometric=photometric, mask_only=cfg.mask_only, blackout=cfg.blackout,
        dilate=cfg.mask_dilate, close=cfg.mask_close,
    )
    return to_model_input(out, cfg.pixel_mean, cfg.pixel_std,
                          mask_channel=cfg.mask_channel)


# --- the folder crop equals the live crop -------------------------------------


@pytest.mark.parametrize("blackout", [True, False])
@pytest.mark.parametrize("mask_channel", [False, True])
@pytest.mark.parametrize(
    ("dilate", "close"), [(0, False), (8, False), (16, False), (8, True), (16, True)]
)
def test_folder_crop_equals_live_crop(tmp_path, dilate, close, mask_channel, blackout):
    """Over 5 frames, every mask setting. `close=True` is the case where margin matters.

    With `close=False` the stored border is provably unused - every mask pixel is inside its
    own tight box - so only the closing cases make the stored extent load-bearing.

    `blackout=False` and `mask_channel=True` are the settings spec_cropcls.md Step 10 ships,
    and the matrix reached neither before: `_cfg` inherits `blackout: True`, so folder-vs-live
    identity had never been exercised at the only blackout setting those arms use.
    """
    cfg = _cfg(mask_dilate=dilate, mask_close=close, margin=16,
               blackout=blackout, mask_channel=mask_channel)
    root, truth = _folder(tmp_path, n=5, margin=16)
    ds = CropFolder(root, cfg, is_train=False)
    assert len(ds) == 5
    by_name = {p.name: i for i, (p, _) in enumerate(ds.items)}
    for name, cls, frame, mask in truth:
        got, label = ds[by_name[name]]
        assert ds.class_names[label] == cls
        assert torch.equal(got, _live(frame, mask, cfg)), (
            f"{name} dilate={dilate} close={close} "
            f"mask_channel={mask_channel} blackout={blackout}"
        )


def test_dilating_changes_the_crop(tmp_path):
    """The assertion the old suite was missing, and the one that caught the fake fixture.

    If this passes with `mask_dilate` ignored, the fixture mask is solid again.
    """
    root, _ = _folder(tmp_path)
    a = CropFolder(root, _cfg(mask_dilate=0), is_train=False)[0][0]
    b = CropFolder(root, _cfg(mask_dilate=8), is_train=False)[0][0]
    assert not torch.equal(a, b), "dilation changed nothing; the fixture mask is not fragmented"


def test_closing_differs_from_plain_dilation(tmp_path):
    root, _ = _folder(tmp_path)
    d = CropFolder(root, _cfg(mask_dilate=8, mask_close=False), is_train=False)[0][0]
    c = CropFolder(root, _cfg(mask_dilate=8, mask_close=True), is_train=False)[0][0]
    assert not torch.equal(d, c)


def test_blackout_off_keeps_background(tmp_path):
    """blackout=False must actually change the crop, or the ablation measures nothing."""
    root, _ = _folder(tmp_path)
    on = CropFolder(root, _cfg(blackout=True), is_train=False)[0][0]
    off = CropFolder(root, _cfg(blackout=False), is_train=False)[0][0]
    assert not torch.equal(on, off)


# --- the 4th channel is what keeps photometric usable -------------------------


def test_photometric_runs_before_blackout(tmp_path):
    """The reason the mask is stored in alpha instead of baked into the RGB.

    Contrast blends toward the image mean; on a blacked crop that mean is near zero, so
    jitter applied after blackout behaves nothing like the frame-level op it replaces. Here
    it must change pixels inside the mask while leaving the background exactly 0.
    """
    cfg = _cfg(photometric=True)
    root, _ = _folder(tmp_path)
    ds = CropFolder(root, cfg, is_train=True, seed=3)
    torch.manual_seed(0)
    a, _ = ds[0]
    torch.manual_seed(1)
    b, _ = ds[0]
    assert not torch.equal(a, b), "photometric did not fire"

    plain = CropFolder(root, _cfg(photometric=False), is_train=False)[0][0]
    bg = plain == normalize_crop(
        torch.zeros(1), cfg.pixel_mean, cfg.pixel_std
    ).view(-1, 1, 1).expand_as(plain)
    assert torch.equal(a[bg], plain[bg]), "jitter leaked into the blacked-out background"


# --- what the extraction script writes ----------------------------------------


def test_write_frame_crops_writes_only_kept_detections(tmp_path):
    """A dropped detection gets a CSV row and no file - never silence."""
    frame = _frame(0)
    gt = torch.stack([_fragmented(0), _fragmented(1)])
    good = _fragmented(0)
    contaminated = gt[0] | gt[1]                      # straddles two GTs -> dropped
    masks = torch.stack([good, contaminated])
    asg = assign_gt_classes(masks, gt, torch.tensor([0, 1]))

    out = tmp_path / "crops"
    out.mkdir()
    rows = write_frame_crops(out, "f000", frame, masks, asg, CLASSES, margin=16)

    assert len(rows) == 2, "one row per detection, kept or not"
    written = [r for r in rows if r["crop"]]
    assert len(written) == sum(bool(x) for x in asg.keep)
    assert len(list(out.rglob("*.png"))) == len(written)
    for r in rows:
        if not r["crop"]:
            assert r["verdict"].startswith("dropped"), r


def test_written_folder_loads_back(tmp_path):
    """The round trip the script depends on: what it writes, CropFolder reads."""
    frame, mask = _frame(0), _fragmented(0)
    asg = assign_gt_classes(mask[None], mask[None], torch.tensor([1]))
    out = tmp_path / "crops"
    out.mkdir()
    rows = write_frame_crops(out, "f000", frame, mask[None], asg, CLASSES, margin=16)
    assert rows[0]["crop"] == f"{CLASSES[1]}/f000_p000.png"
    # the script writes meta.json alongside; CropFolder needs it for the label order
    (out / "meta.json").write_text(json.dumps({"margin": 16, "thing_classes": CLASSES}))

    ds = CropFolder(out, _cfg(), is_train=False)
    assert len(ds) == 1
    got, label = ds[0]
    assert ds.class_names[label] == CLASSES[1]
    assert torch.equal(got, _live(frame, mask, _cfg()))


def test_stored_crop_is_lossless(tmp_path):
    """PNG round-trip must be exact, or bit-identity to the live path is luck."""
    frame, mask = _frame(0), _fragmented(0)
    x0, y0, x1, y1 = patch_bbox(mask, 16)
    p = tmp_path / "c.png"
    save_crop(p, frame[:, y0:y1, x0:x1], mask[y0:y1, x0:x1])
    rgb, m = load_crop(p)
    assert torch.equal(rgb, frame[:, y0:y1, x0:x1])
    assert torch.equal(m, mask[y0:y1, x0:x1])


def test_float_frame_is_refused(tmp_path):
    """A float frame would quantize through PNG and silently break bit-identity."""
    with pytest.raises(TypeError, match="uint8"):
        save_crop(tmp_path / "c.png", torch.zeros((3, 8, 8)), torch.ones((8, 8), dtype=torch.bool))


# --- loading and batching -----------------------------------------------------


def test_class_order_comes_from_thing_classes_not_sorted_dirs(tmp_path):
    """The label ORDER must come from meta.json, never from sorted() directory names.

    This is the bug the first sanity run hit. thing_classes is
    [forcep01, scalpel01, hammer01, sharpspoon01, scarstick01, ...] while sorted() puts
    hammer before scalpel and scarstick before sharpspoon. The forceps coincide, so a sorted
    order trains those four correctly and silently mislabels the rest - measured as forceps
    0.27-0.82 and every other class 0.00-0.03.
    """
    root, truth = _folder(tmp_path, n=3)
    ds = CropFolder(root, _cfg(), is_train=False)
    assert ds.class_names == CLASSES, "order must be thing_classes, not sorted()"
    assert ds.class_names != sorted(CLASSES), "the fixture stopped exercising the mismatch"
    by_name = {p.name: i for i, (p, _) in enumerate(ds.items)}
    for name, cls, _frame, _mask in truth:
        _, label = ds[by_name[name]]
        assert ds.class_names[label] == cls, f"{name} labelled {ds.class_names[label]}, not {cls}"


def test_missing_meta_raises_rather_than_guessing_the_order(tmp_path):
    """Without meta.json the order is unknowable, and guessing it mislabels silently."""
    root, _ = _folder(tmp_path, n=3)
    (root / "meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="class ORDER is unknown"):
        CropFolder(root, _cfg(), is_train=False)


def test_a_folder_outside_the_label_space_raises(tmp_path):
    """Crops extracted against a different dataset must not load as if they belonged."""
    root, _ = _folder(tmp_path, n=3)
    (root / "retractor99_mm").mkdir()
    (root / "retractor99_mm" / "x.png").write_bytes((root / CLASSES[0]).glob("*.png").__next__().read_bytes())
    with pytest.raises(ValueError, match="not in the recorded label space"):
        CropFolder(root, _cfg(), is_train=False)


def test_every_batch_has_the_configured_crop_count(tmp_path):
    """Constant batch size is what makes BatchNorm statistics comparable between steps."""
    root, _ = _folder(tmp_path, n=9)
    sizes = [x.shape[0] for x, _ in build_loader(_cfg(crops_per_batch=4), root, True, seed=0)]
    assert sizes == [4, 4], sizes


def test_successive_passes_draw_in_a_different_order(tmp_path):
    """Asserted on the ORDER: a full shuffled pass yields every crop exactly once, so the
    multiset is identical by construction and a multiset assertion could not fail."""
    root, _ = _folder(tmp_path, n=12)
    cfg = _cfg(crops_per_batch=2)
    runs = [[tuple(y.tolist()) for _, y in build_loader(cfg, root, True, seed=s)] for s in (0, 1)]
    assert sorted(x for b in runs[0] for x in b) == sorted(x for b in runs[1] for x in b)
    assert runs[0] != runs[1], "two passes drew in the same order"


def test_one_crop_gets_more_than_one_rotation_across_draws(tmp_path):
    """Rotation is sampled per draw, so no instance is pinned to one orientation."""
    root, _ = _folder(tmp_path, n=1)
    ds = CropFolder(root, _cfg(rotations=[0, 90, 180, 270]), is_train=True, seed=5)
    seen = {tuple(ds[0][0].flatten()[:32].tolist()) for _ in range(40)}
    assert len(seen) >= 2


def test_eval_loading_is_unrotated_and_deterministic(tmp_path):
    """Augmentation must not leak into validation, or val numbers wander between passes."""
    root, _ = _folder(tmp_path, n=3)
    ds = CropFolder(root, _cfg(rotations=[0, 90, 180, 270], photometric=True), is_train=False)
    assert ds.rotations == [0] and ds.photometric is None
    assert all(torch.equal(ds[i][0], ds[i][0]) for i in range(len(ds)))


def test_eval_loader_keeps_the_short_final_batch(tmp_path):
    """drop_last on val would silently score fewer crops than the folder holds."""
    root, _ = _folder(tmp_path, n=5)
    loader = build_loader(_cfg(crops_per_batch=2), root, is_train=False)
    assert sum(x.shape[0] for x, _ in loader) == 5


def test_too_few_crops_for_one_batch_raises(tmp_path):
    """drop_last would yield nothing and the training loop would spin silently."""
    root, _ = _folder(tmp_path, n=2)
    with pytest.raises(ValueError, match="would yield nothing"):
        build_loader(_cfg(crops_per_batch=8), root, is_train=True)


def test_empty_folder_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no class folders"):
        CropFolder(tmp_path / "empty", _cfg())


def test_csv_rows_cover_every_detection(tmp_path):
    """The assignment rate is read off this file; a missing row would inflate it."""
    frame = _frame(0)
    gt = torch.stack([_fragmented(0), _fragmented(1)])
    masks = torch.stack([_fragmented(0), gt[0] | gt[1], torch.zeros((H, W), dtype=torch.bool)])
    asg = assign_gt_classes(masks, gt, torch.tensor([0, 1]))
    out = tmp_path / "crops"
    out.mkdir()
    rows = write_frame_crops(out, "f000", frame, masks, asg, CLASSES, margin=16)
    path = out / "crops.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with path.open() as fh:
        assert len(list(csv.DictReader(fh))) == len(asg) == 3


def test_closing_wider_than_the_stored_margin_raises(tmp_path):
    """The one silent-wrongness case left: closing needs margin >= dilate, measured exactly.

    Plain dilation is exact at margin=0, so only this combination can quietly stop matching
    the live path - the closing window gets truncated at the patch edge.
    """
    root, _ = _folder(tmp_path, margin=4)
    with pytest.raises(ValueError, match="margin >= 16"):
        CropFolder(root, _cfg(mask_dilate=16, mask_close=True), is_train=False)


def test_closing_within_the_stored_margin_is_allowed(tmp_path):
    root, _ = _folder(tmp_path, margin=16)
    ds = CropFolder(root, _cfg(mask_dilate=16, mask_close=True), is_train=False)
    assert len(ds) == 5


def test_plain_dilation_needs_no_margin_at_all(tmp_path):
    """margin=0 must still serve the whole dilation sweep, or margin=0 forecloses +4.2."""
    root, truth = _folder(tmp_path, margin=0)
    cfg = _cfg(mask_dilate=16, mask_close=False, margin=0)
    ds = CropFolder(root, cfg, is_train=False)
    by_name = {p.name: i for i, (p, _) in enumerate(ds.items)}
    for name, _cls, frame, mask in truth:
        assert torch.equal(ds[by_name[name]][0], _live(frame, mask, cfg)), name



