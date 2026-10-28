"""Static batching: constant batch size, global shuffle, rotation varied per draw.

This is the half of the change that the live pipeline cannot do. There, a batch is
`ims_per_batch` FRAMES and every surviving instance contributes all four rotations to that
same batch - and rotations of a square crop are lossless pixel permutations, so they carry
identical per-channel statistics. BatchNorm therefore sees an effective sample count of
batch/4. These tests pin the three properties that fix it (spec_static_crops.md section 4).
"""

from __future__ import annotations

from cropcls.cache import CachedCropDataset, build_loader
from tests.test_cropcls_cache import _cfg, _register, _write_cache


def test_every_batch_has_the_configured_crop_count(tmp_path):
    """Constant size is what makes BatchNorm statistics comparable between steps.

    The live path's batch size is data-dependent - it varies with how many instances each
    frame happens to contain - so no two steps normalize over the same n.
    """
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, n=9, dropped=0)
    loader = build_loader(_cfg(root, crops_per_batch=4), name, is_train=True, seed=0)
    sizes = [x.shape[0] for x, _ in loader]
    assert sizes == [4, 4], f"expected two full batches of 4 from 9 crops, got {sizes}"


def test_successive_passes_draw_in_a_different_order(tmp_path):
    """Global shuffle, asserted on the ORDER - not the multiset.

    A full `shuffle=True` pass yields every crop exactly once, so the multiset is identical
    by construction and a multiset assertion would fail on a correct implementation. Only the
    sequence can distinguish a shuffled sampler from a sequential one.
    """
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, n=24, dropped=0)
    cfg = _cfg(root, crops_per_batch=4)

    def one_pass(seed):
        return [tuple(y.tolist()) for _, y in build_loader(cfg, name, True, seed=seed)]

    a, b = one_pass(0), one_pass(1)
    assert sorted(x for t in a for x in t) == sorted(x for t in b for x in t), (
        "both passes must cover the same crops"
    )
    assert a != b, "two passes drew in the same order; the sampler is not shuffling"


def test_one_instance_receives_more_than_one_rotation_across_draws(tmp_path):
    """Rotation is sampled per draw, so an instance is not pinned to one orientation.

    The earlier formulation - "no instance contributes more than one rotation to any single
    batch" - could not fail: with one rotation per crop per pass it holds for every sampler,
    including one that pins a rotation forever.
    """
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, n=1, dropped=0)
    ds = CachedCropDataset(_cfg(root, rotations=[0, 90, 180, 270]), name, is_train=True)
    seen = {tuple(ds[0][0].flatten()[:24].tolist()) for _ in range(40)}
    assert len(seen) >= 2, "40 draws of one crop produced a single orientation"


def test_eval_loading_is_deterministic_and_unrotated(tmp_path):
    """Augmentation must not leak into evaluation, or val numbers wander between passes."""
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, n=3, dropped=0)
    ds = CachedCropDataset(_cfg(root, rotations=[0, 90, 180, 270]), name, is_train=False)
    assert ds.rotations == [0]
    assert ds.photometric is None
    first = [ds[i][0] for i in range(len(ds))]
    second = [ds[i][0] for i in range(len(ds))]
    assert all((a == b).all() for a, b in zip(first, second))


def test_eval_loader_does_not_drop_the_short_final_batch(tmp_path):
    """drop_last on val would silently score fewer crops than the dataset holds."""
    name = _register("cachetest")
    root, _ = _write_cache(tmp_path, n=5, dropped=0)
    loader = build_loader(_cfg(root, crops_per_batch=2), name, is_train=False)
    assert sum(x.shape[0] for x, _ in loader) == 5
