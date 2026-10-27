"""Step-3 verification: IoU-based GT class assignment (spec_cropcls.md)."""

import torch

from cropcls.assign import (
    ASSIGNED,
    CONTAMINATED,
    LOW_IOU,
    NO_GT,
    assign_gt_classes,
)

H = W = 64


def _mask(y0, y1, x0, x1):
    m = torch.zeros((H, W), dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def test_clean_single_gt_mask_is_assigned():
    gt = torch.stack([_mask(10, 30, 10, 30)])
    pred = torch.stack([_mask(10, 30, 10, 30)])
    a = assign_gt_classes(pred, gt, torch.tensor([5]))
    assert a.verdict == [ASSIGNED]
    assert a.labels.tolist() == [5]
    assert a.best_iou[0].item() == 1.0
    assert a.runner_up_iou[0].item() == 0.0


def test_mask_below_the_floor_is_dropped():
    gt = torch.stack([_mask(0, 40, 0, 40)])          # 1600 px
    pred = torch.stack([_mask(0, 14, 0, 40)])        # 560 px, IoU = 0.35
    a = assign_gt_classes(pred, gt, torch.tensor([3]))
    assert a.verdict == [LOW_IOU]
    assert a.labels.tolist() == [-1]
    assert a.n_kept == 0


def test_mask_straddling_two_gts_is_dropped_as_contaminated():
    # two adjacent instruments; one prediction covers most of both
    gt = torch.stack([_mask(0, 20, 0, 20), _mask(0, 20, 20, 40)])
    pred = torch.stack([_mask(0, 20, 0, 34)])
    a = assign_gt_classes(pred, gt, torch.tensor([1, 2]))
    assert a.best_iou[0] >= 0.5
    assert a.runner_up_iou[0] >= 0.2
    assert a.verdict == [CONTAMINATED]
    assert a.labels.tolist() == [-1]


def test_slight_bleed_onto_a_neighbour_is_still_kept():
    gt = torch.stack([_mask(0, 20, 0, 20), _mask(0, 20, 40, 60)])
    pred = torch.stack([_mask(0, 20, 0, 22)])   # covers gt0 fully, misses gt1 entirely
    a = assign_gt_classes(pred, gt, torch.tensor([7, 2]))
    assert a.verdict == [ASSIGNED]
    assert a.labels.tolist() == [7]


def test_zero_gt_frame_returns_empty_instead_of_throwing():
    # Such frames exist in this dataset (e.g. 012676FD8_01-008.hdf5, num_gt=0).
    pred = torch.stack([_mask(0, 10, 0, 10), _mask(20, 30, 20, 30)])
    gt = torch.zeros((0, H, W), dtype=torch.bool)
    a = assign_gt_classes(pred, gt, torch.zeros((0,), dtype=torch.long))
    assert len(a) == 2
    assert a.n_kept == 0
    assert a.verdict == [NO_GT, NO_GT]


def test_zero_predictions_returns_empty():
    gt = torch.stack([_mask(0, 10, 0, 10)])
    a = assign_gt_classes(
        torch.zeros((0, H, W), dtype=torch.bool), gt, torch.tensor([0])
    )
    assert len(a) == 0
    assert a.n_kept == 0


def test_single_gt_present_gives_zero_runner_up_not_an_index_error():
    gt = torch.stack([_mask(10, 30, 10, 30)])
    pred = torch.stack([_mask(10, 30, 10, 30)])
    a = assign_gt_classes(pred, gt, torch.tensor([4]))
    assert a.runner_up_iou.tolist() == [0.0]
    assert a.verdict == [ASSIGNED]


def test_predictions_at_lower_resolution_are_upsampled_not_gt_downsampled():
    gt = torch.stack([_mask(0, 32, 0, 32)])
    small = torch.zeros((1, H // 4, W // 4), dtype=torch.bool)
    small[0, 0:8, 0:8] = True            # same region at stride 4
    a = assign_gt_classes(small, gt, torch.tensor([6]))
    assert a.verdict == [ASSIGNED]
    assert a.labels.tolist() == [6]
    assert a.best_iou[0].item() > 0.9


def test_counts_reports_each_verdict():
    gt = torch.stack([_mask(0, 20, 0, 20), _mask(0, 20, 20, 40)])
    pred = torch.stack(
        [
            _mask(0, 20, 0, 20),       # assigned
            _mask(0, 20, 0, 34),       # contaminated
            _mask(60, 64, 60, 64),     # low iou
        ]
    )
    a = assign_gt_classes(pred, gt, torch.tensor([1, 2]))
    c = a.counts()
    assert c[ASSIGNED] == 1
    assert c[CONTAMINATED] == 1
    assert c[LOW_IOU] == 1
