import numpy as np
import pytest

from scripts.benchmark_models import compare_masks, percentiles


def test_mask_comparison_respects_classes_and_missing_targets():
    first = np.eye(4, dtype=bool)
    second = np.fliplr(first)
    report = compare_masks(['weed', 'grass'], [first, second], ['grass', 'weed'], [second, first])
    assert report['matched_iou_ge_0_5'] == 2 and report['mean_matched_mask_iou'] == 1
    report = compare_masks(['weed'], [first], ['grass'], [first])
    assert report['unmatched_reference'] == 1 and report['mean_matched_mask_iou'] is None
    assert compare_masks([], [], [], [])['matched_iou_ge_0_5'] == 0


def test_latency_percentiles_are_milliseconds_not_throughput():
    result = percentiles([10, 20, 30])
    assert result == {'p50_ms': 20.0, 'p95_ms': pytest.approx(29), 'samples': 3}
