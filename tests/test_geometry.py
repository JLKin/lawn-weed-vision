"""Mask geometry tests use synthetic regions, including a hole and disconnected pixels."""

from types import SimpleNamespace

import numpy as np
import pytest

from weed_vision.segmenter import detections_from_result


class TensorStub:
    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class BoxesStub:
    def __init__(self, boxes):
        self.xyxy = TensorStub(boxes)
        self.conf = TensorStub([0.8] * len(boxes))
        self.cls = TensorStub([0] * len(boxes))

    def __len__(self):
        return len(self.xyxy.array)


def make_result(mask):
    height, width = mask.shape
    return SimpleNamespace(
        boxes=BoxesStub([[0, 0, width, height]]),
        masks=SimpleNamespace(
            data=[TensorStub(mask)],
            xy=[np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])],
        ),
        names={0: "model_native_class"},
    )


def test_area_and_centroid_use_foreground_pixels_not_box_or_contour():
    mask = np.zeros((7, 9), dtype=np.float32)
    mask[1:4, 1:4] = 1
    mask[2, 2] = 0  # A hole must not contribute to area or centroid.
    mask[5, 7] = 1  # A disconnected component must contribute.
    detection = detections_from_result(make_result(mask), (9, 7))[0]
    assert detection.area_px == 9
    assert detection.centroid_px == pytest.approx((23 / 9, 21 / 9))
    assert detection.centroid_px != (4.5, 3.5)
    assert detection.class_name == "model_native_class"
    assert detection.ground_position_m is None


def test_native_shape_is_required_to_avoid_scaled_coordinates():
    result = make_result(np.ones((3, 4)))
    with pytest.raises(RuntimeError, match="native image-sized"):
        detections_from_result(result, (8, 6))


def test_missing_masks_on_detected_instances_is_an_error():
    result = make_result(np.ones((3, 4)))
    result.masks = None
    with pytest.raises(RuntimeError, match="missing"):
        detections_from_result(result, (4, 3))


def test_no_instances_returns_empty_list():
    result = SimpleNamespace(boxes=BoxesStub([]), masks=None)
    assert detections_from_result(result, (9, 7)) == []


def test_empty_masks_are_omitted():
    assert detections_from_result(make_result(np.zeros((3, 4))), (4, 3)) == []
