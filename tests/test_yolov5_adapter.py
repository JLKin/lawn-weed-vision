"""Backend-format guards prevent detection models or mislabeled engines producing wrong masks."""

import json

import numpy as np
import pytest

from weed_vision.config import Settings
from weed_vision.segmenter import create_segmenter
from weed_vision.yolov5_runtime import normalized_names, read_engine_metadata, sha256_file
from weed_vision.yolov5_segmenter import Yolov5Segmenter, split_segmentation_outputs


def test_default_backend_is_original_yolov5_segmentation():
    settings = Settings(_env_file=None)
    assert isinstance(create_segmenter(settings), Yolov5Segmenter)
    assert settings.model_path == "models/yolov5n-seg.pt"


def test_segmentation_channels_are_read_from_prototypes():
    # A custom 2-class, 16-mask-channel model is valid; do not hard-code COCO/32.
    pred = np.zeros((1, 20, 5 + 2 + 16))
    proto = np.zeros((1, 16, 8, 8))
    actual, masks, count = split_segmentation_outputs([pred, proto], 2)
    assert actual is pred and masks is proto and count == 16


@pytest.mark.parametrize("outputs,class_count", [
    (np.zeros((1, 20, 85)), 80),  # Detection-only model.
    ([np.zeros((1, 20, 85)), np.zeros((1, 32, 8, 8))], 80),  # Missing mask coefficients.
    ([np.zeros((1, 20, 117)), np.zeros((1, 32, 8, 8))], 2),  # Incorrect engine labels.
    ([np.zeros((2, 20, 117)), np.zeros((2, 32, 8, 8))], 80),  # Wrong batch size.
    ([np.zeros((1, 20, 117)), np.zeros((1, 32, 8))], 80),  # Invalid prototype rank.
])
def test_incompatible_model_outputs_are_rejected(outputs, class_count):
    with pytest.raises(ValueError):
        split_segmentation_outputs(outputs, class_count)


def test_engine_requires_matching_size_hash_and_real_class_names(tmp_path):
    engine = tmp_path / "test.engine"
    engine.write_bytes(b"not-a-real-engine; metadata validation needs no GPU")
    sidecar = engine.with_suffix(".engine.json")
    info = {"backend": "yolov5-seg", "batch_size": 1, "image_size": 640,
            "engine_sha256": sha256_file(engine), "names": ["turfgrass", "weed"]}
    sidecar.write_text(json.dumps(info), encoding="utf-8")
    assert read_engine_metadata(engine, 640)["names"] == {0: "turfgrass", 1: "weed"}
    with pytest.raises(ValueError, match="IMAGE_SIZE"):
        read_engine_metadata(engine, 320)
    engine.write_bytes(b"different artifact")
    with pytest.raises(ValueError, match="checksum"):
        read_engine_metadata(engine, 640)


def test_engine_missing_metadata_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_engine_metadata(tmp_path / "test.engine", 640)


@pytest.mark.parametrize("names", [[], {1: "weed"}, {0: ""}, None])
def test_invalid_class_id_tables_are_rejected(names):
    with pytest.raises(ValueError):
        normalized_names(names)
