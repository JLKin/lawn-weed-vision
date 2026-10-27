import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from weed_vision.preprocessing import PREPROCESS_VERSION, prepare_image
from weed_vision.quantization import ImageReader, image_manifest, quantize_onnx
from weed_vision.yolov5_runtime import sha256_file


def test_shared_preprocess_rgb_padding_and_shape():
    array = prepare_image(Image.new("RGB", (64, 32), color=(255, 0, 0)), 64)
    assert array.shape == (1, 3, 64, 64) and array.dtype == np.float32
    assert np.allclose(array[0, :, 0, 0], 114 / 255)
    assert np.array_equal(array[0, :, 32, 32], [1, 0, 0])


def test_calibration_refuses_empty_and_changed_images(tmp_path):
    with pytest.raises(ValueError, match="no supported"): image_manifest(tmp_path, 5)
    path = tmp_path / "a.png"
    Image.new("RGB", (16, 16)).save(path)
    manifest = image_manifest(tmp_path, 1)
    Image.new("RGB", (16, 16), "red").save(path)
    with pytest.raises(ValueError, match="changed"): ImageReader(tmp_path, manifest, 32).get_next()


def test_real_qdq_calibration_produces_executable_signed_graph(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from onnx import helper, TensorProto, numpy_helper
    source = tmp_path / "source.onnx"
    weights = numpy_helper.from_array(np.ones((3, 3, 1, 1), dtype=np.float32) * .1, "weight")
    graph = helper.make_graph([
        helper.make_node("Conv", ["images", "weight"], ["output0"], name="/model.0/conv"),
        helper.make_node("Identity", ["output0"], ["output1"], name="/model.24/head")],
        "test_only", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 3, 32, 32]) for n in ("output0", "output1")], [weights])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, str(source))
    source.with_suffix(".onnx.json").write_text(json.dumps(dict(backend="yolov5-seg", batch_size=1,
        names=["test"], image_size=32, preprocessing=PREPROCESS_VERSION, onnx_sha256=sha256_file(source), sensitive_node_prefix="/model.24/")))
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 20), (30, 100, 190)).save(images / "sample.png")
    output = tmp_path / "int8.onnx"
    info = quantize_onnx(source, output, images)
    assert info["quantize_node_count"] > 0 and info["calibration_image_count"] == 1
    result = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, {"images": np.ones((1, 3, 32, 32), dtype=np.float32) * .3})
    assert len(result) == 2 and np.isfinite(result[0]).all()
    with pytest.raises(ValueError, match="unused"): quantize_onnx(source, output, images)
