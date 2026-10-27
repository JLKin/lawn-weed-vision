"""Offline, reproducible signed INT8 Q/DQ calibration; no TensorRT dependency."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .preprocessing import PREPROCESS_VERSION, prepare_image
from .yolov5_runtime import sha256_file


def image_manifest(directory: Path, limit: int) -> list[dict]:
    if not directory.is_dir() or not 1 <= limit <= 10000:
        raise ValueError("Provide an image directory and a sample limit from 1 to 10000")
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:limit]
    if not paths:
        raise ValueError("Calibration directory has no supported images; random calibration is not allowed")
    return [{"path": p.relative_to(directory).as_posix(), "sha256": sha256_file(p)} for p in paths]


class ImageReader:
    def __init__(self, directory, manifest, size):
        self.directory, self.manifest, self.size = directory, manifest, size
        self.rewind()

    def rewind(self):
        self.iterator = iter(self.manifest)

    def get_next(self):
        item = next(self.iterator, None)
        if item is None:
            return None
        path = self.directory / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Calibration data changed after manifest creation")
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1 or image.width * image.height > 12_000_000:
                raise ValueError("Calibration requires single-frame images of at most 12 million pixels")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            try:
                return {"images": prepare_image(rgb, self.size)}
            finally:
                rgb.close()


def verify_onnx(path: Path, info: dict):
    import onnx
    from .yolov5_runtime import normalized_names
    if info.get("backend") != "yolov5-seg" or info.get("batch_size") != 1:
        raise ValueError("Expected batch-1 YOLOv5 segmentation metadata")
    if info.get("onnx_sha256") != sha256_file(path) or info.get("preprocessing") != PREPROCESS_VERSION:
        raise ValueError("ONNX checksum or preprocessing version mismatch")
    normalized_names(info.get("names"))
    size = info.get("image_size")
    if not isinstance(size, int) or not 32 <= size <= 2048 or size % 32:
        raise ValueError("Invalid model image size")
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or model.graph.input[0].name != "images":
        raise ValueError("Expected only one images input")
    tensor = model.graph.input[0].type.tensor_type
    if tensor.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in tensor.shape.dim] != [1, 3, size, size]:
        raise ValueError("Expected static FP32 NCHW input matching metadata")
    if [x.name for x in model.graph.output] != ["output0", "output1"]:
        raise ValueError("Expected original YOLOv5 prediction/prototype output names")
    return model


def quantize_onnx(source: Path, output: Path, directory: Path, limit=300, excluded_prefixes=()):
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    if output.suffix != ".onnx" or output.exists() or output.with_suffix(".onnx.json").exists():
        raise ValueError("Choose an unused .onnx output path")
    info = json.loads(source.with_suffix(".onnx.json").read_text(encoding="utf-8"))
    model = verify_onnx(source, info)
    if any(n.op_type in {"QuantizeLinear", "DequantizeLinear"} for n in model.graph.node):
        raise ValueError("Input must be the FP32 source graph, not an already quantized graph")
    # Leave the final segmentation head in floating point by default.
    prefix = info.get("sensitive_node_prefix")
    if not prefix or not any(n.name.startswith(prefix) for n in model.graph.node):
        raise ValueError("Cannot identify segmentation head; export with the project ONNX script")
    prefixes = [prefix, *excluded_prefixes]
    excluded = [n.name for n in model.graph.node if any(n.name.startswith(p) for p in prefixes)]
    eligible = [n for n in model.graph.node if n.op_type == "Conv" and n.name not in excluded]
    if not eligible:
        raise ValueError("No convolution nodes remain eligible for quantization")
    manifest = image_manifest(directory, limit)
    reader = ImageReader(directory, manifest, info["image_size"])
    # Decode every image before spending time on calibration.
    while reader.get_next() is not None:
        pass
    reader.rewind()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".building.onnx")
    if temporary.exists():
        raise ValueError("Staging file already exists; choose another output path")
    quantize_static(str(source), str(temporary), reader, quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
        calibrate_method=CalibrationMethod.MinMax, op_types_to_quantize=["Conv"], nodes_to_exclude=excluded,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False})
    graph = onnx.load(str(temporary))
    onnx.checker.check_model(graph)
    count = sum(n.op_type == "QuantizeLinear" for n in graph.graph.node)
    if count == 0:
        raise ValueError("Quantization produced no Q/DQ nodes")
    zeros = [onnx.numpy_helper.to_array(t) for t in graph.graph.initializer if t.name.endswith("zero_point")]
    if not zeros or any(np.any(z != 0) or z.dtype != np.int8 for z in zeros):
        raise ValueError("TensorRT requires signed symmetric INT8 Q/DQ in this workflow")
    temporary.replace(output)
    info.update(precision_request="int8", quantization="explicit_qdq_s8s8_conv_minmax",
                parent_onnx_sha256=info["onnx_sha256"], onnx_sha256=sha256_file(output),
                calibration_images=manifest, calibration_image_count=len(manifest),
                calibration_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                excluded_nodes=excluded, quantize_node_count=count)
    import onnxruntime
    info["quantization_tools"] = {"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__}
    output.with_suffix(".onnx.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info
