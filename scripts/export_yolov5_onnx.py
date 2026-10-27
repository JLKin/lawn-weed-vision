"""Export static FP32 YOLOv5 segmentation ONNX on Windows or Jetson."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from weed_vision.config import Settings
from weed_vision.preprocessing import PREPROCESS_VERSION
from weed_vision.yolov5_runtime import YOLOV5_REVISION, sha256_file
from weed_vision.yolov5_segmenter import Yolov5Segmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("models/yolov5n-seg.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--repo", type=Path, default=Path("vendor/yolov5"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != ".onnx" or output.exists() or output.with_suffix(".onnx.json").exists():
        parser.error("Choose a new .onnx output path; existing files are never overwritten")
    weights = args.weights.resolve()
    settings = Settings(_env_file=None, backend="yolov5", model_path=str(weights),
                        yolov5_repo=str(args.repo), device="cpu", half=False, image_size=args.image_size)
    model = Yolov5Segmenter(settings)
    model.load()
    names = [model.names[i] for i in range(len(model.names))]
    head_index = len(model._model.model.model) - 1
    del model
    from export import run
    import onnx

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="weed-onnx-") as temporary:
        staged = Path(temporary) / weights.name
        shutil.copyfile(weights, staged)
        results = run(weights=str(staged), imgsz=[args.image_size] * 2, batch_size=1,
                      device="cpu", half=False, dynamic=False, simplify=False, opset=13, include=("onnx",))
        exported = staged.with_suffix(".onnx")
        if not results or not exported.is_file():
            raise RuntimeError("YOLOv5 ONNX export failed")
        graph = onnx.load(str(exported))
        onnx.checker.check_model(graph)
        if len(graph.graph.output) != 2:
            raise ValueError("Expected prediction and prototype outputs")
        shutil.copyfile(exported, output)
    info = dict(backend="yolov5-seg", image_size=args.image_size, batch_size=1, names=names,
                source_sha256=sha256_file(weights), onnx_sha256=sha256_file(output),
                yolov5_revision=YOLOV5_REVISION, preprocessing=PREPROCESS_VERSION,
                precision_request="fp32", sensitive_node_prefix=f"/model.{head_index}/")
    output.with_suffix(".onnx.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"ONNX and metadata written: {output}")


if __name__ == "__main__":
    main()
