"""Build a fixed batch-1 FP16-requested YOLOv5 segmentation engine on the target Jetson."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from weed_vision.config import Settings
from weed_vision.yolov5_runtime import YOLOV5_REVISION, sha256_file
from weed_vision.yolov5_segmenter import Yolov5Segmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("models/yolov5n-seg.pt"))
    parser.add_argument("--repo", type=Path, default=Path("vendor/yolov5"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--workspace-gib", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing .engine")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64" or not Path("/etc/nv_tegra_release").exists():
        raise SystemExit("Run this export on the target Jetson; Windows engines are not portable to Jetson.")
    if not 1 <= args.workspace_gib <= 4:
        parser.error("--workspace-gib must be between 1 and 4")
    weights = args.weights.resolve()
    if weights.suffix != ".pt" or not weights.is_file():
        parser.error("--weights must name an existing trusted YOLOv5 segmentation .pt file")
    engine = weights.with_suffix(".engine")
    if engine.exists() and not args.overwrite:
        parser.error("Engine already exists; choose another model path or pass --overwrite")

    from scripts.check_jetson import inspect_environment

    environment, errors = inspect_environment()
    if errors:
        raise SystemExit("\n".join(errors))
    settings = Settings(
        _env_file=None, backend="yolov5", model_path=str(weights), yolov5_repo=str(args.repo.resolve()),
        device="0", half=False, image_size=args.image_size,
    )
    pytorch_model = Yolov5Segmenter(settings)
    pytorch_model.load()  # Reject detection-only weights before any export.
    names = pytorch_model.names
    del pytorch_model
    import torch
    from export import run  # Pinned YOLOv5 repo was configured by load().

    torch.cuda.empty_cache()
    exported = run(
        weights=str(weights), imgsz=[args.image_size, args.image_size], batch_size=1,
        device="0", half=True, dynamic=False, simplify=False,
        workspace=args.workspace_gib, include=("engine",),
    )
    if not exported or engine not in [Path(path).resolve() for path in exported] or not engine.is_file():
        raise SystemExit("YOLOv5 export failed. Inspect the exporter output; no valid new engine was reported.")
    info = {
        "backend": "yolov5-seg", "image_size": args.image_size, "batch_size": 1,
        "precision_request": "fp16", "names": [names[index] for index in range(len(names))],
        "engine_sha256": sha256_file(engine), "source_sha256": sha256_file(weights),
        "yolov5_revision": YOLOV5_REVISION, "environment": environment,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    sidecar = engine.with_suffix(".engine.json")
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(sidecar)
    engine_settings = settings.model_copy(update={"model_path": str(engine), "half": True})
    engine_model = Yolov5Segmenter(engine_settings)
    engine_model.load()  # Ensure the exported engine can be loaded and warmed up.
    print(f"Engine warmup passed: {engine}")
    print(f"Required metadata sidecar: {sidecar}")
    print("Next: compare PyTorch and TensorRT outputs on your task validation images before deployment.")


if __name__ == "__main__":
    main()
