"""Pinned original YOLOv5 runtime and artifact metadata shared by setup/export/API."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

YOLOV5_REVISION = "af08ba156fd58ed5b0fe7e4e58a22c72c3cdd0ce"
YOLOV5_REPOSITORY = "https://github.com/ultralytics/yolov5.git"
SMOKE_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt"
SMOKE_MODEL_SHA256 = "1ee00e5e14e3863898c5446c62734776759e29be08fd1c174774992f7aaa9985"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_repo(repo_path: str | Path) -> Path:
    """Make the original repo's models/utils imports available for trusted checkpoints."""
    repo = Path(repo_path).resolve()
    if not (repo / "models/common.py").is_file() or not (repo / "utils/segment/general.py").is_file():
        raise FileNotFoundError("YOLOv5 source missing. Run: python -m scripts.prepare_yolov5")
    for name in ("models", "utils"):
        module = sys.modules.get(name)
        if module is not None:
            location = getattr(module, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(repo):
                raise RuntimeError(f"Conflicting Python package {name!r}; use a dedicated service process.")
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    # Require this dependency before upstream code can try to install it itself.
    importlib.import_module("ultralytics")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def normalized_names(names) -> dict[int, str]:
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise ValueError("Model class names must be a list or a dictionary.")
    result = {int(key): str(value) for key, value in names.items()}
    if not result or set(result) != set(range(len(result))) or any(not value for value in result.values()):
        raise ValueError("Model class IDs must be contiguous from zero, with nonempty names.")
    return result


def read_engine_metadata(engine: Path, image_size: int) -> dict:
    metadata = json.loads(engine.with_suffix(".engine.json").read_text(encoding="utf-8"))
    if metadata.get("backend") != "yolov5-seg" or metadata.get("batch_size") != 1:
        raise ValueError("Expected a YOLOv5 segmentation engine with batch size 1.")
    if metadata.get("image_size") != image_size:
        raise ValueError("WEED_IMAGE_SIZE must match the fixed engine input size.")
    if metadata.get("engine_sha256") != sha256_file(engine):
        raise ValueError("Engine checksum does not match its metadata sidecar.")
    from .preprocessing import PREPROCESS_VERSION
    if metadata.get("preprocessing", PREPROCESS_VERSION) != PREPROCESS_VERSION:
        raise ValueError("Engine preprocessing version does not match this runtime.")
    metadata["names"] = normalized_names(metadata.get("names"))
    return metadata
