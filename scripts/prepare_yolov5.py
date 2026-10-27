"""Fetch a pinned upstream source revision and verified COCO smoke-test weights."""

import argparse
from pathlib import Path
import subprocess
import urllib.request

from weed_vision.yolov5_runtime import (
    SMOKE_MODEL_SHA256, SMOKE_MODEL_URL, YOLOV5_REPOSITORY, YOLOV5_REVISION, sha256_file,
)


def prepare_repo(repo: Path) -> None:
    if (repo / ".git").exists():
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
        if head != YOLOV5_REVISION or dirty:
            raise RuntimeError("Existing YOLOv5 checkout differs from the pinned clean revision; use a new directory.")
        return
    if repo.exists() and any(repo.iterdir()):
        raise RuntimeError("Refusing to overwrite a nonempty source directory.")
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", YOLOV5_REPOSITORY], check=True)
    subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", YOLOV5_REVISION], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], check=True)


def prepare_weights(path: Path) -> None:
    if path.exists():
        if sha256_file(path) != SMOKE_MODEL_SHA256:
            raise RuntimeError("Existing model differs from the official smoke-test artifact; refusing to overwrite it.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.download")
    try:
        with urllib.request.urlopen(SMOKE_MODEL_URL, timeout=60) as response, temporary.open("wb") as target:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                target.write(block)
        if sha256_file(temporary) != SMOKE_MODEL_SHA256:
            raise RuntimeError("Downloaded weight checksum mismatch.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("vendor/yolov5"))
    parser.add_argument("--weights", type=Path, default=Path("models/yolov5n-seg.pt"))
    args = parser.parse_args()
    prepare_repo(args.repo)
    prepare_weights(args.weights)
    print(f"YOLOv5 source ready: {YOLOV5_REVISION}")
    print(f"COCO deployment smoke-test weights ready: {args.weights.resolve()}")
    print("No lawn/weed-specific model has been trained or supplied.")


if __name__ == "__main__":
    main()
