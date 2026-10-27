"""Upload one frame to the local segmentation API; timestamps are never invented."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import httpx


MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def timestamp_ns(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must be an integer") from exc
    if not 0 <= parsed <= 2**63 - 1:
        raise argparse.ArgumentTypeError("timestamp must be in [0, 2^63 - 1]")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="JPEG, PNG or WebP file")
    parser.add_argument("--camera-id", required=True, help="Camera name, e.g. front_rgb")
    parser.add_argument("--capture-timestamp-ns", type=timestamp_ns, default=None)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout in seconds")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image file does not exist: {args.image}")
    mime_type = MIME_TYPES.get(args.image.suffix.lower())
    if mime_type is None:
        parser.error("image extension must be .jpg, .jpeg, .png or .webp")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be a finite, positive number")

    data = {"camera_id": args.camera_id}
    if args.capture_timestamp_ns is not None:
        data["capture_timestamp_ns"] = str(args.capture_timestamp_ns)

    try:
        with args.image.open("rb") as image_file, httpx.Client(timeout=args.timeout) as client:
            response = client.post(
                args.url.rstrip("/") + "/v1/weed-segmentation",
                data=data,
                files={"file": (args.image.name, image_file, mime_type)},
            )
    except (OSError, httpx.HTTPError, httpx.InvalidURL) as exc:
        print(json.dumps({"error": "request_failed", "message": str(exc)}), file=sys.stderr)
        return 1

    try:
        payload = response.json()
    except ValueError:
        print(
            json.dumps({"error": "non_json_response", "http_status": response.status_code}),
            file=sys.stderr,
        )
        return 1

    if response.status_code != 200:
        print(
            json.dumps({"http_status": response.status_code, "response": payload}, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 1
    if not isinstance(payload, dict) or "detections" not in payload:
        print(json.dumps({"error": "unexpected_response_shape"}), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
