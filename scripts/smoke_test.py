"""Run the real model through the API using Ultralytics' bundled bus photograph.

Run from the project root: python -m scripts.smoke_test
This verifies deployment, not lawn/weed recognition accuracy.
"""

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient
from ultralytics.utils import ASSETS

from weed_vision.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke-response.json"))
    args = parser.parse_args()
    sample = ASSETS / "bus.jpg"
    if not sample.is_file():
        raise SystemExit(f"Bundled smoke-test image missing: {sample}")

    with TestClient(create_app()) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise SystemExit(f"Model is not ready: {health.text}")
        with sample.open("rb") as stream:
            response = client.post(
                "/v1/weed-segmentation",
                data={"camera_id": "deployment_smoke_test"},
                files={"file": ("bus.jpg", stream, "image/jpeg")},
            )
        response.raise_for_status()
        result = response.json()
        assert result["capture_timestamp_ns"] is None
        assert result["image_width"] > 0 and result["image_height"] > 0
        assert result["inference_ms"] >= 0
        assert result["detections"], "Expected COCO objects in the bundled bus image."
        for detection in result["detections"]:
            assert detection["ground_position_m"] is None
            assert detection["area_px"] > 0
            assert len(detection["mask_polygon_px"]) >= 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model_version": result["model_version"],
        "detections": len(result["detections"]),
        "classes": sorted({item["class_name"] for item in result["detections"]}),
        "inference_ms": result["inference_ms"],
        "response_file": str(args.output.resolve()),
        "scope": "COCO deployment smoke test only; no weed accuracy claim",
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
