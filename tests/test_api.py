"""Boundary and service-contract tests independent of Ultralytics and the network."""

from io import BytesIO
import math
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from weed_vision.main import create_app

from conftest import FakeSegmenter, image_bytes


ENDPOINT = "/v1/weed-segmentation"


def assert_error(response, status, code=None):
    assert response.status_code == status, response.text
    body = response.json()
    assert UUID(body["request_id"])
    assert response.headers["x-request-id"] == body["request_id"]
    assert isinstance(body["error"]["message"], str)
    assert body["error"]["message"]
    assert isinstance(body["error"]["code"], str)
    if code is not None:
        assert body["error"]["code"] == code
    return body


def upload(client, data, *, camera_id="front.camera:01", mime="image/png", extra=None):
    form = {"camera_id": camera_id}
    form.update(extra or {})
    return client.post(ENDPOINT, data=form, files={"file": ("frame", data, mime)})


def test_health_reports_loaded_backend(client, backend):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_ready": True,
        "model_version": backend.model_version,
        "model_path": "yolov5n-seg.pt",
        "detail": None,
    }
    assert backend.load_calls == 1


@pytest.mark.parametrize("timestamp", [None, "0", "1792735200123456789", str(2**63 - 1)])
def test_segmentation_preserves_metadata_and_pixel_contract(client, backend, png, timestamp):
    extra = {} if timestamp is None else {"capture_timestamp_ns": timestamp}
    response = upload(client, png, extra=extra)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {
        "request_id", "camera_id", "capture_timestamp_ns", "model_version",
        "image_width", "image_height", "inference_ms", "detections",
    }
    assert UUID(body["request_id"])
    assert response.headers["x-request-id"] == body["request_id"]
    assert body["camera_id"] == "front.camera:01"
    assert body["capture_timestamp_ns"] == (None if timestamp is None else int(timestamp))
    assert body["model_version"] == backend.model_version
    assert (body["image_width"], body["image_height"]) == (16, 12)
    assert math.isfinite(body["inference_ms"]) and body["inference_ms"] >= 0
    assert body["detections"] == [{
        "class_name": "deployment_test_object",
        "confidence": 0.9,
        "bbox_xyxy": [1.0, 2.0, 7.0, 8.0],
        "centroid_px": [4.0, 5.0],
        "area_px": 36,
        "mask_polygon_px": [[1.0, 2.0], [7.0, 2.0], [7.0, 8.0], [1.0, 8.0]],
        "ground_position_m": None,
    }]
    assert backend.images[0].mode == "RGB"


@pytest.mark.parametrize("image_format,mime", [
    ("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp"),
])
def test_supported_image_formats(client, image_format, mime):
    response = upload(client, image_bytes(image_format), mime=mime)
    assert response.status_code == 200, response.text


def test_request_ids_are_unique(client, png):
    first = upload(client, png).json()["request_id"]
    second = upload(client, png).json()["request_id"]
    assert first != second


def test_rgba_image_is_converted_to_rgb(client, backend):
    response = upload(client, image_bytes(mode="RGBA"))
    assert response.status_code == 200
    assert backend.images[0].mode == "RGB"


def test_exif_orientation_is_applied_before_inference(client, backend):
    image = Image.new("RGB", (16, 12), color="green")
    exif = image.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise for display.
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    response = upload(client, buffer.getvalue(), mime="image/jpeg")
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["image_width"], body["image_height"]) == (12, 16)
    assert backend.images[0].size == (12, 16)


def test_empty_detection_list_is_valid(client, backend, png):
    backend.predict = lambda _: []
    response = upload(client, png)
    assert response.status_code == 200
    assert response.json()["detections"] == []


@pytest.mark.parametrize("data", [b"", b"this is not an image", b"\x89PNG\r\n\x1a\n"])
def test_empty_or_corrupt_image_is_rejected_before_inference(client, backend, data):
    assert_error(upload(client, data), 400)
    assert backend.images == []


@pytest.mark.parametrize("mime", ["application/octet-stream", "image/svg+xml", "text/plain", "image/gif"])
def test_unapproved_mime_is_rejected(client, backend, png, mime):
    assert_error(upload(client, png, mime=mime), 415, "unsupported_image_type")
    assert backend.images == []


def test_mime_must_match_actual_image(client, backend, png):
    assert_error(upload(client, png, mime="image/jpeg"), 415, "image_type_mismatch")
    assert backend.images == []


def test_animated_png_is_rejected(client, backend):
    buffer = BytesIO()
    first = Image.new("RGB", (16, 12), "green")
    second = Image.new("RGB", (16, 12), "blue")
    first.save(buffer, format="PNG", save_all=True, append_images=[second], duration=100, loop=0)
    assert_error(upload(client, buffer.getvalue()), 400, "animated_image")
    assert backend.images == []


def test_uploaded_file_size_limit(settings, backend, png):
    settings = settings.model_copy(update={"max_upload_bytes": len(png) - 1})
    app = create_app(settings=settings, segmenter_factory=lambda _: backend)
    with TestClient(app) as client:
        assert_error(upload(client, png), 413, "image_too_large")
    assert backend.images == []


@pytest.mark.parametrize("advertised_length", [None, "1"])
def test_total_body_limit_counts_received_bytes_even_without_honest_length(settings, backend, advertised_length):
    settings = settings.model_copy(update={"max_upload_bytes": 100})
    app = create_app(settings=settings, segmenter_factory=lambda _: backend)
    headers = {"Content-Type": "multipart/form-data; boundary=boundary"}
    if advertised_length is not None:
        headers["Content-Length"] = advertised_length
    chunks = [b"x" * 32768, b"y" * 32768, b"z" * 101]
    with TestClient(app) as client:
        response = client.post(ENDPOINT, headers=headers, content=iter(chunks))
        assert_error(response, 413, "request_too_large")
    assert backend.images == []


def test_total_body_limit_checks_content_length(settings, backend):
    settings = settings.model_copy(update={"max_upload_bytes": 100})
    app = create_app(settings=settings, segmenter_factory=lambda _: backend)
    with TestClient(app) as client:
        response = client.post(ENDPOINT, content=b"x" * (settings.max_request_bytes + 1))
        assert_error(response, 413, "request_too_large")
    assert backend.images == []


def test_decoded_pixel_limit(settings, backend, png):
    settings = settings.model_copy(update={"max_image_pixels": 16 * 12 - 1})
    app = create_app(settings=settings, segmenter_factory=lambda _: backend)
    with TestClient(app) as client:
        assert_error(upload(client, png), 413, "image_too_large")
    assert backend.images == []


@pytest.mark.parametrize("camera_id", ["", " has-space", "camera/a", "摄像头", "c" * 65])
def test_invalid_camera_id(client, backend, png, camera_id):
    assert_error(upload(client, png, camera_id=camera_id), 422)
    assert backend.images == []


@pytest.mark.parametrize("timestamp", ["-1", str(2**63), "1.5", "invalid"])
def test_invalid_capture_timestamp(client, backend, png, timestamp):
    assert_error(upload(client, png, extra={"capture_timestamp_ns": timestamp}), 422)
    assert backend.images == []


def test_missing_camera_id(client, backend, png):
    response = client.post(ENDPOINT, files={"file": ("frame.png", png, "image/png")})
    assert_error(response, 422)
    assert backend.images == []


def test_missing_image(client, backend):
    assert_error(client.post(ENDPOINT, data={"camera_id": "front"}), 422)
    assert backend.images == []


def test_model_load_failure_keeps_health_and_inference_unavailable(settings, png):
    class UnavailableSegmenter(FakeSegmenter):
        def load(self):
            raise RuntimeError("Private model path C:/private/weights/secret.pt failed")

    backend = UnavailableSegmenter()
    app = create_app(settings=settings, segmenter_factory=lambda _: backend)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "unavailable"
        assert health.json()["model_ready"] is False
        assert "secret.pt" not in health.text
        response = upload(client, png)
        assert_error(response, 503)
        assert "secret.pt" not in response.text
    assert backend.images == []


def test_inference_error_is_sanitized(client, backend, png):
    def explode(_):
        raise RuntimeError("Private CUDA details C:/private/weights/secret.pt")

    backend.predict = explode
    response = upload(client, png)
    assert_error(response, 500, "inference_failed")
    assert "secret.pt" not in response.text
    assert "RuntimeError" not in response.text
    assert "CUDA" not in response.text


def test_unknown_route_uses_json_error_envelope(client):
    assert_error(client.get("/nonexistent"), 404)
