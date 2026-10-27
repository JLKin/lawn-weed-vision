"""API fixtures use a fake backend; tests never download or load model weights."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from weed_vision.config import Settings
from weed_vision.main import create_app
from weed_vision.schemas import Detection


class FakeSegmenter:
    def __init__(self):
        self.ready = False
        self.model_version = "fake-seg-test-v1"
        self.images = []
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        self.ready = True

    def predict(self, image):
        self.images.append(image.copy())
        return [
            Detection(
                class_name="deployment_test_object",
                confidence=0.9,
                bbox_xyxy=(1.0, 2.0, 7.0, 8.0),
                centroid_px=(4.0, 5.0),
                area_px=36,
                mask_polygon_px=[(1.0, 2.0), (7.0, 2.0), (7.0, 8.0), (1.0, 8.0)],
                ground_position_m=None,
            )
        ]


def image_bytes(image_format="PNG", size=(16, 12), mode="RGB"):
    buffer = BytesIO()
    Image.new(mode, size, color="green").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture
def backend():
    return FakeSegmenter()


@pytest.fixture
def settings():
    # Explicit values isolate the contract from a developer's optional .env.
    return Settings(
        _env_file=None,
        model_path="models/yolov5n-seg.pt",
        device="cpu",
        max_upload_bytes=1024 * 1024,
        max_image_pixels=1024 * 1024,
    )


@pytest.fixture
def client(settings, backend):
    app = create_app(settings=settings, segmenter_factory=lambda _: backend)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def png():
    return image_bytes()
