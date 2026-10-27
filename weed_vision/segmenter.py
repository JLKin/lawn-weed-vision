"""Ultralytics adapter. No weed class names are invented or remapped here."""

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .config import Settings
from .schemas import Detection


class Segmenter(Protocol):
    ready: bool
    model_version: str

    def load(self) -> None: ...

    def predict(self, image: Image.Image) -> list[Detection]: ...


def create_segmenter(settings: Settings) -> Segmenter:
    if settings.backend == "yolov5":
        from .yolov5_segmenter import Yolov5Segmenter

        return Yolov5Segmenter(settings)
    return UltralyticsSegmenter(settings)


def detections_from_result(result, image_size: tuple[int, int], retained_masks=None) -> list[Detection]:
    """Convert native-resolution masks; reject inconsistent backend output."""
    width, height = image_size
    if result.boxes is None or len(result.boxes) == 0:
        return []
    if result.masks is None or len(result.masks.data) != len(result.boxes):
        raise RuntimeError("Segmentation masks are missing or do not match the boxes.")
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    # Ultralytics returns one pixel contour per instance, possibly joining components.
    polygons = result.masks.xy
    if len(polygons) != len(boxes):
        raise RuntimeError("Segmentation contour count does not match the boxes.")

    detections = []
    for index, box in enumerate(boxes):
        binary = result.masks.data[index].cpu().numpy() > 0.5
        if binary.shape != (height, width):
            raise RuntimeError("Expected native image-sized masks; enable retina_masks.")
        area = int(np.count_nonzero(binary))
        polygon = np.asarray(polygons[index], dtype=np.float64)
        # Empty / sub-pixel instances have no usable region and are omitted.
        if area == 0 or polygon.ndim != 2 or polygon.shape[0] < 3:
            continue
        if polygon.shape[1] != 2 or not np.isfinite(polygon).all():
            raise RuntimeError("Invalid mask polygon returned by the model.")

        # Pixel centers use integer coordinates: top-left pixel center is (0, 0).
        center_x = float(np.dot(binary.sum(axis=0), np.arange(width)) / area)
        center_y = float(np.dot(binary.sum(axis=1), np.arange(height)) / area)
        clipped_box = np.asarray(box, dtype=np.float64).copy()
        clipped_box[[0, 2]] = np.clip(clipped_box[[0, 2]], 0, width)
        clipped_box[[1, 3]] = np.clip(clipped_box[[1, 3]], 0, height)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

        detections.append(
            Detection(
                class_name=str(result.names[int(classes[index])]),
                confidence=float(confidences[index]),
                bbox_xyxy=tuple(float(value) for value in clipped_box),
                centroid_px=(center_x, center_y),
                area_px=area,
                mask_polygon_px=[tuple(float(v) for v in point) for point in polygon],
                ground_position_m=None,
            )
        )
        if retained_masks is not None:
            retained_masks.append(binary)
    return detections


class UltralyticsSegmenter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
        self.model_version = settings.model_version or Path(settings.model_path).name
        self._model = None

    def load(self) -> None:
        # Lazy import lets API contract tests run without loading Torch or weights.
        from ultralytics import YOLO

        self.ready = False
        model_path = Path(self.settings.model_path)
        if model_path.suffix.lower() not in {".pt", ".engine"}:
            raise ValueError("Use a trusted Ultralytics segmentation .pt or .engine file.")
        if model_path.suffix.lower() == ".engine" and not model_path.is_file():
            raise FileNotFoundError("The configured TensorRT engine does not exist.")
        self._model = YOLO(self.settings.model_path, task="segment")
        if self._model.task != "segment":
            raise ValueError("The configured model is not an instance segmentation model.")

        resolved = Path(getattr(self._model, "ckpt_path", None) or self.settings.model_path)
        if not self.settings.model_version:
            if not resolved.is_file():
                raise FileNotFoundError("Cannot identify the loaded model artifact.")
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.model_version = f"{resolved.name}@sha256:{digest.hexdigest()[:12]}"

        # Also initializes the predictor / engine so /health reflects usable inference.
        probe = Image.new("RGB", (self.settings.image_size, self.settings.image_size))
        self._predict(probe)
        self.ready = True

    def _predict(self, image: Image.Image, retained_masks=None) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("Model is not loaded.")
        results = self._model.predict(
            source=image,
            device=self.settings.device,
            imgsz=self.settings.image_size,
            conf=self.settings.confidence,
            max_det=self.settings.max_detections,
            retina_masks=True,
            rect=False,
            verbose=False,
            save=False,
            stream=False,
        )
        return detections_from_result(results[0], image.size, retained_masks)

    def predict_with_masks(self, image: Image.Image):
        if not self.ready:
            raise RuntimeError("Model is not ready.")
        masks = []
        return self._predict(image, masks), masks

    def predict(self, image: Image.Image) -> list[Detection]:
        if not self.ready:
            raise RuntimeError("Model is not ready.")
        return self._predict(image)
