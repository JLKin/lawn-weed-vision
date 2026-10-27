"""Original YOLOv5 segmentation pipeline for .pt and native TensorRT .engine."""

from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .schemas import Detection
from .preprocessing import prepare_image
from .segmenter import detections_from_result
from .yolov5_runtime import configure_repo, normalized_names, read_engine_metadata, sha256_file


def split_segmentation_outputs(outputs, class_count: int):
    """Reject detection-only weights and incorrect engine labels before NMS."""
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
        raise ValueError("Expected YOLOv5 segmentation predictions and mask prototypes.")
    prediction, prototypes = outputs[:2]
    if prediction.ndim != 3 or prototypes.ndim != 4:
        raise ValueError("Invalid YOLOv5 segmentation output ranks.")
    if prediction.shape[0] != 1 or prototypes.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported.")
    mask_channels = prototypes.shape[1]
    if mask_channels < 1 or prediction.shape[2] != 5 + class_count + mask_channels:
        raise ValueError("Class names or mask channels do not match the segmentation model.")
    return prediction, prototypes, mask_channels


class Yolov5Segmenter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
        self.model_version = settings.model_version or Path(settings.model_path).name
        self._model = None
        self.names = {}

    def load(self) -> None:
        self.ready = False
        path = Path(self.settings.model_path).resolve()
        if not path.is_file() or path.suffix.lower() not in {".pt", ".engine"}:
            raise ValueError("Configure an existing YOLOv5 segmentation .pt or .engine file.")
        configure_repo(self.settings.yolov5_repo)
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device

        device = select_device(self.settings.device)
        metadata = None
        if path.suffix == ".engine":
            if device.type != "cuda":
                raise ValueError("TensorRT requires WEED_DEVICE=0 (CUDA).")
            metadata = read_engine_metadata(path, self.settings.image_size)
        self._model = DetectMultiBackend(
            str(path), device=device, fp16=self.settings.half and device.type == "cuda"
        )
        self.names = metadata["names"] if metadata else normalized_names(self._model.names)
        if metadata:
            expected = (1, 3, self.settings.image_size, self.settings.image_size)
            if self._model.dynamic or tuple(self._model.bindings["images"].shape) != expected:
                raise ValueError("Expected a fixed-size batch-1 engine matching WEED_IMAGE_SIZE.")
            if self._model.bindings["images"].dtype not in (np.float16, np.float32):
                raise ValueError("Only floating-point engine I/O is supported; quantization belongs inside the graph.")
        elif not hasattr(self._model.model.model[-1], "proto"):
            raise ValueError("Detection-only YOLOv5 weights cannot serve the segmentation endpoint.")
        if self.settings.image_size % int(self._model.stride):
            raise ValueError("WEED_IMAGE_SIZE must align to the model stride.")
        self.model_version = self.settings.model_version or f"{path.name}@sha256:{sha256_file(path)[:12]}"
        with Image.new("RGB", (self.settings.image_size, self.settings.image_size)) as probe:
            self._predict(probe)
        self.ready = True

    def _predict(self, image: Image.Image, retained_masks=None) -> list[Detection]:
        import torch
        from ultralytics.engine.results import Results
        from utils.general import non_max_suppression, scale_boxes
        from utils.segment.general import process_mask_native

        # PIL is already RGB; do not apply the BGR flip used by YOLOv5's file loader.
        original = np.asarray(image)
        tensor = torch.from_numpy(prepare_image(image, self.settings.image_size)).to(self._model.device)
        tensor = tensor.half() if self._model.fp16 else tensor.float()
        with torch.inference_mode():
            prediction, prototypes, mask_channels = split_segmentation_outputs(self._model(tensor), len(self.names))
            selected = non_max_suppression(
                prediction, self.settings.confidence, self.settings.iou_threshold,
                max_det=self.settings.max_detections, nm=mask_channels,
            )[0]
            if len(selected) == 0:
                return []
            selected[:, :4] = scale_boxes(tensor.shape[2:], selected[:, :4], original.shape).round()
            masks = process_mask_native(prototypes[0], selected[:, 6:], selected[:, :4], original.shape[:2])
            result = Results(orig_img=original, path="", names=self.names, boxes=selected[:, :6], masks=masks)
            return detections_from_result(result, image.size, retained_masks)

    def predict_with_masks(self, image: Image.Image):
        if not self.ready:
            raise RuntimeError("YOLOv5 model is not ready.")
        masks = []
        return self._predict(image, masks), masks

    def predict(self, image: Image.Image) -> list[Detection]:
        if not self.ready:
            raise RuntimeError("YOLOv5 model is not ready.")
        return self._predict(image)
