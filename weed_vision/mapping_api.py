"""Separate mapping route keeps the existing segmentation contract stable."""

from io import BytesIO
import logging
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from PIL import Image
from pydantic import ValidationError

from .errors import APIError
from .images import decode_image
from .mapping import load_calibration, map_scene
from .schemas import ErrorResponse, SegmentationResponse
from .sensor_schemas import MappingContext, MappingResult

logger = logging.getLogger(__name__)


class SceneMappingResponse(SegmentationResponse):
    # This new endpoint uses a string to preserve nanoseconds in browsers.
    capture_timestamp_ns: str
    mapping: MappingResult


def mapping_router(config):
    router = APIRouter()

    @router.post("/v1/scene-mapping", response_model=SceneMappingResponse,
                 summary="单帧图像与雷达局部映射", tags=["传感器映射"],
                 responses={code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 503)})
    def scene_mapping(request: Request, file: Annotated[UploadFile, File()],
                      context: Annotated[str, Form(max_length=49152)]):
        try:
            data = file.file.read(config.max_upload_bytes + 1)
        finally:
            file.file.close()
        if len(context.encode("utf-8")) > 49152:
            raise APIError(413, "context_too_large", "Sensor context exceeds 48 KiB.")
        try:
            sensors = MappingContext.model_validate_json(context)
        except ValidationError as exc:
            raise APIError(422, "invalid_sensor_context", "Invalid sensor context; check timestamps, frames and schema.") from exc
        image = decode_image(data, file.content_type, config)
        try:
            with Image.open(BytesIO(data)) as raw:
                if raw.getexif().get(274, 1) != 1:
                    raise APIError(422, "image_orientation_mismatch", "Mapping requires orientation 1 pixels matching camera calibration.")
            calibration = load_calibration(config.calibration_dir, sensors)
            segmenter = request.app.state.segmenter
            if segmenter is None or not segmenter.ready:
                raise APIError(503, "model_unavailable", "The segmentation model is unavailable. Check /health.")
            if not hasattr(segmenter, "predict_with_masks"):
                raise APIError(503, "mapping_backend_unavailable", "Backend does not provide native masks for mapping.")
            try:
                with request.app.state.inference_lock:
                    start = perf_counter()
                    detections, masks = segmenter.predict_with_masks(image)
                    elapsed = (perf_counter() - start) * 1000
                mapping = map_scene(sensors, calibration, detections, masks, image.size, config.max_sensor_skew_ms)
            except Exception as exc:
                logger.exception("Mapping failed request_id=%s", request.state.request_id)
                raise APIError(500, "mapping_failed", "Mapping failed; inspect logs using request_id.") from exc
            return SceneMappingResponse(request_id=request.state.request_id, camera_id=sensors.camera_id,
                capture_timestamp_ns=sensors.capture_timestamp_ns, model_version=segmenter.model_version,
                image_width=image.width, image_height=image.height, inference_ms=round(elapsed, 3),
                detections=detections, mapping=mapping)
        finally:
            image.close()

    return router
