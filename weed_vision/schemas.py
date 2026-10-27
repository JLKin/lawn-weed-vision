"""Public response types. All pixel coordinates refer to the EXIF-oriented image."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox_xyxy: tuple[float, float, float, float]
    centroid_px: tuple[float, float]
    area_px: int = Field(gt=0)
    mask_polygon_px: list[tuple[float, float]] = Field(min_length=3)
    # Do not infer physical coordinates from a monocular, uncalibrated image.
    ground_position_m: None = None


class SegmentationResponse(BaseModel):
    request_id: str
    camera_id: str
    capture_timestamp_ns: int | None
    model_version: str
    image_width: int
    image_height: int
    inference_ms: float = Field(ge=0)
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_ready: bool
    model_version: str
    model_path: str
    detail: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail
