"""Validated configuration from WEED_* environment variables and an optional .env."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WEED_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    backend: Literal["yolov5", "ultralytics"] = "yolov5"
    model_path: str = Field(default="models/yolov5n-seg.pt", min_length=1)
    yolov5_repo: str = "vendor/yolov5"
    model_version: str | None = Field(default=None, min_length=1, max_length=160)
    device: str = Field(default="cpu", min_length=1)
    half: bool = False
    confidence: float = Field(default=0.25, gt=0, le=1)
    iou_threshold: float = Field(default=0.45, gt=0, le=1)
    image_size: int = Field(default=640, ge=32, le=2048)
    max_detections: int = Field(default=100, ge=1, le=300)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=12_000_000, ge=1)
    calibration_dir: str = "calibrations"
    max_sensor_skew_ms: float = Field(default=50.0, ge=0, le=1000, allow_inf_nan=False)

    @field_validator("image_size")
    @classmethod
    def image_size_must_be_stride_aligned(cls, value: int) -> int:
        if value % 32:
            raise ValueError("image_size must be a multiple of 32")
        return value

    @property
    def max_request_bytes(self) -> int:
        # Includes multipart boundaries and small metadata fields.
        return self.max_upload_bytes + 64 * 1024
