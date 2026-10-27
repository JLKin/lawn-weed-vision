"""Typed per-frame sensor inputs. New timestamps are decimal strings, not JS numbers."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
Vector3 = tuple[float, float, float]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Stamped(StrictModel):
    capture_timestamp_ns: str = Field(pattern=r"^(0|[1-9][0-9]{0,18})$")
    clock_id: Identifier

    @field_validator("capture_timestamp_ns")
    @classmethod
    def timestamp_range(cls, value):
        if int(value) > 2**63 - 1:
            raise ValueError("Timestamp exceeds int64 range")
        return value


class RadarXYZ(StrictModel):
    target_id: Identifier
    coordinate_type: Literal["cartesian3d"]
    position_m: Vector3
    radial_velocity_mps: float | None = None


class RadarPolar(StrictModel):
    target_id: Identifier
    coordinate_type: Literal["polar2d"]
    range_m: float = Field(gt=0)
    azimuth_rad: float = Field(ge=-3.141592653589793, le=3.141592653589793)
    radial_velocity_mps: float | None = None


class RadarFrame(Stamped):
    radar_id: Identifier
    frame_id: Identifier
    targets: list[Annotated[RadarXYZ | RadarPolar, Field(discriminator="coordinate_type")]] = Field(max_length=256)

    @model_validator(mode="after")
    def unique_targets(self):
        if len({t.target_id for t in self.targets}) != len(self.targets):
            raise ValueError("Radar target IDs must be unique within a frame")
        return self


class Altitude(Stamped):
    sensor_id: Identifier
    value_m: float
    reference: Literal["msl", "ellipsoid", "relative", "unknown"]
    uncertainty_m: float | None = Field(default=None, ge=0)


class MappingContext(Stamped):
    camera_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
    image_space: Literal["raw", "rectified"] = "raw"
    calibration_id: Identifier | None = None
    ground_plane_valid: bool = Field(default=False, strict=True)
    radar: RadarFrame | None = None
    altitude: Altitude | None = None


class GroundPlane(StrictModel):
    normal_base: Vector3
    offset_m: float


class Calibration(StrictModel):
    calibration_id: Identifier
    camera_id: str = Field(min_length=1, max_length=64)
    radar_id: Identifier | None = None
    radar_frame_id: Identifier | None = None
    synthetic: bool = False
    image_space: Literal["raw", "rectified"] = "raw"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    camera_matrix: tuple[Vector3, Vector3, Vector3]
    distortion: list[float] = Field(default_factory=lambda: [0, 0, 0, 0, 0])
    T_base_from_camera: list[list[float]]
    T_base_from_radar: list[list[float]] | None = None
    ground_plane: GroundPlane | None = None
    max_distance_m: float = Field(default=10.0, gt=0, le=1000)

    @model_validator(mode="after")
    def valid_geometry(self):
        import numpy as np
        k = np.array(self.camera_matrix)
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Invalid pinhole intrinsics")
        if not np.allclose([k[0, 1], k[1, 0]], 0):
            raise ValueError("Only zero-skew pinhole intrinsics are supported")
        if len(self.distortion) not in (4, 5, 8, 12, 14):
            raise ValueError("Unsupported OpenCV distortion coefficient count")
        if self.image_space == "rectified" and any(self.distortion):
            raise ValueError("Rectified images require zero distortion")
        for matrix in (self.T_base_from_camera, self.T_base_from_radar):
            if matrix is None:
                continue
            transform = np.asarray(matrix)
            if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
                raise ValueError("Expected a homogeneous 4x4 transform")
            rotation = transform[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
                raise ValueError("Rotation must be orthonormal with determinant +1")
        if self.ground_plane and not np.isclose(np.linalg.norm(self.ground_plane.normal_base), 1, atol=1e-6):
            raise ValueError("Ground plane normal must have unit length")
        return self


class MappedDetection(StrictModel):
    detection_index: int
    ground_position_m: Vector3 | None = None
    position_method: Literal["ray_ground_plane"] | None = None
    reference_pixel_px: tuple[float, float] | None = None
    status: str


class ProjectedTarget(StrictModel):
    target_id: str
    status: str
    radar_position_base_m: Vector3 | None = None
    pixel_px: tuple[float, float] | None = None
    candidate_detection_indices: list[int] = Field(default_factory=list)


class MappingResult(StrictModel):
    status: Literal["ready", "partial", "unavailable"]
    reasons: list[str]
    calibration_id: str | None
    calibration_kind: Literal["synthetic", "measured"] | None
    output_frame_id: Literal["base_link"] = "base_link"
    radar_time_offset_ms: float | None = None
    altitude_time_offset_ms: float | None = None
    altitude: Altitude | None = None
    radar_targets: list[ProjectedTarget]
    detections: list[MappedDetection]
