"""Stateless sensor projection; no tracking, motion extrapolation, or implicit calibration."""

from pathlib import Path

import cv2
import numpy as np

from .errors import APIError
from .sensor_schemas import Calibration, MappedDetection, MappingContext, MappingResult, ProjectedTarget


def load_calibration(directory: str, context: MappingContext) -> Calibration | None:
    if context.calibration_id is None:
        return None
    root = Path(directory).resolve()
    path = (root / f"{context.calibration_id}.json").resolve()
    if not path.is_relative_to(root):
        raise APIError(422, "invalid_calibration", "Calibration must be inside the configured directory.")
    if not path.exists():
        return None
    try:
        if path.stat().st_size > 65536:
            raise ValueError("Calibration file exceeds 64 KiB")
        calibration = Calibration.model_validate_json(path.read_text(encoding="utf-8"))
        if calibration.calibration_id != context.calibration_id:
            raise ValueError("Calibration ID mismatch")
        return calibration
    except (OSError, ValueError) as exc:
        raise APIError(422, "invalid_calibration", "Calibration file is invalid; check its schema and geometry.") from exc


def sensor_time(context, sensor, tolerance_ms):
    if sensor.clock_id != context.clock_id:
        return None, "clock_mismatch"
    difference = (int(sensor.capture_timestamp_ns) - int(context.capture_timestamp_ns)) / 1e6
    return difference, "time_skew" if abs(difference) > tolerance_ms else None


def ground_intersection(pixel, calibration):
    k = np.array(calibration.camera_matrix, dtype=np.float64)
    dist = np.array(calibration.distortion, dtype=np.float64)
    normalized = cv2.undistortPoints(np.array([[pixel]], dtype=np.float64), k, dist)[0, 0]
    transform = np.array(calibration.T_base_from_camera)
    direction = transform[:3, :3] @ np.r_[normalized, 1.0]
    direction /= np.linalg.norm(direction)
    origin = transform[:3, 3]
    normal = np.array(calibration.ground_plane.normal_base)
    denominator = float(normal @ direction)
    if abs(denominator) < 1e-8:
        return None
    distance = -(float(normal @ origin) + calibration.ground_plane.offset_m) / denominator
    point = origin + distance * direction
    if distance <= 0 or distance > calibration.max_distance_m or not np.isfinite(point).all():
        return None
    return tuple(float(x) for x in point)


def map_scene(context: MappingContext, calibration: Calibration | None, detections, masks,
              image_size: tuple[int, int], tolerance_ms: float) -> MappingResult:
    width, height = image_size
    if len(detections) != len(masks) or any(mask.shape != (height, width) for mask in masks):
        raise ValueError("Native masks must align with all returned detections")
    reasons = []
    result = MappingResult(status="unavailable", reasons=reasons,
                           calibration_id=context.calibration_id, calibration_kind=None,
                           altitude=context.altitude, radar_targets=[], detections=[])
    # Pydantic copies mutable inputs: append only to the model-owned list.
    reasons = result.reasons
    if context.altitude:
        result.altitude_time_offset_ms, issue = sensor_time(context, context.altitude, tolerance_ms)
        if issue:
            reasons.append(f"altitude_{issue}")
        reasons.append("altitude_metadata_only")
    radar_issue = None
    if context.radar:
        result.radar_time_offset_ms, radar_issue = sensor_time(context, context.radar, tolerance_ms)
        if radar_issue:
            reasons.append(f"radar_{radar_issue}")
    else:
        reasons.append("radar_missing")

    unusable = None
    if calibration is None:
        unusable = "calibration_missing"
    elif calibration.camera_id != context.camera_id:
        unusable = "camera_mismatch"
    elif (calibration.image_width, calibration.image_height) != image_size:
        unusable = "image_size_mismatch"
    elif calibration.image_space != context.image_space:
        unusable = "image_space_mismatch"
    if unusable:
        reasons.append(unusable)
    else:
        result.calibration_kind = "synthetic" if calibration.synthetic else "measured"
        if calibration.synthetic:
            reasons.append("synthetic_calibration")

    for index, (detection, mask) in enumerate(zip(detections, masks)):
        item = MappedDetection(detection_index=index, status=unusable or "ground_plane_unavailable")
        if not unusable and calibration.ground_plane and context.ground_plane_valid:
            center = np.array(detection.centroid_px)
            x, y = np.rint(center).astype(int)
            if not (0 <= x < width and 0 <= y < height and mask[y, x]):
                yy, xx = np.nonzero(mask)
                if len(xx):
                    nearest = np.argmin((xx - center[0])**2 + (yy - center[1])**2)
                    x, y = int(xx[nearest]), int(yy[nearest])
                else:
                    item.status = "empty_mask"
                    result.detections.append(item)
                    continue
            item.reference_pixel_px = (float(x), float(y))
            item.ground_position_m = ground_intersection((x, y), calibration)
            item.status = "estimated" if item.ground_position_m is not None else "invalid_ground_intersection"
            if item.ground_position_m is not None:
                item.position_method = "ray_ground_plane"
        result.detections.append(item)
    if not unusable and (not calibration.ground_plane or not context.ground_plane_valid):
        reasons.append("ground_plane_unavailable")

    if context.radar:
        radar_geometry_issue = unusable or radar_issue
        if not radar_geometry_issue:
            if calibration.T_base_from_radar is None:
                radar_geometry_issue = "radar_calibration_missing"
            elif (calibration.radar_id, calibration.radar_frame_id) != (context.radar.radar_id, context.radar.frame_id):
                radar_geometry_issue = "radar_frame_mismatch"
        if radar_geometry_issue and radar_geometry_issue not in reasons:
            reasons.append(radar_geometry_issue)
        for target in context.radar.targets:
            item = ProjectedTarget(target_id=target.target_id, status=radar_geometry_issue or "insufficient_geometry")
            if radar_geometry_issue or target.coordinate_type != "cartesian3d":
                result.radar_targets.append(item)
                continue
            base = np.array(calibration.T_base_from_radar) @ np.r_[target.position_m, 1.0]
            camera = np.linalg.inv(np.array(calibration.T_base_from_camera)) @ base
            if not np.isfinite(camera).all() or np.linalg.norm(camera[:3]) > calibration.max_distance_m:
                item.status = "out_of_range"
            else:
                item.radar_position_base_m = tuple(float(v) for v in base[:3])
                if camera[2] <= 1e-8:
                    item.status = "behind_camera"
                else:
                    projected, _ = cv2.projectPoints(camera[:3].reshape(1, 3), np.zeros(3), np.zeros(3),
                        np.array(calibration.camera_matrix, dtype=float), np.array(calibration.distortion, dtype=float))
                    pixel = projected[0, 0]
                    if not np.isfinite(pixel).all():
                        item.status = "invalid_projection"
                    else:
                        item.pixel_px = tuple(float(v) for v in pixel)
                        x, y = np.rint(pixel).astype(np.int64) if np.max(np.abs(pixel)) < 1e9 else (-1, -1)
                        if 0 <= pixel[0] < width and 0 <= pixel[1] < height and 0 <= x < width and 0 <= y < height:
                            item.candidate_detection_indices = [i for i, mask in enumerate(masks) if mask[y, x]]
                            count = len(item.candidate_detection_indices)
                            item.status = "candidate" if count == 1 else "ambiguous" if count > 1 else "unassociated"
                        else:
                            item.status = "outside_image"
            result.radar_targets.append(item)
    valid_ground = any(x.ground_position_m is not None for x in result.detections)
    valid_radar = any(x.pixel_px is not None and x.status in {"candidate", "ambiguous", "unassociated"} for x in result.radar_targets)
    for item in result.radar_targets:
        if item.status not in {"candidate", "unassociated"}:
            reasons.append(item.status)
    for item in result.detections:
        if item.status != "estimated":
            reasons.append(item.status)
    result.reasons = list(dict.fromkeys(reasons))
    impairments = set(result.reasons) - {"altitude_metadata_only"}
    result.status = "partial" if (valid_ground or valid_radar) and impairments else "ready" if valid_ground or valid_radar else "unavailable"
    return result
