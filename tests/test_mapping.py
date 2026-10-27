import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from weed_vision.main import create_app
from weed_vision.mapping import map_scene
from weed_vision.sensor_schemas import Calibration, MappingContext
from conftest import FakeSegmenter


def calibration_data():
    # Synthetic downward-facing camera at z=1. Optical x -> base y, y -> base x.
    return dict(calibration_id="synthetic", camera_id="front_rgb", synthetic=True,
        radar_id="radar0", radar_frame_id="radar_frame", image_width=16, image_height=12,
        camera_matrix=[[10, 0, 4], [0, 10, 5], [0, 0, 1]],
        T_base_from_camera=[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]],
        T_base_from_radar=np.eye(4).tolist(), ground_plane={"normal_base": [0, 0, 1], "offset_m": 0})


def context_data():
    return dict(camera_id="front_rgb", capture_timestamp_ns="1792735200123456789", clock_id="jetson",
        calibration_id="synthetic", ground_plane_valid=True,
        radar=dict(radar_id="radar0", frame_id="radar_frame", capture_timestamp_ns="1792735200123456789",
                   clock_id="jetson", targets=[dict(target_id="r1", coordinate_type="cartesian3d", position_m=[0, 0, 0])]))


class MappingBackend(FakeSegmenter):
    def predict_with_masks(self, image):
        detections = self.predict(image)
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[2:8, 1:7] = True
        return detections, [mask]


def scene(context=None, calibration=None, hole=False):
    from PIL import Image
    backend = MappingBackend()
    backend.load()
    detections, masks = backend.predict_with_masks(Image.new("RGB", (16, 12)))
    if hole:
        masks[0][5, 4] = False
    return map_scene(MappingContext.model_validate(context or context_data()),
        Calibration.model_validate(calibration or calibration_data()), detections, masks, (16, 12), 50)


def test_known_ground_truth_projection_and_ground_intersection():
    result = scene()
    assert result.radar_targets[0].pixel_px == (4, 5)
    assert result.radar_targets[0].radar_position_base_m == (0, 0, 0)
    assert result.radar_targets[0].candidate_detection_indices == [0]
    assert result.detections[0].ground_position_m == (0, 0, 0)
    assert result.calibration_kind == "synthetic" and result.status == "partial"


def test_mask_hole_is_not_matched_using_visualization_polygon():
    result = scene(hole=True)
    assert result.radar_targets[0].status == "unassociated"
    assert result.detections[0].reference_pixel_px != (4, 5)


@pytest.mark.parametrize("mutation,reason", [("clock", "clock_mismatch"), ("time", "time_skew"), ("frame", "radar_frame_mismatch")])
def test_invalid_sensor_alignment_is_excluded(mutation, reason):
    data = context_data()
    if mutation == "clock": data["radar"]["clock_id"] = "other"
    if mutation == "time": data["radar"]["capture_timestamp_ns"] = str(int(data["capture_timestamp_ns"]) + 50_000_001)
    if mutation == "frame": data["radar"]["frame_id"] = "wrong"
    point = scene(context=data).radar_targets[0]
    assert point.status == reason and point.pixel_px is None


def test_time_tolerance_boundary_and_polar_not_fabricated_3d():
    data = context_data()
    data["radar"]["capture_timestamp_ns"] = str(int(data["capture_timestamp_ns"]) + 50_000_000)
    assert scene(context=data).radar_targets[0].status == "candidate"
    data["radar"]["targets"] = [dict(target_id="r1", coordinate_type="polar2d", range_m=1, azimuth_rad=0)]
    assert scene(context=data).radar_targets[0].status == "insufficient_geometry"


def test_ground_plane_requires_explicit_current_validity():
    data = context_data()
    data["ground_plane_valid"] = False
    assert scene(context=data).detections[0].ground_position_m is None


@pytest.mark.parametrize("position,expected", [([0, 0, 2], "behind_camera"), ([0, 1, 0], "unassociated"), ([0, 3, 0], "outside_image"), ([0, 0, -30], "out_of_range")])
def test_invalid_and_unassociated_radar_points(position, expected):
    data = context_data()
    data["radar"]["targets"][0]["position_m"] = position
    assert scene(context=data).radar_targets[0].status == expected


@pytest.mark.parametrize("value", [1792735200123456789, "9223372036854775808", "1e9", "-1"])
def test_new_context_rejects_numeric_or_out_of_range_timestamps(value):
    data = context_data()
    data["capture_timestamp_ns"] = value
    with pytest.raises(ValidationError): MappingContext.model_validate(data)


@pytest.mark.parametrize("mutation", ["reflection", "nan", "shape", "normal", "rectified"])
def test_invalid_calibration_geometry(mutation):
    data = calibration_data()
    if mutation == "reflection": data["T_base_from_radar"][0][0] = -1
    if mutation == "nan": data["camera_matrix"][0][0] = float("nan")
    if mutation == "shape": data["T_base_from_camera"] = [[1]]
    if mutation == "normal": data["ground_plane"]["normal_base"] = [0, 0, 2]
    if mutation == "rectified": data.update(image_space="rectified", distortion=[1, 0, 0, 0])
    with pytest.raises(ValidationError): Calibration.model_validate(data)


def test_mapping_http_preserves_contract_and_degrades_without_calibration(settings, tmp_path, png):
    settings.calibration_dir = str(tmp_path)
    app = create_app(settings, lambda _: MappingBackend())
    with TestClient(app) as client:
        data = context_data()
        response = client.post("/v1/scene-mapping", data={"context": json.dumps(data)}, files={"file": ("a.png", png, "image/png")})
        assert response.status_code == 200
        body = response.json()
        assert body["capture_timestamp_ns"] == data["capture_timestamp_ns"]
        assert body["mapping"]["status"] == "unavailable"
        assert body["detections"][0]["ground_position_m"] is None
        (tmp_path / "synthetic.json").write_text(json.dumps(calibration_data()))
        response = client.post("/v1/scene-mapping", data={"context": json.dumps(data)}, files={"file": ("a.png", png, "image/png")})
        assert response.status_code == 200
        assert response.json()["mapping"]["detections"][0]["ground_position_m"] == [0, 0, 0]
        data["calibration_id"] = "../escape"
        assert client.post("/v1/scene-mapping", data={"context": json.dumps(data)}, files={"file": ("a.png", png, "image/png")}).status_code == 422


def test_altitude_alone_is_not_depth():
    data = context_data()
    data["ground_plane_valid"] = False
    data["altitude"] = dict(sensor_id="gps", value_m=123.0, reference="msl", clock_id="jetson", capture_timestamp_ns=data["capture_timestamp_ns"])
    result = scene(context=data)
    assert result.altitude.value_m == 123
    assert result.detections[0].ground_position_m is None


def test_multiple_overlapping_masks_preserve_ambiguity():
    from PIL import Image
    backend = MappingBackend()
    backend.load()
    detections, masks = backend.predict_with_masks(Image.new("RGB", (16, 12)))
    result = map_scene(MappingContext.model_validate(context_data()), Calibration.model_validate(calibration_data()),
        detections * 2, masks * 2, (16, 12), 50)
    assert result.radar_targets[0].status == "ambiguous"
    assert result.radar_targets[0].candidate_detection_indices == [0, 1]


def test_parallel_ray_never_produces_ground_coordinate():
    calibration = calibration_data()
    calibration["ground_plane"] = {"normal_base": [1, 0, 0], "offset_m": -1}
    assert scene(calibration=calibration).detections[0].ground_position_m is None


def test_distortion_is_applied_to_radar_projection():
    data = context_data()
    data["radar"]["targets"][0]["position_m"] = [0, .2, 0]
    calibration = calibration_data()
    calibration["distortion"] = [1, 0, 0, 0, 0]
    # x normalized=.2 -> distorted .2*(1+.2^2) -> pixel 6.08.
    assert scene(context=data, calibration=calibration).radar_targets[0].pixel_px == pytest.approx((6.08, 5))


def test_bad_context_and_calibration_return_structured_errors(settings, tmp_path, png):
    settings.calibration_dir = str(tmp_path)
    with TestClient(create_app(settings, lambda _: MappingBackend())) as client:
        for raw in ('not-json', json.dumps({**context_data(), "unknown": 1})):
            r = client.post('/v1/scene-mapping', data={'context': raw}, files={'file': ('a.png', png, 'image/png')})
            assert r.status_code == 422 and r.json()['request_id']
        (tmp_path / 'synthetic.json').write_text('{}')
        r = client.post('/v1/scene-mapping', data={'context': json.dumps(context_data())}, files={'file': ('a.png', png, 'image/png')})
        assert r.status_code == 422 and r.json()['error']['code'] == 'invalid_calibration'


def test_mapping_rejects_rotated_exif(settings):
    from PIL import Image
    from io import BytesIO
    image = Image.new("RGB", (16, 12))
    exif = Image.Exif()
    exif[274] = 6
    stream = BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    with TestClient(create_app(settings, lambda _: MappingBackend())) as client:
        response = client.post("/v1/scene-mapping", data={"context": json.dumps(context_data())}, files={"file": ("a.jpg", stream.getvalue(), "image/jpeg")})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "image_orientation_mismatch"
