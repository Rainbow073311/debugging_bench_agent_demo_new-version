import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from calibration.calibrate_extrinsics import build_extrinsics, load_dataset, write_config as write_calibration_config
from calibration.coordinate_transforms import PixelToWorld
from calibration.eye_in_hand_xyz import (
    compose_base_to_camera,
    estimate_eye_in_hand_xyz,
    make_transform,
)
from perception.height_estimator import HeightEstimator


DOWNWARD_ROTATION = np.diag([1.0, -1.0, -1.0])


def synthetic_samples():
    marker_base = make_transform(np.eye(3), [320.0, 40.0, -228.0])
    end_camera = make_transform(DOWNWARD_ROTATION, [12.0, -8.0, 35.0])
    robot_poses = [
        {"x": 240.0, "y": -40.0, "z": 100.0, "r": -90.0},
        {"x": 260.0, "y": 0.0, "z": 120.0, "r": -30.0},
        {"x": 280.0, "y": 40.0, "z": 140.0, "r": 0.0},
        {"x": 300.0, "y": -20.0, "z": 160.0, "r": 45.0},
        {"x": 320.0, "y": 20.0, "z": 160.0, "r": 120.0},
    ]
    samples = []
    for pose in robot_poses:
        base_camera = compose_base_to_camera(pose, end_camera)
        camera_marker = np.linalg.inv(base_camera) @ marker_base
        rvec, _ = cv2.Rodrigues(camera_marker[:3, :3])
        samples.append(
            {
                "robot_pose": pose,
                "marker_pose_in_camera": {
                    "rvec": rvec.reshape(3).tolist(),
                    "t_mm": camera_marker[:3, 3].tolist(),
                },
            }
        )
    return marker_base, end_camera, samples


def write_config(path, extrinsics):
    data = {
        "calibration": {"resolution": [640, 480]},
        "intrinsics": {
            "camera_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": [[0.0, 0.0, 0.0, 0.0, 0.0]],
        },
        "extrinsics": extrinsics,
        "table_homography": {
            "table_z_mm": 0.0,
            "H": np.eye(3).tolist(),
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def calibrated_extrinsics():
    return {
        "mount_mode": "eye_in_hand_xyz",
        "status": "calibrated",
        "robot_axes_used": ["x", "y", "z"],
        "robot_axes_ignored": ["r"],
        "camera_orientation_source": "fixed_calibration",
        "T_end_to_camera": {
            "R": DOWNWARD_ROTATION.tolist(),
            "t_mm": [10.0, 20.0, 30.0],
        },
    }


def test_constrained_calibration_recovers_end_to_camera_and_ignores_r():
    marker_base, expected, samples = synthetic_samples()
    result = estimate_eye_in_hand_xyz(
        samples,
        marker_base[:3, 3],
        marker_base[:3, :3],
    )
    np.testing.assert_allclose(result.t_end_to_camera, expected, atol=1e-8)
    assert result.max_residual_mm < 1e-8

    changed_r = json.loads(json.dumps(samples))
    for index, sample in enumerate(changed_r):
        sample["robot_pose"]["r"] = 1000.0 + index
    changed = estimate_eye_in_hand_xyz(
        changed_r,
        marker_base[:3, 3],
        marker_base[:3, :3],
    )
    np.testing.assert_allclose(changed.t_end_to_camera, expected, atol=1e-8)


def test_pnp_orientation_fit_recovers_noisy_fixed_orientation():
    marker_base, expected, samples = synthetic_samples()
    rng = np.random.default_rng(20260803)
    noisy = json.loads(json.dumps(samples))
    for sample in noisy:
        measured = np.asarray(sample["marker_pose_in_camera"]["t_mm"])
        sample["marker_pose_in_camera"]["t_mm"] = (
            measured + rng.normal(0.0, 0.15, size=3)
        ).tolist()
        sample["marker_pose_in_camera"]["rvec"][0] += float(
            rng.normal(0.0, 0.03)
        )

    result = estimate_eye_in_hand_xyz(
        noisy, marker_base[:3, 3], marker_base[:3, :3]
    )
    np.testing.assert_allclose(
        result.t_end_to_camera[:3, 3], expected[:3, 3], atol=0.8
    )
    np.testing.assert_allclose(
        result.t_end_to_camera[:3, :3], expected[:3, :3], atol=0.01
    )
    assert result.mean_residual_mm < 0.3


def test_translation_fit_rejects_degenerate_samples():
    marker_base, _, samples = synthetic_samples()
    degenerate = json.loads(json.dumps(samples))
    for index, sample in enumerate(degenerate):
        sample["robot_pose"].update(
            {"x": 200.0 + 20.0 * index, "y": 10.0, "z": 100.0}
        )
        sample["marker_pose_in_camera"]["t_mm"] = [
            50.0 - 20.0 * index, 20.0, 300.0
        ]

    with pytest.raises(ValueError, match="geometrically degenerate"):
        estimate_eye_in_hand_xyz(
            degenerate,
            marker_base[:3, 3],
            marker_base[:3, :3],
            minimum_axis_span_mm=0.0,
        )


def test_live_camera_pose_uses_xyz_but_not_r():
    end_camera = make_transform(DOWNWARD_ROTATION, [10.0, 20.0, 30.0])
    first = compose_base_to_camera(
        {"x": 100.0, "y": 200.0, "z": 300.0, "r": 0.0}, end_camera
    )
    different_r = compose_base_to_camera(
        {"x": 100.0, "y": 200.0, "z": 300.0, "r": 173.2}, end_camera
    )
    different_z = compose_base_to_camera(
        {"x": 100.0, "y": 200.0, "z": 305.0, "r": 173.2}, end_camera
    )
    np.testing.assert_allclose(first, different_r)
    np.testing.assert_allclose(different_z[:3, 3] - first[:3, 3], [0.0, 0.0, 5.0])


def test_pixel_projection_requires_frame_pose_and_intersects_table(tmp_path):
    config = tmp_path / "camera.yaml"
    write_config(config, calibrated_extrinsics())
    converter = PixelToWorld(config)

    with pytest.raises(ValueError, match="robot XYZ pose is required"):
        converter.pixel_to_table(320, 240)

    center = converter.pixel_to_table(
        320,
        240,
        robot_pose={"x": 100.0, "y": 200.0, "z": 300.0, "r": 999.0},
    )
    np.testing.assert_allclose(center, [110.0, 220.0, 0.0], atol=1e-8)

    same_center = converter.pixel_to_table(
        320,
        240,
        robot_pose={"x": 100.0, "y": 200.0, "z": 300.0, "r": -999.0},
    )
    np.testing.assert_allclose(center, same_center, atol=1e-8)


def test_pose_aware_xy_correction_requires_fixed_r_and_valid_range(tmp_path):
    config = tmp_path / "camera.yaml"
    write_config(config, calibrated_extrinsics())
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["xy_pose_correction"] = {
        "status": "calibrated",
        "features": [
            "raw_x", "raw_y", "robot_x", "robot_y", "robot_z", "constant"
        ],
        "coefficients": [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, -3.0],
        ],
        "fixed_r_deg": 7.686619,
        "r_tolerance_deg": 0.1,
        "valid_robot_xyz_min": [90.0, 190.0, 290.0],
        "valid_robot_xyz_max": [110.0, 210.0, 310.0],
    }
    config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    converter = PixelToWorld(config)

    corrected = converter.pixel_to_table(
        320,
        240,
        robot_pose={"x": 100.0, "y": 200.0, "z": 300.0, "r": 7.686619},
    )
    np.testing.assert_allclose(corrected, [112.0, 217.0, 0.0], atol=1e-8)

    with pytest.raises(ValueError, match="robot R differs"):
        converter.pixel_to_table(
            320, 240, robot_pose={"x": 100, "y": 200, "z": 300, "r": 8.0}
        )
    with pytest.raises(ValueError, match="outside calibrated XY correction range"):
        converter.pixel_to_table(
            320, 240, robot_pose={"x": 120, "y": 200, "z": 300, "r": 7.686619}
        )


def test_legacy_fixed_camera_transform_is_rejected(tmp_path):
    config = tmp_path / "legacy.yaml"
    write_config(config, {"T_base_to_cam": {"R": np.eye(3).tolist(), "t": [0, 0, 1]}})
    with pytest.raises(ValueError, match="legacy fixed T_base_to_cam"):
        PixelToWorld(config)


def test_height_estimator_camera_position_uses_xyz_and_ignores_r(tmp_path):
    config = tmp_path / "camera.yaml"
    write_config(config, calibrated_extrinsics())
    estimator = HeightEstimator(config)
    estimator.set_robot_pose({"x": 100, "y": 200, "z": 300, "r": 0})
    first = estimator.cam_t.copy()
    estimator.set_robot_pose({"x": 100, "y": 200, "z": 300, "r": 173})
    np.testing.assert_allclose(estimator.cam_t, first)
    estimator.set_robot_pose({"x": 100, "y": 200, "z": 305, "r": -173})
    np.testing.assert_allclose(estimator.cam_t - first, [0, 0, 5])
    height = estimator.estimate(
        {"x": 150.0, "y": 260.0, "pixel_area": 1000},
        [(0, 0)] * 10,
    )
    assert np.isfinite(height)


def test_calibration_dataset_and_output_schema(tmp_path):
    marker_base, _, samples = synthetic_samples()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "marker_pose_in_base": {
                    "translation_mm": marker_base[:3, 3].tolist(),
                    "R": marker_base[:3, :3].tolist(),
                },
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    dataset = load_dataset(dataset_path)
    result = estimate_eye_in_hand_xyz(
        dataset["samples"],
        dataset["marker_pose_in_base"]["translation_mm"],
        dataset["marker_pose_in_base"]["R"],
    )
    output = build_extrinsics(result, len(samples))
    assert output["mount_mode"] == "eye_in_hand_xyz"
    assert output["robot_axes_used"] == ["x", "y", "z"]
    assert output["robot_axes_ignored"] == ["r"]
    assert "T_base_to_cam" not in output


def test_extrinsic_write_replaces_fixed_camera_homography_with_target_plane(tmp_path):
    config = tmp_path / "camera.yaml"
    write_config(config, calibrated_extrinsics())
    write_calibration_config(config, calibrated_extrinsics(), target_plane_z_mm=-228.0)
    written = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert written["table_homography"]["table_z_mm"] == -228.0
    assert written["table_homography"]["source"] == "eye_in_hand_target_plane_in_robot_base"
    assert "H" not in written["table_homography"]
    assert "H" in written["legacy_table_homography_do_not_use"]


def test_standalone_vlm_calibration_copy_matches_canonical_files():
    repo = Path(__file__).resolve().parents[1]
    standalone = repo / "Vlm agent" / "Debugging-agent-v2" / "calibration"
    for name in (
        "eye_in_hand_xyz.py",
        "coordinate_transforms.py",
        "calibrate_extrinsics.py",
        "verify_transform.py",
    ):
        assert (repo / "calibration" / name).read_bytes() == (standalone / name).read_bytes()
