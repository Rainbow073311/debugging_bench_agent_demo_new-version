"""Fit and independently validate pose-aware Base-XY correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibrate_extrinsics import load_dataset
from capture_basler_extrinsics_sample import _canonical_grid, _detect, _object_points
from eye_in_hand_xyz import compose_base_to_camera, estimate_eye_in_hand_xyz


HERE = Path(__file__).resolve().parent


def _project_pixels_to_plane(
    pixels: np.ndarray,
    pose: dict,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    t_end_to_camera: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2).astype(np.float64), camera_matrix, distortion
    ).reshape(-1, 2)
    rays_camera = np.column_stack((normalized, np.ones(len(normalized))))
    t_base_camera = compose_base_to_camera(pose, t_end_to_camera)
    origin = t_base_camera[:3, 3]
    rays_base = (t_base_camera[:3, :3] @ rays_camera.T).T
    scale = (plane_z - origin[2]) / rays_base[:, 2]
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise RuntimeError("One or more chessboard rays do not meet the table plane")
    return origin + scale[:, None] * rays_base


def _frame_rows(
    session_dir: Path,
    dataset: dict,
    session: dict,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    t_end_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    board = session["board"]
    pattern = tuple(int(value) for value in board["pattern_inner_corners"])
    object_points = _object_points(pattern, float(board["square_size_mm"]))
    board_pose = session["board_pose_in_base_operational"]
    board_rotation = np.asarray(board_pose["R"], dtype=np.float64)
    board_translation = np.asarray(board_pose["translation_mm"], dtype=np.float64)
    expected_base = (
        board_rotation @ object_points.astype(np.float64).T
    ).T + board_translation
    orientation = session["orientation"]
    plane_z = float(dataset["target_plane_z_mm"])
    features = []
    targets = []
    frame_rmse = []
    for sample in dataset["samples"]:
        image = cv2.imread(str(session_dir / sample["file"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {sample['file']}")
        found, corners, _ = _detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pattern)
        if not found or corners is None:
            raise RuntimeError(f"Could not redetect all corners in {sample['file']}")
        pixels = _canonical_grid(corners, pattern, orientation).reshape(-1, 2)
        raw_base = _project_pixels_to_plane(
            pixels,
            sample["robot_pose"],
            camera_matrix,
            distortion,
            t_end_to_camera,
            plane_z,
        )
        pose = sample["robot_pose"]
        frame_features = np.column_stack(
            (
                raw_base[:, :2],
                np.full((len(raw_base), 1), float(pose["x"])),
                np.full((len(raw_base), 1), float(pose["y"])),
                np.full((len(raw_base), 1), float(pose["z"])),
                np.ones((len(raw_base), 1)),
            )
        )
        features.append(frame_features)
        targets.append(expected_base[:, :2])
        raw_error = np.linalg.norm(raw_base[:, :2] - expected_base[:, :2], axis=1)
        frame_rmse.append(float(np.sqrt(np.mean(raw_error * raw_error))))
    return np.vstack(features), np.vstack(targets), frame_rmse


def _fit_robust(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inliers = np.ones(len(features), dtype=bool)
    coefficients = np.linalg.lstsq(features, targets, rcond=None)[0]
    for _ in range(5):
        error = np.linalg.norm(features @ coefficients - targets, axis=1)
        median = float(np.median(error[inliers]))
        mad = float(np.median(np.abs(error[inliers] - median)))
        threshold = max(0.75, median + 3.5 * 1.4826 * mad)
        updated = error <= threshold
        if np.array_equal(updated, inliers):
            break
        inliers = updated
        coefficients = np.linalg.lstsq(features[inliers], targets[inliers], rcond=None)[0]
    return coefficients, inliers


def _metrics(features: np.ndarray, targets: np.ndarray, coefficients: np.ndarray) -> dict:
    errors = np.linalg.norm(features @ coefficients - targets, axis=1)
    return {
        "points": int(len(errors)),
        "mean_error_mm": float(np.mean(errors)),
        "median_error_mm": float(np.median(errors)),
        "max_error_mm": float(np.max(errors)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", default=str(HERE / "camera_config.yaml"))
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    session_dir = Path(args.session)
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8-sig"))
    training = load_dataset(session_dir / "samples.json")
    validation = load_dataset(session_dir / "validation_samples.json")
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    camera_matrix = np.asarray(config["intrinsics"]["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(config["intrinsics"]["dist_coeffs"], dtype=np.float64)
    marker = training["marker_pose_in_base"]
    calibration = estimate_eye_in_hand_xyz(
        training["samples"], marker["translation_mm"], marker["R"], minimum_samples=10
    )
    train_x, train_y, raw_train_rmse = _frame_rows(
        session_dir, training, session, camera_matrix, distortion, calibration.t_end_to_camera
    )
    valid_x, valid_y, raw_valid_rmse = _frame_rows(
        session_dir, validation, session, camera_matrix, distortion, calibration.t_end_to_camera
    )
    coefficients, inliers = _fit_robust(train_x, train_y)
    train_metrics = _metrics(train_x[inliers], train_y[inliers], coefficients)
    validation_metrics = _metrics(valid_x, valid_y, coefficients)
    poses = np.asarray(
        [[s["robot_pose"][k] for k in ("x", "y", "z")] for s in training["samples"]],
        dtype=np.float64,
    )
    range_margin_mm = 0.1
    result = {
        "status": "calibrated",
        "method": "robust linear Base-XY correction after eye-in-hand ray-plane projection",
        "features": ["raw_x", "raw_y", "robot_x", "robot_y", "robot_z", "constant"],
        "coefficients": coefficients.tolist(),
        "fixed_r_deg": float(training["fixed_r_reference_deg"]),
        "r_tolerance_deg": 0.1,
        "valid_robot_xyz_min": (poses.min(axis=0) - range_margin_mm).tolist(),
        "valid_robot_xyz_max": (poses.max(axis=0) + range_margin_mm).tolist(),
        "valid_range_margin_mm": range_margin_mm,
        "training": {
            "session": str(Path("calibration_sessions") / session_dir.name),
            "frames_total": len(training["samples"]),
            "chessboard_points_total": int(len(train_x)),
            "inliers_used": int(np.count_nonzero(inliers)),
            **train_metrics,
            "raw_frame_rmse_mm": raw_train_rmse,
        },
        "independent_holdout": {
            "frames": len(validation["samples"]),
            **validation_metrics,
            "raw_frame_rmse_mm": raw_valid_rmse,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.write:
        if validation_metrics["mean_error_mm"] > 1.5 or validation_metrics["max_error_mm"] > 3.0:
            raise RuntimeError("Independent XY validation exceeds promotion limits")
        config["xy_pose_correction"] = result
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"wrote: {config_path}")
    else:
        print("dry-run only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
