"""Constrained eye-in-hand geometry for the MG400 XYZ-only camera model.

Transform convention
--------------------
``T_a_to_b`` stores the pose of frame ``b`` expressed in frame ``a`` and maps
homogeneous coordinates from frame ``b`` into frame ``a``.  Therefore:

    T_base_to_camera = T_base_to_end @ T_end_to_camera

The installation used by this project has a fixed camera orientation.  Robot
X/Y/Z translate the camera, while robot R is deliberately ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


XYZ_KEYS = ("x", "y", "z")


def _as_translation(value: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain three finite millimetre values")
    return result


def _as_rotation(value: Sequence[Sequence[float]], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 3x3 rotation matrix")
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-5):
        raise ValueError(f"{label} is not orthonormal")
    if not np.isclose(np.linalg.det(result), 1.0, atol=1e-5):
        raise ValueError(f"{label} must have determinant +1")
    return result


def _project_to_rotation(matrices: Iterable[np.ndarray]) -> np.ndarray:
    values = list(matrices)
    if not values:
        raise ValueError("at least one rotation sample is required")
    u, _, vt = np.linalg.svd(np.sum(values, axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def robot_xyz(pose: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    """Return robot XYZ and intentionally ignore any R value."""
    if isinstance(pose, Mapping):
        missing = [key for key in XYZ_KEYS if key not in pose]
        if missing:
            raise ValueError(f"robot pose is missing {', '.join(missing)}")
        values = [pose[key] for key in XYZ_KEYS]
    else:
        values = list(pose)
        if len(values) < 3:
            raise ValueError("robot pose must contain at least X, Y and Z")
        values = values[:3]
    return _as_translation(values, "robot XYZ")


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _as_rotation(rotation, "rotation")
    transform[:3, 3] = _as_translation(translation, "translation")
    return transform


def robot_xyz_to_base_end(pose: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    """Build ``T_base_to_end`` using translation only; R never participates."""
    return make_transform(np.eye(3), robot_xyz(pose))


def compose_base_to_camera(
    robot_pose: Mapping[str, Any] | Sequence[float],
    t_end_to_camera: np.ndarray,
) -> np.ndarray:
    """Compose the live camera pose from robot XYZ and fixed mount extrinsics."""
    transform = np.asarray(t_end_to_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_end_to_camera must be a finite 4x4 matrix")
    return robot_xyz_to_base_end(robot_pose) @ transform


def rotation_from_sample(sample: Mapping[str, Any]) -> np.ndarray:
    marker = sample["marker_pose_in_camera"]
    if "R" in marker:
        return _as_rotation(marker["R"], "marker_pose_in_camera.R")
    if "rvec" in marker:
        rvec = _as_translation(marker["rvec"], "marker_pose_in_camera.rvec")
        rotation, _ = cv2.Rodrigues(rvec)
        return rotation
    raise ValueError("marker_pose_in_camera requires R or rvec")


@dataclass(frozen=True)
class EyeInHandCalibration:
    t_end_to_camera: np.ndarray
    marker_residuals_mm: tuple[float, ...]
    mean_residual_mm: float
    max_residual_mm: float
    robot_xyz_span_mm: tuple[float, float, float]


def estimate_eye_in_hand_xyz(
    samples: Sequence[Mapping[str, Any]],
    marker_translation_in_base: Sequence[float],
    marker_rotation_in_base: Sequence[Sequence[float]],
    *,
    minimum_samples: int = 5,
    minimum_axis_span_mm: float = 10.0,
) -> EyeInHandCalibration:
    """Estimate a fixed end-to-camera transform from a known stationary marker.

    Each sample contains robot XYZ and the ArUco marker pose measured in the
    camera frame.  Marker position and orientation in the robot base frame must
    be independently known.  Robot R is accepted in sample dictionaries but is
    never used.
    """
    if len(samples) < minimum_samples:
        raise ValueError(f"at least {minimum_samples} calibration samples are required")

    marker_t_base = _as_translation(
        marker_translation_in_base, "marker_translation_in_base"
    )
    marker_r_base = _as_rotation(marker_rotation_in_base, "marker_rotation_in_base")

    robot_positions = np.array(
        [robot_xyz(sample["robot_pose"]) for sample in samples], dtype=np.float64
    )
    spans = np.ptp(robot_positions, axis=0)
    if np.any(spans < minimum_axis_span_mm):
        raise ValueError(
            "calibration samples must vary X, Y and Z by at least "
            f"{minimum_axis_span_mm:.1f} mm; observed spans were "
            f"{spans[0]:.1f}, {spans[1]:.1f}, {spans[2]:.1f} mm"
        )

    marker_rotations_camera = [rotation_from_sample(sample) for sample in samples]
    camera_rotations_base = [
        marker_r_base @ marker_r_camera.T
        for marker_r_camera in marker_rotations_camera
    ]
    camera_r_base = _project_to_rotation(camera_rotations_base)

    offsets = []
    marker_vectors_camera = []
    for robot_position, sample in zip(robot_positions, samples):
        marker_t_camera = _as_translation(
            sample["marker_pose_in_camera"]["t_mm"],
            "marker_pose_in_camera.t_mm",
        )
        marker_vectors_camera.append(marker_t_camera)
        offsets.append(
            marker_t_base - robot_position - camera_r_base @ marker_t_camera
        )

    camera_t_end = np.mean(offsets, axis=0)
    transform = make_transform(camera_r_base, camera_t_end)

    residuals = []
    for robot_position, marker_t_camera in zip(
        robot_positions, marker_vectors_camera
    ):
        predicted_marker = (
            robot_position + camera_t_end + camera_r_base @ marker_t_camera
        )
        residuals.append(float(np.linalg.norm(predicted_marker - marker_t_base)))

    return EyeInHandCalibration(
        t_end_to_camera=transform,
        marker_residuals_mm=tuple(residuals),
        mean_residual_mm=float(np.mean(residuals)),
        max_residual_mm=float(np.max(residuals)),
        robot_xyz_span_mm=tuple(float(value) for value in spans),
    )
