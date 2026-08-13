"""Eye-in-hand geometry for the MG400 with J1-rotating end frame.

Transform convention
--------------------
``T_a_to_b`` stores the pose of frame ``b`` expressed in frame ``a`` and maps
homogeneous coordinates from frame ``b`` into frame ``a``.  Therefore:

    T_base_to_camera = Trans(TCP_xyz) @ Rz(J1) @ T_end_to_camera

``T_end_to_camera`` is fixed in the arm-head / J1 frame.  Robot flange ``R``
(J4) is deliberately ignored.  ``J1`` comes from ``pose.j1_deg`` / ``pose.j1``
when present, otherwise ``atan2(TCP_y, TCP_x)`` (MG400 proxy).
"""

from __future__ import annotations

import math
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


def _fit_rigid_translation_model(
    marker_vectors_camera: np.ndarray,
    marker_offsets_from_end_base: np.ndarray,
    minimum_observability_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``y = t + R @ x`` and reject geometrically weak datasets."""
    camera_centered = marker_vectors_camera - np.mean(marker_vectors_camera, axis=0)
    base_centered = marker_offsets_from_end_base - np.mean(
        marker_offsets_from_end_base, axis=0
    )
    covariance = base_centered.T @ camera_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    if singular_values[0] <= np.finfo(np.float64).eps:
        raise ValueError("calibration translations have no measurable variation")
    observability_ratio = singular_values[-1] / singular_values[0]
    if observability_ratio < minimum_observability_ratio:
        raise ValueError(
            "calibration translations are geometrically degenerate; covariance "
            f"singular values were {', '.join(f'{value:.4f}' for value in singular_values)} "
            f"(minimum ratio {minimum_observability_ratio:.6f})"
        )

    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = np.mean(
        marker_offsets_from_end_base
        - (rotation @ marker_vectors_camera.T).T,
        axis=0,
    )
    return rotation, translation, singular_values


def robot_xyz(pose: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    """Return robot XYZ and intentionally ignore any flange R value."""
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


def j1_deg_from_pose(pose: Mapping[str, Any] | Sequence[float]) -> float:
    """Return J1 in degrees: explicit pose field, else atan2(TCP_y, TCP_x)."""
    if isinstance(pose, Mapping):
        for key in ("j1_deg", "j1"):
            if key in pose and pose[key] is not None:
                value = float(pose[key])
                if not math.isfinite(value):
                    raise ValueError(f"robot pose.{key} must be finite")
                return value
        xyz = robot_xyz(pose)
        return math.degrees(math.atan2(float(xyz[1]), float(xyz[0])))
    xyz = robot_xyz(pose)
    return math.degrees(math.atan2(float(xyz[1]), float(xyz[0])))


def rot_z(j1_deg: float) -> np.ndarray:
    """Rotation about base/end Z by J1 (degrees)."""
    angle = math.radians(float(j1_deg))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _as_rotation(rotation, "rotation")
    transform[:3, 3] = _as_translation(translation, "translation")
    return transform


def robot_xyz_to_base_end(pose: Mapping[str, Any] | Sequence[float]) -> np.ndarray:
    """Build ``T_base_to_end = Trans(TCP) @ Rz(J1)``; flange R never participates."""
    return make_transform(rot_z(j1_deg_from_pose(pose)), robot_xyz(pose))


def compose_base_to_camera(
    robot_pose: Mapping[str, Any] | Sequence[float],
    t_end_to_camera: np.ndarray,
) -> np.ndarray:
    """Compose live camera pose: ``Trans(TCP) @ Rz(J1) @ T_end_to_camera``."""
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
    translation_fit_singular_values: tuple[float, float, float]
    rotation_crosscheck_error_deg: float


def estimate_eye_in_hand_xyz(
    samples: Sequence[Mapping[str, Any]],
    marker_translation_in_base: Sequence[float],
    marker_rotation_in_base: Sequence[Sequence[float]],
    *,
    minimum_samples: int = 5,
    minimum_axis_span_mm: float = 10.0,
    minimum_observability_ratio: float = 1e-3,
) -> EyeInHandCalibration:
    """Estimate fixed end-to-camera transform in the J1-rotating end frame.

    Each sample contains robot XYZ (and optional ``j1_deg``) plus the marker pose
    in the camera.  Marker pose in base must be independently known.  Flange R is
    accepted but never used.
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
    j1_degs = np.array(
        [j1_deg_from_pose(sample["robot_pose"]) for sample in samples], dtype=np.float64
    )
    spans = np.ptp(robot_positions, axis=0)
    if np.any(spans < minimum_axis_span_mm):
        raise ValueError(
            "calibration samples must vary X, Y and Z by at least "
            f"{minimum_axis_span_mm:.1f} mm; observed spans were "
            f"{spans[0]:.1f}, {spans[1]:.1f}, {spans[2]:.1f} mm"
        )

    marker_vectors_camera = np.array(
        [
            _as_translation(
                sample["marker_pose_in_camera"]["t_mm"],
                "marker_pose_in_camera.t_mm",
            )
            for sample in samples
        ],
        dtype=np.float64,
    )
    # Express (P_marker - TCP) in the J1 end frame.
    marker_offsets_from_end = np.array(
        [
            rot_z(-j1) @ (marker_t_base - tcp)
            for tcp, j1 in zip(robot_positions, j1_degs)
        ],
        dtype=np.float64,
    )
    kabsch_r_end, _, singular_values = _fit_rigid_translation_model(
        marker_vectors_camera,
        marker_offsets_from_end,
        minimum_observability_ratio,
    )

    # PnP path: R_base_cam = Rz(J1) @ R_ec  =>  R_ec = Rz(-J1) @ R_base_cam
    marker_rotations_camera = [rotation_from_sample(sample) for sample in samples]
    pnp_r_end = _project_to_rotation(
        rot_z(-j1) @ (marker_r_base @ marker_r_camera.T)
        for j1, marker_r_camera in zip(j1_degs, marker_rotations_camera)
    )

    def _solve(rotation: np.ndarray):
        translation = np.mean(
            marker_offsets_from_end - (rotation @ marker_vectors_camera.T).T,
            axis=0,
        )
        residuals = [
            float(
                np.linalg.norm(
                    tcp
                    + rot_z(j1) @ (translation + rotation @ marker_t_camera)
                    - marker_t_base
                )
            )
            for tcp, j1, marker_t_camera in zip(
                robot_positions, j1_degs, marker_vectors_camera
            )
        ]
        return translation, residuals

    pnp_t_end, pnp_residuals = _solve(pnp_r_end)
    kabsch_t_end, kabsch_residuals = _solve(kabsch_r_end)
    if np.mean(kabsch_residuals) < np.mean(pnp_residuals):
        camera_r_end, camera_t_end, residuals = (
            kabsch_r_end,
            kabsch_t_end,
            kabsch_residuals,
        )
    else:
        camera_r_end, camera_t_end, residuals = pnp_r_end, pnp_t_end, pnp_residuals

    transform = make_transform(camera_r_end, camera_t_end)
    rotation_delta = pnp_r_end.T @ kabsch_r_end
    rotation_crosscheck_error_deg = float(
        np.degrees(
            np.arccos(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
        )
    )

    return EyeInHandCalibration(
        t_end_to_camera=transform,
        marker_residuals_mm=tuple(residuals),
        mean_residual_mm=float(np.mean(residuals)),
        max_residual_mm=float(np.max(residuals)),
        robot_xyz_span_mm=tuple(float(value) for value in spans),
        translation_fit_singular_values=tuple(
            float(value) for value in singular_values
        ),
        rotation_crosscheck_error_deg=rotation_crosscheck_error_deg,
    )
