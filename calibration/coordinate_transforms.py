"""Pixel/robot transforms for the project's XYZ-only eye-in-hand camera.

The live camera pose is recomputed for every image:

    T_base_to_camera = T_base_to_end(X, Y, Z) @ T_end_to_camera

Robot R is intentionally ignored.  Z comes from the robot pose associated with
the captured frame; this module does not command motion or replace the existing
probe descent/threshold logic.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

try:
    from .eye_in_hand_xyz import compose_base_to_camera, make_transform, robot_xyz
except ImportError:  # Direct execution: python calibration/coordinate_transforms.py
    from eye_in_hand_xyz import compose_base_to_camera, make_transform, robot_xyz


class PixelToWorld:
    """Convert pixels using a per-frame MG400 XYZ pose."""

    def __init__(self, config_path=None, table_z=None, robot_pose=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "camera_config.yaml")

        with open(config_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        self.K = np.asarray(data["intrinsics"]["camera_matrix"], dtype=np.float64)
        self.D = np.asarray(data["intrinsics"]["dist_coeffs"], dtype=np.float64)
        self.img_w, self.img_h = data["calibration"]["resolution"]

        extrinsics = data.get("extrinsics", {})
        if "T_base_to_cam" in extrinsics:
            raise ValueError(
                "legacy fixed T_base_to_cam is invalid for the eye-in-hand camera; "
                "run the XYZ eye-in-hand calibration first"
            )
        if extrinsics.get("mount_mode") != "eye_in_hand_xyz":
            raise ValueError("extrinsics.mount_mode must be eye_in_hand_xyz")
        if extrinsics.get("status") != "calibrated":
            raise ValueError("eye-in-hand extrinsics are not calibrated")
        if extrinsics.get("robot_axes_used") != ["x", "y", "z"]:
            raise ValueError("eye-in-hand extrinsics must use robot axes x, y and z")
        if "r" not in extrinsics.get("robot_axes_ignored", []):
            raise ValueError("eye-in-hand extrinsics must explicitly ignore robot R")

        end_camera = extrinsics.get("T_end_to_camera", {})
        self.T_end_to_camera = make_transform(
            np.asarray(end_camera["R"], dtype=np.float64),
            end_camera["t_mm"],
        )

        correction = data.get("xy_pose_correction", {})
        self.xy_pose_correction = None
        if correction.get("status") == "calibrated":
            if correction.get("features") != [
                "raw_x", "raw_y", "robot_x", "robot_y", "robot_z", "constant"
            ]:
                raise ValueError("unsupported xy_pose_correction feature schema")
            coefficients = np.asarray(correction["coefficients"], dtype=np.float64)
            if coefficients.shape != (6, 2) or not np.all(np.isfinite(coefficients)):
                raise ValueError("xy_pose_correction coefficients must be finite 6x2")
            fixed_r_deg = float(correction["fixed_r_deg"])
            r_tolerance_deg = float(correction.get("r_tolerance_deg", 0.1))
            xyz_min = np.asarray(correction["valid_robot_xyz_min"], dtype=np.float64)
            xyz_max = np.asarray(correction["valid_robot_xyz_max"], dtype=np.float64)
            if (
                not np.isfinite(fixed_r_deg)
                or not np.isfinite(r_tolerance_deg)
                or r_tolerance_deg < 0
            ):
                raise ValueError("xy_pose_correction R guard must be finite and nonnegative")
            if (
                xyz_min.shape != (3,)
                or xyz_max.shape != (3,)
                or not np.all(np.isfinite(xyz_min))
                or not np.all(np.isfinite(xyz_max))
                or np.any(xyz_min > xyz_max)
            ):
                raise ValueError("xy_pose_correction XYZ range must be finite ordered 3-vectors")
            self.xy_pose_correction = {
                "coefficients": coefficients,
                "fixed_r_deg": fixed_r_deg,
                "r_tolerance_deg": r_tolerance_deg,
                "valid_robot_xyz_min": xyz_min,
                "valid_robot_xyz_max": xyz_max,
            }

        table_config = data.get("table_homography", {})
        self.table_z = (
            float(table_z)
            if table_z is not None
            else (
                float(table_config["table_z_mm"])
                if "table_z_mm" in table_config
                else None
            )
        )
        self._robot_pose = None
        if robot_pose is not None:
            self.set_robot_pose(robot_pose)

    def set_robot_pose(self, pose: Mapping[str, Any] | Sequence[float]):
        """Bind frame XYZ; R is checked as a fixed-calibration safety guard."""
        xyz = robot_xyz(pose)
        self._robot_pose = {"x": xyz[0], "y": xyz[1], "z": xyz[2]}
        if self.xy_pose_correction is not None:
            if not isinstance(pose, Mapping) or "r" not in pose:
                raise ValueError("robot R is required for calibrated XY correction")
            self._robot_pose["r"] = float(pose["r"])
        return self

    def set_table_z(self, z):
        self.table_z = float(z)
        return self

    def _resolve_robot_pose(self, robot_pose):
        if robot_pose is not None:
            xyz = robot_xyz(robot_pose)
            resolved = {"x": xyz[0], "y": xyz[1], "z": xyz[2]}
            if self.xy_pose_correction is not None:
                if not isinstance(robot_pose, Mapping) or "r" not in robot_pose:
                    raise ValueError("robot R is required for calibrated XY correction")
                resolved["r"] = float(robot_pose["r"])
            return resolved
        if self._robot_pose is None:
            raise ValueError(
                "robot XYZ pose is required for every eye-in-hand image"
            )
        return self._robot_pose

    def _correct_table_xy(self, raw_x, raw_y, pose):
        correction = self.xy_pose_correction
        if correction is None:
            return float(raw_x), float(raw_y)
        xyz = np.asarray([pose[key] for key in ("x", "y", "z")], dtype=np.float64)
        if np.any(xyz < correction["valid_robot_xyz_min"]) or np.any(
            xyz > correction["valid_robot_xyz_max"]
        ):
            raise ValueError("robot XYZ is outside calibrated XY correction range")
        r_error = abs(
            ((float(pose["r"]) - correction["fixed_r_deg"] + 180.0) % 360.0)
            - 180.0
        )
        if r_error > correction["r_tolerance_deg"]:
            raise ValueError(
                f"robot R differs by {r_error:.3f} deg from calibrated fixed R"
            )
        features = np.asarray(
            [raw_x, raw_y, pose["x"], pose["y"], pose["z"], 1.0],
            dtype=np.float64,
        )
        corrected = features @ correction["coefficients"]
        return float(corrected[0]), float(corrected[1])

    def base_to_camera(self, robot_pose=None):
        """Return the live ``T_base_to_camera``; robot R never participates."""
        return compose_base_to_camera(
            self._resolve_robot_pose(robot_pose), self.T_end_to_camera
        )

    def pixel_to_camera_ray(self, u, v):
        points = np.array([[[float(u), float(v)]]], dtype=np.float64)
        # P=None returns normalized camera coordinates.
        normalized = cv2.undistortPoints(points, self.K, self.D)
        x_norm, y_norm = normalized[0, 0]
        direction = np.array([x_norm, y_norm, 1.0], dtype=np.float64)
        return direction / np.linalg.norm(direction)

    def pixel_to_world_ray(self, u, v, robot_pose=None):
        transform = self.base_to_camera(robot_pose)
        direction = transform[:3, :3] @ self.pixel_to_camera_ray(u, v)
        direction /= np.linalg.norm(direction)
        origin = transform[:3, 3].copy()
        return origin, direction

    def pixel_to_table(self, u, v, robot_pose=None):
        if self.table_z is None:
            raise ValueError("table_z is required for ray/plane intersection")

        pose = self._resolve_robot_pose(robot_pose)
        origin, direction = self.pixel_to_world_ray(u, v, pose)
        if abs(direction[2]) < 1e-9:
            return None
        distance = (self.table_z - origin[2]) / direction[2]
        if distance <= 0:
            return None
        point = origin + direction * distance
        x, y = self._correct_table_xy(point[0], point[1], pose)
        return x, y, float(self.table_z)

    def world_to_pixel(self, x, y, z, robot_pose=None):
        transform = self.base_to_camera(robot_pose)
        camera_from_base = np.linalg.inv(transform)
        point_base = np.array([float(x), float(y), float(z), 1.0])
        point_camera = camera_from_base @ point_base
        if point_camera[2] <= 0:
            return None

        projected, _ = cv2.projectPoints(
            point_camera[:3].reshape(1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            self.K,
            self.D,
        )
        u, v = projected[0, 0]
        if not (0 <= u < self.img_w and 0 <= v < self.img_h):
            return None
        return float(u), float(v)


if __name__ == "__main__":
    print(
        "This module is computation-only. Instantiate PixelToWorld with a "
        "calibrated config and pass the robot XYZ pose captured with each image."
    )
