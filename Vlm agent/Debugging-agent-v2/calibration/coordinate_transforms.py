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
        """Bind the XYZ pose captured with the current image; R is ignored."""
        xyz = robot_xyz(pose)
        self._robot_pose = {"x": xyz[0], "y": xyz[1], "z": xyz[2]}
        return self

    def set_table_z(self, z):
        self.table_z = float(z)
        return self

    def _resolve_robot_pose(self, robot_pose):
        if robot_pose is not None:
            xyz = robot_xyz(robot_pose)
            return {"x": xyz[0], "y": xyz[1], "z": xyz[2]}
        if self._robot_pose is None:
            raise ValueError(
                "robot XYZ pose is required for every eye-in-hand image"
            )
        return self._robot_pose

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

        origin, direction = self.pixel_to_world_ray(u, v, robot_pose)
        if abs(direction[2]) < 1e-9:
            return None
        distance = (self.table_z - origin[2]) / direction[2]
        if distance <= 0:
            return None
        point = origin + direction * distance
        return float(point[0]), float(point[1]), float(self.table_z)

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
