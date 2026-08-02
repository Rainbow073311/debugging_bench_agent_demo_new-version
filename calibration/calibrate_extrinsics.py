"""Offline XYZ-only eye-in-hand extrinsic calibration.

This command performs no robot or camera I/O.  It consumes observations that
were captured with the robot already stationary.  Robot X/Y/Z are used and R is
explicitly ignored.

Dataset format::

    {
      "target_plane_z_mm": -228.0,
      "marker_pose_in_base": {
        "translation_mm": [300, 0, -228],
        "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      },
      "samples": [
        {
          "robot_pose": {"x": 250, "y": 0, "z": 120, "r": 90},
          "marker_pose_in_camera": {
            "rvec": [3.14159, 0, 0],
            "t_mm": [0, 0, 348]
          }
        }
      ]
    }

The marker pose in the robot base frame must be independently measured.  This
constraint makes the translation-only MG400 model identifiable without
pretending that its R axis provides full six-axis hand-eye observability.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import yaml

try:
    from .eye_in_hand_xyz import estimate_eye_in_hand_xyz
except ImportError:  # Direct execution: python calibration/calibrate_extrinsics.py
    from eye_in_hand_xyz import estimate_eye_in_hand_xyz


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "camera_config.yaml")


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    marker = data.get("marker_pose_in_base", {})
    if "translation_mm" not in marker or "R" not in marker:
        raise ValueError(
            "dataset.marker_pose_in_base requires translation_mm and R"
        )
    return data


def build_extrinsics(calibration, sample_count):
    transform = calibration.t_end_to_camera
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mount_mode": "eye_in_hand_xyz",
        "status": "calibrated",
        "method": "known-marker constrained XYZ eye-in-hand",
        "transform_convention": (
            "T_end_to_camera is the camera pose expressed in the end frame "
            "and maps camera coordinates into end coordinates"
        ),
        "robot_axes_used": ["x", "y", "z"],
        "robot_axes_ignored": ["r"],
        "camera_orientation_source": "fixed_calibration",
        "num_samples": int(sample_count),
        "T_end_to_camera": {
            "R": transform[:3, :3].tolist(),
            "t_mm": transform[:3, 3].tolist(),
        },
        "quality": {
            "mean_marker_residual_mm": calibration.mean_residual_mm,
            "max_marker_residual_mm": calibration.max_residual_mm,
            "marker_residuals_mm": list(calibration.marker_residuals_mm),
            "robot_xyz_span_mm": list(calibration.robot_xyz_span_mm),
        },
    }


def write_config(config_path, extrinsics, target_plane_z_mm=None):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    legacy = config.get("extrinsics")
    if legacy and "T_base_to_cam" in legacy:
        config["legacy_extrinsics_do_not_use"] = legacy
    config["extrinsics"] = extrinsics
    if target_plane_z_mm is not None:
        previous_plane = config.get("table_homography")
        if previous_plane and "H" in previous_plane:
            config["legacy_table_homography_do_not_use"] = previous_plane
        config["table_homography"] = {
            "table_z_mm": float(target_plane_z_mm),
            "source": "eye_in_hand_target_plane_in_robot_base",
            "note": "Only the base-frame plane Z is used; no fixed-camera H is reused.",
        }


    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Calibrate the XYZ-only eye-in-hand camera without moving hardware"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="JSON observations with known marker pose and robot XYZ per frame",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="camera_config.yaml to update",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the calibrated result; without this flag the command is dry-run",
    )
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--minimum-axis-span-mm", type=float, default=10.0)
    parser.add_argument(
        "--target-plane-z-mm",
        type=float,
        default=None,
        help="PCB/table plane Z in robot-base millimetres; overrides dataset target_plane_z_mm",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset = load_dataset(args.dataset)
    marker = dataset["marker_pose_in_base"]
    samples = dataset.get("samples", [])
    calibration = estimate_eye_in_hand_xyz(
        samples,
        marker["translation_mm"],
        marker["R"],
        minimum_samples=args.minimum_samples,
        minimum_axis_span_mm=args.minimum_axis_span_mm,
    )
    extrinsics = build_extrinsics(calibration, len(samples))

    print("XYZ-only eye-in-hand calibration")
    target_plane_z_mm = args.target_plane_z_mm if args.target_plane_z_mm is not None else dataset.get("target_plane_z_mm")
    print("  robot axes used: X, Y, Z")
    print("  robot axes ignored: R")
    print(
        "  camera offset in end frame (mm): "
        + ", ".join(
            f"{value:.4f}" for value in calibration.t_end_to_camera[:3, 3]
        )
    )
    print(f"  mean marker residual: {calibration.mean_residual_mm:.4f} mm")
    print(f"  max marker residual: {calibration.max_residual_mm:.4f} mm")

    if args.write:
        if target_plane_z_mm is None:
            raise ValueError("target_plane_z_mm is required with --write for safe ray/plane projection")
        write_config(args.config, extrinsics, target_plane_z_mm)
        print(f"  target PCB/table plane Z: {float(target_plane_z_mm):.4f} mm")
        print(f"  wrote: {args.config}")
    else:
        print("  dry-run only; pass --write after reviewing the quality metrics")
    return extrinsics


if __name__ == "__main__":
    main()
