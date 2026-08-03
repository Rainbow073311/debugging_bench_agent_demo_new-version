"""Calibrate Basler intrinsics from the active session without overwriting production config."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from capture_basler_intrinsics import ACTIVE_SESSION_FILE, _detect


def _calibrate(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[float]]:
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view = []
    for obj, observed, rotation, translation in zip(
        object_points, image_points, rotations, translations
    ):
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        residual = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
    return rms, matrix, distortion, rotations, translations, per_view


def main() -> int:
    session_dir = Path(ACTIVE_SESSION_FILE.read_text(encoding="utf-8-sig").strip())
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8-sig"))
    validation = json.loads(
        (session_dir / "pending_validation.json").read_text(encoding="utf-8-sig")
    )
    pattern = tuple(int(value) for value in session["pattern_inner_corners"])
    square_size = float(session["square_size_mm"])

    sources: list[tuple[str, Path]] = [
        (image["file"], session_dir / image["file"])
        for image in session.get("images", [])
    ]
    sources.extend(
        (f"pending/{name}", session_dir / "pending" / name)
        for name in validation["selected_files"]
    )

    template = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    template[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    template *= square_size

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_names: list[str] = []
    detection_failures: list[str] = []
    image_size = None
    for name, path in sources:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            detection_failures.append(name)
            continue
        current_size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            raise RuntimeError(f"Mixed image sizes: {name} has {current_size}, expected {image_size}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners, _ = _detect(gray, pattern)
        if not found or corners is None:
            detection_failures.append(name)
            continue
        object_points.append(template.copy())
        image_points.append(corners.reshape(-1, 1, 2).astype(np.float32))
        used_names.append(name)

    if image_size is None or len(used_names) < 15:
        raise RuntimeError(f"Only {len(used_names)} usable images; need at least 15")

    initial = _calibrate(object_points, image_points, image_size)
    initial_errors = np.asarray(initial[-1], dtype=float)
    median = float(np.median(initial_errors))
    mad = float(np.median(np.abs(initial_errors - median)))
    threshold = max(1.0, median + 3.0 * 1.4826 * mad)
    keep_indices = [index for index, error in enumerate(initial_errors) if error <= threshold]
    outlier_indices = [index for index, error in enumerate(initial_errors) if error > threshold]

    if outlier_indices and len(keep_indices) >= 15:
        final = _calibrate(
            [object_points[index] for index in keep_indices],
            [image_points[index] for index in keep_indices],
            image_size,
        )
        final_names = [used_names[index] for index in keep_indices]
    else:
        final = initial
        final_names = used_names
        outlier_indices = []

    rms, matrix, distortion, _, _, per_view_errors = final
    per_image = [
        {"file": name, "rmse_px": float(error)}
        for name, error in zip(final_names, per_view_errors)
    ]
    outliers = [
        {"file": used_names[index], "initial_rmse_px": float(initial_errors[index])}
        for index in outlier_indices
    ]

    candidate = {
        "calibration": {
            "date": datetime.now().astimezone().isoformat(),
            "status": "candidate_not_promoted",
            "camera_model": session["camera_model"],
            "serial": str(session["serial"]),
            "resolution": [int(image_size[0]), int(image_size[1])],
            "roi": session["roi"],
            "pixel_format": session["pixel_format"],
            "exposure_us": session["exposure_us"],
            "gain": session["gain"],
            "chessboard": {
                "pattern": f"{pattern[0]}x{pattern[1]}",
                "square_size_mm": square_size,
                "images_used": len(final_names),
            },
            "session": str(Path("calibration_sessions") / session_dir.name),
        },
        "intrinsics": {
            "camera_matrix": matrix.tolist(),
            "dist_coeffs": distortion.tolist(),
        },
        "quality": {
            "rms_error_px": float(rms),
            "mean_per_view_rmse_px": float(np.mean(per_view_errors)),
            "median_per_view_rmse_px": float(np.median(per_view_errors)),
            "max_per_view_rmse_px": float(np.max(per_view_errors)),
            "outlier_threshold_px": float(threshold),
            "outliers_excluded": outliers,
            "detection_failures": detection_failures,
            "per_image_errors_px": per_image,
        },
        "extrinsics": {
            "mount_mode": "eye_in_hand_xyz",
            "status": "uncalibrated",
            "robot_axes_used": ["x", "y", "z"],
            "robot_axes_ignored": ["r"],
            "camera_orientation_source": "pending_fixed_calibration",
            "note": "Run XYZ eye-in-hand calibration before pixel-to-robot projection.",
        },
    }
    yaml_path = session_dir / "camera_config_candidate.yaml"
    yaml_path.write_text(
        yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "images_used": len(final_names),
        "detection_failures": detection_failures,
        "outliers_excluded": outliers,
        "rms_error_px": float(rms),
        "mean_per_view_rmse_px": float(np.mean(per_view_errors)),
        "median_per_view_rmse_px": float(np.median(per_view_errors)),
        "max_per_view_rmse_px": float(np.max(per_view_errors)),
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
        "dist_coeffs": distortion.reshape(-1).tolist(),
        "candidate_config": str(yaml_path),
    }
    (session_dir / "intrinsics_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
