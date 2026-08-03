"""Verify the Basler intrinsic candidate on qualified images excluded from fitting."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from capture_basler_intrinsics import ACTIVE_SESSION_FILE, _detect


def main() -> int:
    session_dir = Path(ACTIVE_SESSION_FILE.read_text(encoding="utf-8-sig").strip())
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8-sig"))
    validation = json.loads(
        (session_dir / "pending_validation.json").read_text(encoding="utf-8-sig")
    )
    candidate = yaml.safe_load(
        (session_dir / "camera_config_candidate.yaml").read_text(encoding="utf-8-sig")
    )
    matrix = np.asarray(candidate["intrinsics"]["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(candidate["intrinsics"]["dist_coeffs"], dtype=np.float64)
    pattern = tuple(int(value) for value in session["pattern_inner_corners"])

    object_points = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    object_points *= float(session["square_size_mm"])

    holdouts = [
        item["file"]
        for item in validation["records"]
        if item.get("qualified") and not item.get("selected_for_calibration")
    ]
    results = []
    failures = []
    for name in holdouts:
        frame = cv2.imread(str(session_dir / "pending" / name), cv2.IMREAD_COLOR)
        if frame is None:
            failures.append({"file": name, "reason": "unreadable"})
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners, detector = _detect(gray, pattern)
        if not found or corners is None:
            failures.append({"file": name, "reason": "corners not detected"})
            continue
        ok, rotation, translation = cv2.solvePnP(
            object_points,
            corners.reshape(-1, 1, 2).astype(np.float32),
            matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            failures.append({"file": name, "reason": "solvePnP failed"})
            continue
        projected, _ = cv2.projectPoints(
            object_points, rotation, translation, matrix, distortion
        )
        residual = corners.reshape(-1, 2) - projected.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        results.append({"file": name, "rmse_px": rmse, "detector": detector})

    errors = np.asarray([item["rmse_px"] for item in results], dtype=float)
    if not len(errors):
        raise RuntimeError("No independent holdout image could be verified")
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    resolution = candidate["calibration"]["resolution"]
    sanity = {
        "fx_fy_relative_difference": abs(fx - fy) / ((fx + fy) / 2.0),
        "principal_point_inside_image": bool(
            0 <= matrix[0, 2] < resolution[0] and 0 <= matrix[1, 2] < resolution[1]
        ),
        "training_rms_below_1px": candidate["quality"]["rms_error_px"] < 1.0,
        "holdout_mean_below_1px": float(errors.mean()) < 1.0,
        "holdout_max_below_1_5px": float(errors.max()) < 1.5,
    }
    report = {
        "holdout_images": len(results),
        "failures": failures,
        "mean_rmse_px": float(errors.mean()),
        "median_rmse_px": float(np.median(errors)),
        "max_rmse_px": float(errors.max()),
        "sanity": sanity,
        "passed": all(sanity.values()),
        "per_image": results,
    }
    path = session_dir / "intrinsics_holdout_verification.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
