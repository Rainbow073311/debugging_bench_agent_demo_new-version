"""Fast intrinsic calibration from active session pending images."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ACTIVE = HERE / "calibration_sessions" / ".active_basler_intrinsics_session"


def detect(gray: np.ndarray, pattern: tuple[int, int]):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        # Faster SB without EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            return None
        return corners.reshape(-1, 1, 2).astype(np.float32)
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return refined


def main() -> int:
    session_dir = Path(ACTIVE.read_text(encoding="utf-8-sig").strip())
    files = sorted((session_dir / "pending").glob("pending_*.jpg"))
    print(f"session={session_dir.name} files={len(files)}", flush=True)
    pattern = (9, 6)
    square = 5.0
    template = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    template[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    template *= square

    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    used: list[str] = []
    fails: list[str] = []
    image_size = None

    for i, path in enumerate(files, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            fails.append(path.name)
            print(f"[{i}/{len(files)}] unread {path.name}", flush=True)
            continue
        size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            fails.append(path.name)
            print(f"[{i}/{len(files)}] size mismatch {path.name}", flush=True)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = detect(gray, pattern)
        if corners is None:
            fails.append(path.name)
            print(f"[{i}/{len(files)}] no board {path.name}", flush=True)
            continue
        obj_pts.append(template.copy())
        img_pts.append(corners.astype(np.float32))
        used.append(path.name)
        print(f"[{i}/{len(files)}] OK {path.name}", flush=True)

    print(f"detected={len(used)} failed={len(fails)}", flush=True)
    if len(used) < 10:
        print("ERROR: need at least 10 detections", flush=True)
        return 1

    print("calibrating...", flush=True)
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, image_size, None, None
    )
    per_view = []
    for obj, observed, rotation, translation in zip(obj_pts, img_pts, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        residual = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))

    summary = {
        "rms": float(rms),
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
        "camera_matrix": matrix.tolist(),
        "dist_coeffs": distortion.tolist(),
        "num_images": len(used),
        "image_size": list(image_size),
        "mean_per_view_rmse_px": float(np.mean(per_view)),
        "max_per_view_rmse_px": float(np.max(per_view)),
        "square_size_mm": square,
        "pattern": "9x6",
        "used_files": used,
        "failed_files": fails,
    }
    out = session_dir / "intrinsics.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "rms",
                    "fx",
                    "fy",
                    "cx",
                    "cy",
                    "num_images",
                    "mean_per_view_rmse_px",
                    "max_per_view_rmse_px",
                    "dist_coeffs",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
