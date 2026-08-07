"""Final intrinsic calibration over all pending images with robust detection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ACTIVE = HERE / "calibration_sessions" / ".active_basler_intrinsics_session"
PATTERN = (9, 6)
SQUARE = 5.0


def detect(gray: np.ndarray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
    if found:
        refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        return refined.reshape(-1, 2), "classic"
    for label, g in (("sb", gray), ("sb_clahe", cv2.createCLAHE(3.0, (8, 8)).apply(gray))):
        ok, c = cv2.findChessboardCornersSB(g, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            return c.reshape(-1, 2), label
    return None, None


def main() -> int:
    session_dir = Path(ACTIVE.read_text(encoding="utf-8-sig").strip())
    files = sorted(
        f for f in (session_dir / "pending").glob("pending_*.jpg") if "_marked" not in f.name
    )
    print(f"session={session_dir.name} files={len(files)}", flush=True)

    template = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    template[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2)
    template *= SQUARE

    obj_pts, img_pts, used, fails = [], [], [], []
    tilt_ratios = []
    image_size = None
    for i, path in enumerate(files, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            fails.append(path.name)
            continue
        size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = size
        if size != image_size:
            fails.append(path.name)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, det = detect(gray)
        if corners is None:
            fails.append(path.name)
            print(f"[{i}/{len(files)}] FAIL {path.name}", flush=True)
            continue
        g = corners.reshape(PATTERN[1], PATTERN[0], 2)
        top = np.linalg.norm(g[0, -1] - g[0, 0])
        bot = np.linalg.norm(g[-1, -1] - g[-1, 0])
        left = np.linalg.norm(g[-1, 0] - g[0, 0])
        right = np.linalg.norm(g[-1, -1] - g[0, -1])
        tilt_ratios.append(
            max(
                max(top, bot) / max(min(top, bot), 1e-6),
                max(left, right) / max(min(left, right), 1e-6),
            )
        )
        obj_pts.append(template.copy())
        img_pts.append(corners.reshape(-1, 1, 2).astype(np.float32))
        used.append(path.name)
        print(f"[{i}/{len(files)}] OK {path.name} ({det})", flush=True)

    print(f"detected={len(used)} failed={len(fails)}", flush=True)
    if len(used) < 20:
        print("ERROR: too few detections", flush=True)
        return 1

    print("calibrating (pass 1)...", flush=True)
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None)
    per = []
    for obj, obs, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, D)
        res = obs.reshape(-1, 2) - proj.reshape(-1, 2)
        per.append(float(np.sqrt(np.mean(np.sum(res * res, axis=1)))))
    per = np.asarray(per)

    # Drop outlier views and refit once.
    median = float(np.median(per))
    mad = float(np.median(np.abs(per - median)))
    threshold = max(1.0, median + 3.0 * 1.4826 * mad)
    keep = [i for i, e in enumerate(per) if e <= threshold]
    dropped = [(used[i], float(per[i])) for i, e in enumerate(per) if per[i] > threshold]
    if dropped and len(keep) >= 20:
        print(f"refitting without {len(dropped)} outliers (thr={threshold:.2f}px)...", flush=True)
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            [obj_pts[i] for i in keep], [img_pts[i] for i in keep], image_size, None, None
        )
        per2 = []
        for i, (rv, tv) in enumerate(zip(rvecs, tvecs)):
            proj, _ = cv2.projectPoints(obj_pts[keep[i]], rv, tv, K, D)
            res = img_pts[keep[i]].reshape(-1, 2) - proj.reshape(-1, 2)
            per2.append(float(np.sqrt(np.mean(np.sum(res * res, axis=1)))))
        per = np.asarray(per2)
        used = [used[i] for i in keep]

    summary = {
        "rms": float(rms),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "camera_matrix": K.tolist(),
        "dist_coeffs": D.tolist(),
        "num_images": len(used),
        "image_size": list(image_size),
        "mean_per_view_rmse_px": float(per.mean()),
        "median_per_view_rmse_px": float(np.median(per)),
        "max_per_view_rmse_px": float(per.max()),
        "outliers_dropped": dropped,
        "detection_failures": fails,
        "square_size_mm": SQUARE,
        "pattern": "9x6",
        "tilt_ratio_max": float(np.max(tilt_ratios)),
        "used_files": used,
    }
    out = session_dir / "intrinsics.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "rms", "fx", "fy", "cx", "cy", "num_images",
                    "mean_per_view_rmse_px", "max_per_view_rmse_px",
                    "outliers_dropped", "tilt_ratio_max",
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
