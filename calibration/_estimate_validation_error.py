"""Estimate how new vs old intrinsics change post-extrinsics validation residuals."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from eye_in_hand_xyz import estimate_eye_in_hand_xyz


HERE = Path(__file__).resolve().parent
SESSION = HERE / "calibration_sessions" / "extrinsics_20260806_164510"
PATTERN = (9, 6)
SQUARE = 5.0


def detect(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not found:
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None
        return corners.reshape(-1, 1, 2).astype(np.float32)
    return cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )


def pnp(corners, K, D):
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQUARE
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def main():
    old_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    new = json.loads(
        (HERE / "calibration_sessions" / "intrinsics_20260806_175412" / "intrinsics.json").read_text(
            encoding="utf-8"
        )
    )
    Ko = np.array(old_cfg["intrinsics"]["camera_matrix"], float)
    Do = np.array(old_cfg["intrinsics"]["dist_coeffs"], float).reshape(-1, 1)
    Kn = np.array(new["camera_matrix"], float)
    Dn = np.array(new["dist_coeffs"], float).reshape(-1, 1)

    dataset = json.loads((SESSION / "dataset.json").read_text(encoding="utf-8"))
    samples_old = []
    samples_new = []
    deltas = []

    for sample in dataset["samples"]:
        path = SESSION / sample["image"]
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = detect(gray)
        if corners is None:
            print("skip no board", sample["image"])
            continue
        po = pnp(corners, Ko, Do)
        pn = pnp(corners, Kn, Dn)
        if po is None or pn is None:
            continue
        Ro, to = po
        Rn, tn = pn
        d = tn - to
        deltas.append(d)
        pose = sample["robot_pose"]
        samples_old.append(
            {
                "robot_pose": pose,
                "marker_pose_in_camera": {
                    "t_mm": to.tolist(),
                    "R": Ro.tolist(),
                },
            }
        )
        samples_new.append(
            {
                "robot_pose": pose,
                "marker_pose_in_camera": {
                    "t_mm": tn.tolist(),
                    "R": Rn.tolist(),
                },
            }
        )
        print(
            f"{sample['image']}: told={to} tnew={tn} d={d} |d|={np.linalg.norm(d):.2f}"
        )

    deltas = np.asarray(deltas)
    print("\nPnP t_cam delta (new-old) mean mm:", deltas.mean(axis=0))
    print("PnP t_cam delta rms mm:", np.sqrt((deltas**2).mean(axis=0)))
    print("PnP |d| mean/max:", float(np.mean(np.linalg.norm(deltas, axis=1))), float(np.max(np.linalg.norm(deltas, axis=1))))

    # Recover a pseudo marker_t_base from OLD extrinsics + first sample (consistent with old model)
    T = np.eye(4)
    R_ec = np.array(old_cfg["extrinsics"]["T_end_to_camera"]["R"], float)
    t_ec = np.array(old_cfg["extrinsics"]["T_end_to_camera"]["t_mm"], float)
    # In this codebase: predicted_marker = robot + t_end_to_cam + R_cam_base @ t_cam
    # Wait - eye_in_hand stores T_end_to_camera with camera_r_base and camera_t_end.
    # predicted = robot_position + camera_t_end + camera_r_base @ marker_t_camera
    # So marker_t_base ≈ that prediction; use mean over old samples as "truth"
    marker_estimates = []
    for s in samples_old:
        rp = np.array([s["robot_pose"]["x"], s["robot_pose"]["y"], s["robot_pose"]["z"]], float)
        tc = np.array(s["marker_pose_in_camera"]["t_mm"], float)
        marker_estimates.append(rp + t_ec + R_ec @ tc)
    marker_t_base = np.mean(marker_estimates, axis=0)
    # Use identity for rotation of marker in base (plane parallel approx); orientation mainly for R fit
    marker_r_base = np.eye(3)

    print("\nPseudo marker_t_base from old model mean:", marker_t_base)

    # Fit with old PnP (should be small residual vs this pseudo truth)
    # Need marker_r_camera - use R from PnP; marker_r_base = eye may be wrong for rotation crosscheck
    # Use average R from PnP to set marker_r_base = mean(R_ec @ R_cam) roughly
    # camera_r_base ≈ marker_r_base @ marker_r_camera.T  => marker_r_base ≈ camera_r_base @ marker_r_camera
    marker_r_base = R_ec @ np.array(samples_old[0]["marker_pose_in_camera"]["R"], float)

    fit_old = estimate_eye_in_hand_xyz(
        samples_old,
        marker_translation_in_base=marker_t_base,
        marker_rotation_in_base=marker_r_base,
        minimum_samples=8,
        minimum_axis_span_mm=5.0,
    )
    fit_new = estimate_eye_in_hand_xyz(
        samples_new,
        marker_translation_in_base=marker_t_base,
        marker_rotation_in_base=marker_r_base,
        minimum_samples=8,
        minimum_axis_span_mm=5.0,
    )
    print("\n--- Fit residuals vs same marker_t_base (train set) ---")
    print(f"OLD K fit: mean={fit_old.mean_residual_mm:.3f} max={fit_old.max_residual_mm:.3f} mm")
    print(f"NEW K fit: mean={fit_new.mean_residual_mm:.3f} max={fit_new.max_residual_mm:.3f} mm")

    # Holdout: use last 3 samples as validation with each fitted model
    def holdout(fit, samples, label):
        R = fit.t_end_to_camera[:3, :3]
        t = fit.t_end_to_camera[:3, 3]
        errs = []
        for s in samples[-3:]:
            rp = np.array([s["robot_pose"]["x"], s["robot_pose"]["y"], s["robot_pose"]["z"]], float)
            tc = np.array(s["marker_pose_in_camera"]["t_mm"], float)
            pred = rp + t + R @ tc
            err = float(np.linalg.norm(pred - marker_t_base))
            xy = float(np.linalg.norm((pred - marker_t_base)[:2]))
            errs.append((err, xy, pred - marker_t_base))
            print(f"  {label} holdout |xyz|={err:.2f} |xy|={xy:.2f} d={pred-marker_t_base}")
        return errs

    print("\n--- Holdout last 3 (same as validation-style check) ---")
    # Refit on first N-3 only
    fit_old_h = estimate_eye_in_hand_xyz(
        samples_old[:-3],
        marker_translation_in_base=marker_t_base,
        marker_rotation_in_base=marker_r_base,
        minimum_samples=8,
        minimum_axis_span_mm=5.0,
    )
    fit_new_h = estimate_eye_in_hand_xyz(
        samples_new[:-3],
        marker_translation_in_base=marker_t_base,
        marker_rotation_in_base=marker_r_base,
        minimum_samples=8,
        minimum_axis_span_mm=5.0,
    )
    print(f"OLD K train residual mean={fit_old_h.mean_residual_mm:.3f}")
    print(f"NEW K train residual mean={fit_new_h.mean_residual_mm:.3f}")
    eo = holdout(fit_old_h, samples_old, "OLD")
    en = holdout(fit_new_h, samples_new, "NEW")
    print(
        "\nHoldout XY mean OLD/NEW:",
        float(np.mean([e[1] for e in eo])),
        float(np.mean([e[1] for e in en])),
    )

    # Also: keep OLD extrinsics, only swap K at verify time (worst case - no re-fit)
    print("\n--- Worst case: keep OLD extrinsics, only swap to NEW K (no re-fit) ---")
    for s_old, s_new in zip(samples_old[-3:], samples_new[-3:]):
        rp = np.array([s_old["robot_pose"]["x"], s_old["robot_pose"]["y"], s_old["robot_pose"]["z"]], float)
        tc = np.array(s_new["marker_pose_in_camera"]["t_mm"], float)
        pred = rp + t_ec + R_ec @ tc
        d = pred - marker_t_base
        print(f"  |xyz|={np.linalg.norm(d):.2f} |xy|={np.linalg.norm(d[:2]):.2f} d={d}")


if __name__ == "__main__":
    main()
