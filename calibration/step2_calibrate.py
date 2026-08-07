"""Step 2: Compute eye-in-hand extrinsics from dataset + probe measurements.

Usage:
  python step2_calibrate.py <dataset.json> <probe_points.json>

probe_points.json must contain 4 inner-corner probe readings:
  {"TL":[x,y,z], "TR":[x,y,z], "BL":[x,y,z], "BR":[x,y,z]}

Board axes are fit in 3D (not forced into XY), so a slightly tilted board is OK.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eye_in_hand_xyz import estimate_eye_in_hand_xyz


def marker_from_probes(probes: dict):
    """Compute marker origin + rotation in robot base from 4 probed corners."""
    tl = np.array(probes["TL"], dtype=np.float64)
    tr = np.array(probes["TR"], dtype=np.float64)
    bl = np.array(probes["BL"], dtype=np.float64)
    br = np.array(probes["BR"], dtype=np.float64)

    # Prefer edge midpoints so a single noisy corner doesn't skew the axes.
    vx = 0.5 * ((tr - tl) + (br - bl))
    vy = 0.5 * ((bl - tl) + (br - tr))
    x_span = float(np.linalg.norm(vx))
    y_span = float(np.linalg.norm(vy))
    print(f"Board X span: {x_span:.2f} mm (expect ~40 for 9x6 / 5mm)")
    print(f"Board Y span: {y_span:.2f} mm (expect ~25 for 9x6 / 5mm)")
    if abs(x_span - 40.0) > 3.0 or abs(y_span - 25.0) > 3.0:
        print("WARNING: probe spans disagree with 5 mm / 9x6 board; check corner order")

    # Plane normal from the quadrilateral (handles mild board tilt).
    n1 = np.cross(tr - tl, bl - tl)
    n2 = np.cross(bl - br, tr - br)
    if np.linalg.norm(n1) < 1e-9 or np.linalg.norm(n2) < 1e-9:
        raise ValueError("degenerate probe corners")
    ez = n1 / np.linalg.norm(n1) + n2 / np.linalg.norm(n2)
    ez /= np.linalg.norm(ez)
    # Camera looks roughly -Z in base for this setup; keep board +Z upward.
    if ez[2] < 0:
        ez = -ez

    ex = vx - np.dot(vx, ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)

    R = np.column_stack((ex, ey, ez))
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt

    # Diagonal midpoint is more stable than TL alone for the board origin.
    origin = 0.25 * (tl + tr + bl + br)
    # OpenCV chessboard origin is the first inner corner (TL in our convention),
    # not the board center. Shift back along -X/-Y by 0 squares: probes are the
    # four outer inner-corners, so origin = TL.
    origin = tl.copy()
    plane_tilt_deg = float(np.degrees(np.arccos(np.clip(abs(ez[2]), 0.0, 1.0))))
    z_spread = float(np.ptp([tl[2], tr[2], bl[2], br[2]]))
    print(f"Board plane tilt vs horizontal: {plane_tilt_deg:.2f} deg")
    print(f"Probe Z spread: {z_spread:.2f} mm")
    return origin, R, {
        "x_span_mm": x_span,
        "y_span_mm": y_span,
        "plane_tilt_deg": plane_tilt_deg,
        "probe_z_spread_mm": z_spread,
    }


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python step2_calibrate.py <dataset.json> <probe_points.json>")
        return 1

    ds_path = Path(sys.argv[1])
    probe_path = Path(sys.argv[2])
    ds = json.loads(ds_path.read_text(encoding="utf-8"))
    probes = json.loads(probe_path.read_text(encoding="utf-8"))

    marker_origin, marker_R, board_quality = marker_from_probes(probes)
    print(f"Marker origin in base: {[round(float(v), 2) for v in marker_origin]}")
    print(f"det(R) = {np.linalg.det(marker_R):.4f}")

    samples = ds.get("samples", [])
    if len(samples) < 12:
        print(f"WARNING: only {len(samples)} samples; prefer >= 18 from widened capture")

    poses = np.array([[s["robot_pose"][k] for k in ("x", "y", "z")] for s in samples], float)
    spans = np.ptp(poses, axis=0) if len(samples) else np.zeros(3)
    print(
        f"Robot XYZ spans: X={spans[0]:.1f} Y={spans[1]:.1f} Z={spans[2]:.1f} mm"
    )
    if spans[0] < 20 or spans[1] < 20 or spans[2] < 20:
        print(
            "WARNING: translation span is small; rerun capture_extrinsics.py "
            "with larger --xy-span/--z-span for a stronger fit"
        )

    calib_samples = [
        {"robot_pose": s["robot_pose"], "marker_pose_in_camera": s["marker_in_camera"]}
        for s in samples
    ]

    # Drop OpenCV chessboard-origin flips (|rvec|~0 instead of ~pi).
    filtered = []
    dropped_flip = 0
    for sample in calib_samples:
        if np.linalg.norm(sample["marker_pose_in_camera"]["rvec"]) < 1.0:
            dropped_flip += 1
            continue
        filtered.append(sample)
    if len(filtered) >= 12:
        if dropped_flip:
            print(f"Dropped {dropped_flip} samples with flipped chessboard origin")
        calib_samples = filtered

    def _fit(rotation):
        return estimate_eye_in_hand_xyz(
            calib_samples,
            marker_origin,
            rotation,
            minimum_samples=8,
            minimum_axis_span_mm=5.0,
        )

    calib = _fit(marker_R)
    # OpenCV board +Z faces the camera; if our +Z was chosen "up" but PnP has the
    # opposite sense, Kabsch and orientation disagree by ~180°. Flip Y/Z once.
    if calib.rotation_crosscheck_error_deg > 90.0 or calib.mean_residual_mm > 5.0:
        flipped = marker_R.copy()
        flipped[:, 1] *= -1.0
        flipped[:, 2] *= -1.0
        calib_flip = _fit(flipped)
        print(
            f"Orientation repair: residual {calib.mean_residual_mm:.2f}->{calib_flip.mean_residual_mm:.2f} mm, "
            f"rot {calib.rotation_crosscheck_error_deg:.1f}->{calib_flip.rotation_crosscheck_error_deg:.1f} deg"
        )
        if (
            calib_flip.mean_residual_mm < calib.mean_residual_mm
            or calib_flip.rotation_crosscheck_error_deg < calib.rotation_crosscheck_error_deg
        ):
            calib = calib_flip
            marker_R = flipped
            board_quality["orientation_repaired"] = True

    print("\n=== Hand-Eye Calibration Result ===")
    print(f"Mean residual:   {calib.mean_residual_mm:.3f} mm")
    print(f"Max residual:    {calib.max_residual_mm:.3f} mm")
    print(f"Rotation xcheck: {calib.rotation_crosscheck_error_deg:.2f} deg")
    t = calib.t_end_to_camera[:3, 3]
    print(f"T_end_to_camera t_mm: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")

    # Holdout: fit on first N-3, score last 3 if enough samples.
    holdout = None
    if len(calib_samples) >= 15:
        fit = estimate_eye_in_hand_xyz(
            calib_samples[:-3],
            marker_origin,
            marker_R,
            minimum_samples=8,
            minimum_axis_span_mm=5.0,
        )
        R = fit.t_end_to_camera[:3, :3]
        te = fit.t_end_to_camera[:3, 3]
        errs = []
        for s in calib_samples[-3:]:
            rp = np.array([s["robot_pose"][k] for k in ("x", "y", "z")], float)
            tc = np.array(s["marker_pose_in_camera"]["t_mm"], float)
            pred = rp + te + R @ tc
            d = pred - marker_origin
            errs.append(
                {
                    "xyz_mm": float(np.linalg.norm(d)),
                    "xy_mm": float(np.linalg.norm(d[:2])),
                }
            )
        holdout = {
            "mean_xyz_mm": float(np.mean([e["xyz_mm"] for e in errs])),
            "mean_xy_mm": float(np.mean([e["xy_mm"] for e in errs])),
            "max_xy_mm": float(np.max([e["xy_mm"] for e in errs])),
            "samples": errs,
        }
        print(
            f"Holdout (last 3): XY mean={holdout['mean_xy_mm']:.3f} "
            f"max={holdout['max_xy_mm']:.3f} mm"
        )

    if calib.mean_residual_mm >= 3.0:
        print(f"\nFAIL: residual {calib.mean_residual_mm:.1f} > 3.0 mm")
        return 1
    if calib.rotation_crosscheck_error_deg >= 5.0:
        print(f"\nFAIL: crosscheck {calib.rotation_crosscheck_error_deg:.1f} > 5.0 deg")
        return 1
    if holdout and holdout["max_xy_mm"] >= 3.0:
        print(f"\nFAIL: holdout XY max {holdout['max_xy_mm']:.1f} > 3.0 mm")
        return 1

    cfg_path = HERE / "camera_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["extrinsics"] = {
        "status": "calibrated",
        "mount_mode": "eye_in_hand_xyz",
        "date": f"recalibrated-{ds.get('fixed_r_deg', 7.687)}",
        "method": "known-board PnP + XYZ-constrained hand-eye (wide lattice)",
        "num_samples": len(calib_samples),
        "dataset": str(ds_path),
        "T_end_to_camera": {
            "R": calib.t_end_to_camera[:3, :3].tolist(),
            "t_mm": calib.t_end_to_camera[:3, 3].tolist(),
        },
        "robot_axes_used": ["x", "y", "z"],
        "robot_axes_ignored": ["r"],
        "transform_convention": "T_end_to_camera maps camera coords into end-effector coords",
        "camera_orientation_source": "fixed_calibration",
        "quality": {
            "mean_marker_residual_mm": calib.mean_residual_mm,
            "max_marker_residual_mm": calib.max_residual_mm,
            "rotation_crosscheck_error_deg": calib.rotation_crosscheck_error_deg,
            "robot_xyz_span_mm": [float(v) for v in spans],
            "board_probe": board_quality,
            "holdout": holdout,
        },
    }
    cfg.setdefault("table_homography", {})
    # Use mean probe Z as the working plane (board may be mildly tilted).
    mean_z = float(np.mean([probes[k][2] for k in ("TL", "TR", "BL", "BR")]))
    cfg["table_homography"]["table_z_mm"] = mean_z
    cfg["runtime_safety"] = {
        "fixed_r_deg": ds.get("fixed_r_deg", 7.687),
        "safe_height_z_mm": 50.0,
        "close_capture_z_mm": mean_z + 112.0,
        "global_capture_z_mm": 50.0,
        "valid_robot_xyz_min": [float(poses[:, i].min()) for i in range(3)],
        "valid_robot_xyz_max": [float(poses[:, i].max()) for i in range(3)],
    }

    bak = cfg_path.with_suffix(".yaml.bak2")
    shutil.copy(cfg_path, bak)
    cfg_path.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    summary = {
        "mean_residual_mm": calib.mean_residual_mm,
        "max_residual_mm": calib.max_residual_mm,
        "rotation_crosscheck_error_deg": calib.rotation_crosscheck_error_deg,
        "holdout": holdout,
        "config": str(cfg_path),
        "backup": str(bak),
    }
    (ds_path.parent / "extrinsics_fit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nDone — written to {cfg_path}")
    print(f"Backup: {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
