"""Step 2: Compute eye-in-hand extrinsics from dataset + probe measurements.

Usage:
  python step2_calibrate.py <dataset.json> <probe_points.json>

probe_points.json must contain 4 inner-corner probe readings:
  {"TL":[x,y,z], "TR":[x,y,z], "BL":[x,y,z], "BR":[x,y,z]}
"""
import json, yaml, numpy as np, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eye_in_hand_xyz import estimate_eye_in_hand_xyz


def marker_from_probes(probes):
    """Compute marker origin + rotation in robot base from 4 probed corners."""
    tl = np.array(probes["TL"], dtype=np.float64)
    tr = np.array(probes["TR"], dtype=np.float64)
    bl = np.array(probes["BL"], dtype=np.float64)
    br = np.array(probes["BR"], dtype=np.float64)

    vx = tr - tl; vx[2] = 0  # board X, XY-plane only
    vy = bl - tl; vy[2] = 0  # board Y, XY-plane only

    print(f"Board X span: {np.linalg.norm(vx):.1f} mm (expect ~40)")
    print(f"Board Y span: {np.linalg.norm(vy):.1f} mm (expect ~25)")

    ex = vx / np.linalg.norm(vx)
    ey = vy - np.dot(vy, ex) * ex; ey /= np.linalg.norm(ey)
    ez = np.array([0., 0., 1.])

    R = np.column_stack((ex, ey, ez))
    u, _, vt = np.linalg.svd(R); R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1; R = u @ vt

    origin = tl.copy()
    origin[2] = float(np.mean([tl[2], tr[2], bl[2], br[2]]))
    return origin, R


def main():
    if len(sys.argv) < 3:
        print("Usage: python step2_calibrate.py <dataset.json> <probe_points.json>")
        sys.exit(1)

    ds = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    probes = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    marker_origin, marker_R = marker_from_probes(probes)
    print(f"Marker origin in base: {[f'{v:.2f}' for v in marker_origin]}")
    print(f"det(R) = {np.linalg.det(marker_R):.4f}")

    calib_samples = [
        {"robot_pose": s["robot_pose"], "marker_pose_in_camera": s["marker_in_camera"]}
        for s in ds["samples"]
    ]

    calib = estimate_eye_in_hand_xyz(
        calib_samples, marker_origin, marker_R,
        minimum_samples=8, minimum_axis_span_mm=5.0,
    )

    print(f"\n=== Hand-Eye Calibration Result ===")
    print(f"Mean residual:   {calib.mean_residual_mm:.3f} mm")
    print(f"Max residual:    {calib.max_residual_mm:.3f} mm")
    print(f"Rotation xcheck: {calib.rotation_crosscheck_error_deg:.2f} deg")
    t = calib.t_end_to_camera[:3, 3]
    print(f"T_end_to_camera t_mm: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")

    if calib.mean_residual_mm >= 5.0:
        print(f"\nFAIL: residual {calib.mean_residual_mm:.1f} > 5.0mm")
        sys.exit(1)
    if calib.rotation_crosscheck_error_deg >= 5.0:
        print(f"\nFAIL: crosscheck {calib.rotation_crosscheck_error_deg:.1f} > 5.0deg")
        sys.exit(1)

    # Write config
    cfg_path = HERE / "camera_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["extrinsics"] = {
        "status": "calibrated",
        "mount_mode": "eye_in_hand_xyz",
        "date": f"recalibrated-{ds.get('fixed_r_deg', 7.687)}",
        "method": "known-board PnP + XYZ-constrained hand-eye",
        "num_samples": len(calib_samples),
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
        },
    }
    cfg["table_homography"]["table_z_mm"] = float(marker_origin[2])
    cfg["runtime_safety"] = {
        "fixed_r_deg": ds.get("fixed_r_deg", 7.687),
        "safe_height_z_mm": 50.0,
        "close_capture_z_mm": float(marker_origin[2]) + 112.0,
        "global_capture_z_mm": 50.0,
    }

    bak = cfg_path.with_suffix(".yaml.bak2")
    import shutil; shutil.copy(cfg_path, bak)
    cfg_path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
    print(f"\nDone — written to {cfg_path}")
    print(f"Backup: {bak}")


if __name__ == "__main__":
    main()
