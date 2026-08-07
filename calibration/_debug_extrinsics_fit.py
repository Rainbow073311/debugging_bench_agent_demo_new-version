import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eye_in_hand_xyz import estimate_eye_in_hand_xyz, rotation_from_sample
from step2_calibrate import marker_from_probes

HERE = Path(__file__).resolve().parent
DS = HERE / "calibration_sessions" / "extrinsics_20260807_110142" / "dataset.json"
PROBES = HERE / "_probe_points.json"


def main() -> None:
    ds = json.loads(DS.read_text(encoding="utf-8"))
    probes = json.loads(PROBES.read_text(encoding="utf-8"))
    print("=== sample tvecs ===")
    xs, txs, ys, tys = [], [], [], []
    for s in ds["samples"]:
        t = np.array(s["marker_in_camera"]["t_mm"], float)
        r = np.array(s["marker_in_camera"]["rvec"], float)
        print(
            f"i={s['index']:02d} robot=({s['robot_pose']['x']:.1f},"
            f"{s['robot_pose']['y']:.1f},{s['robot_pose']['z']:.1f}) "
            f"t=({t[0]:6.1f},{t[1]:6.1f},{t[2]:6.1f}) |r|={np.linalg.norm(r):.3f}"
        )
        xs.append(s["robot_pose"]["x"])
        txs.append(t[0])
        ys.append(s["robot_pose"]["y"])
        tys.append(t[1])
    print("corr(robotX,tX)", float(np.corrcoef(xs, txs)[0, 1]))
    print("corr(robotY,tY)", float(np.corrcoef(ys, tys)[0, 1]))
    print("corr(robotX,tY)", float(np.corrcoef(xs, tys)[0, 1]))
    print("corr(robotY,tX)", float(np.corrcoef(ys, txs)[0, 1]))

    # Re-detect images and force a consistent board origin using the probe geometry.
    # Flip board origin (corner 0 <-> corner 53) when it reduces disagreement with robot motion.
    print("\n=== try flipping chessboard origin on each sample ===")
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 5.0
    # For each sample, also consider 180-rotated object points
    samples_norm = []
    samples_flip = []
    for s in ds["samples"]:
        samples_norm.append(
            {"robot_pose": s["robot_pose"], "marker_pose_in_camera": s["marker_in_camera"]}
        )
        R = rotation_from_sample(
            {"marker_pose_in_camera": s["marker_in_camera"]}
        )
        t = np.array(s["marker_in_camera"]["t_mm"], float)
        # 180° about board Z: new_origin = old_origin + R_cam @ (40,25,0)
        # and rvec becomes R * Rz(180)
        offset_board = np.array([40.0, 25.0, 0.0])
        t_flip = t + R @ offset_board
        Rz = np.diag([-1.0, -1.0, 1.0])
        R_flip = R @ Rz
        rvec_flip, _ = cv2.Rodrigues(R_flip)
        samples_flip.append(
            {
                "robot_pose": s["robot_pose"],
                "marker_pose_in_camera": {
                    "rvec": rvec_flip.reshape(-1).tolist(),
                    "t_mm": t_flip.tolist(),
                },
            }
        )

    for name, samples in [("as_captured", samples_norm), ("flip180_all", samples_flip)]:
        for label, cand in {
            "as_is": probes,
            "rot180_probes": {
                "TL": probes["BR"],
                "TR": probes["BL"],
                "BL": probes["TR"],
                "BR": probes["TL"],
            },
            "flip_x": {
                "TL": probes["TR"],
                "TR": probes["TL"],
                "BL": probes["BR"],
                "BR": probes["BL"],
            },
        }.items():
            origin, Rm, _ = marker_from_probes(cand)
            calib = estimate_eye_in_hand_xyz(
                samples, origin, Rm, minimum_samples=8, minimum_axis_span_mm=5.0
            )
            print(
                f"{name:14s} {label:14s} mean={calib.mean_residual_mm:6.2f} "
                f"max={calib.max_residual_mm:6.2f} rot={calib.rotation_crosscheck_error_deg:6.2f} "
                f"t={np.round(calib.t_end_to_camera[:3, 3], 2)}"
            )


if __name__ == "__main__":
    main()
