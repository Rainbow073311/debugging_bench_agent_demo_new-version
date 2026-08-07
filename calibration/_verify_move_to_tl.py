"""Capture at current pose, project OpenCV TL, move TCP to that XY (hover)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_basler_intrinsics import _capture, _open_camera
from capture_extrinsics import FIXED_R, _bridge, _config, detect_chessboard, get_pose, move
from coordinate_transforms import PixelToWorld
from eye_in_hand_xyz import compose_base_to_camera

PATTERN = (9, 6)
SQUARE = 5.0
HOVER_ABOVE_MM = 10.0


def main() -> int:
    bridge = _bridge()
    config = _config(15)
    pose = get_pose(bridge, config)
    print(json.dumps({"event": "capture_pose", "pose": pose}), flush=True)

    cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    cam = _open_camera(cfg["calibration"])
    try:
        frame = _capture(cam)
    finally:
        cam.Close()
    cv2.imwrite(str(HERE / "_verify_tl_capture.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    corners = detect_chessboard(frame)
    if corners is None:
        print(json.dumps({"event": "no_board"}), flush=True)
        return 2
    corners = corners.reshape(-1, 2)
    tl_px = corners[0]
    print(
        json.dumps(
            {
                "event": "tl_pixel",
                "px": [round(float(tl_px[0]), 1), round(float(tl_px[1]), 1)],
            }
        ),
        flush=True,
    )

    # Method A: pixel -> table via calibrated pipeline
    ptw = PixelToWorld(str(HERE / "camera_config.yaml"))
    ptw.set_robot_pose(pose)
    tl_table = ptw.pixel_to_table(float(tl_px[0]), float(tl_px[1]))

    # Method B: PnP corner 0 -> base via T_base_to_camera
    K = np.array(cfg["intrinsics"]["camera_matrix"], float)
    D = np.array(cfg["intrinsics"]["dist_coeffs"], float)
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQUARE
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1, 1, 2), K, D)
    R_cam, _ = cv2.Rodrigues(rvec)
    T_ec = np.eye(4)
    T_ec[:3, :3] = np.array(cfg["extrinsics"]["T_end_to_camera"]["R"], float)
    T_ec[:3, 3] = np.array(cfg["extrinsics"]["T_end_to_camera"]["t_mm"], float)
    T_bc = compose_base_to_camera(pose, T_ec)
    tl_cam = (R_cam @ np.array([0.0, 0.0, 0.0]) + tvec.reshape(3)).ravel()
    tl_base = (T_bc @ np.array([tl_cam[0], tl_cam[1], tl_cam[2], 1.0]))[:3]

    probed = json.loads((HERE / "_probe_points.json").read_text(encoding="utf-8"))["TL"]
    probed = np.array(probed, float)

    print(
        json.dumps(
            {
                "event": "projection",
                "pixel_to_table": None if tl_table is None else [round(float(v), 3) for v in tl_table],
                "pnp_base": [round(float(v), 3) for v in tl_base],
                "probed_TL": [round(float(v), 3) for v in probed],
                "error_vs_probe_xy_mm": round(float(np.linalg.norm(tl_base[:2] - probed[:2])), 3),
                "error_vs_probe_xyz_mm": round(float(np.linalg.norm(tl_base - probed)), 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    vis = frame.copy()
    cv2.drawChessboardCorners(vis, PATTERN, corners.reshape(-1, 1, 2), True)
    cv2.circle(vis, (int(tl_px[0]), int(tl_px[1])), 22, (0, 0, 255), 3)
    cv2.putText(
        vis,
        "TL",
        (int(tl_px[0]) + 20, int(tl_px[1]) - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 255),
        3,
    )
    cv2.imwrite(str(HERE / "_verify_tl_marked.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Move to projected TL XY, hover above probed/table Z.
    target_xy = tl_base[:2]
    target_z = float(probed[2] + HOVER_ABOVE_MM)
    target = {
        "x": round(float(target_xy[0]), 2),
        "y": round(float(target_xy[1]), 2),
        "z": round(target_z, 2),
        "r": FIXED_R,
    }
    print(json.dumps({"event": "move_to_tl", "target": target}), flush=True)

    def wait_idle(timeout: float = 40.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = bridge.action_status({"config": config})
            code = st["robot"]["mode"]["code"]
            p = st["robot"]["pose"]
            if code == 5:
                return {k: float(p[k]) for k in ("x", "y", "z", "r")}
            time.sleep(0.25)
        raise TimeoutError("robot not ENABLED_IDLE")

    def movj(p: dict) -> dict:
        r = bridge.action_command(
            {"config": config, "command": {"name": "move", "pose": p, "motionCommand": "MovJ"}}
        )
        if not r.get("ok"):
            raise RuntimeError(r)
        return wait_idle()

    # Staged path: inward -> high Z -> XY over TL -> descend (avoids radius limits)
    cur = wait_idle()
    print(json.dumps({"event": "pre_move_pose", "pose": cur}), flush=True)
    mid_xy = {"x": 305.0, "y": -20.0, "z": min(max(cur["z"], -40.0), 45.0), "r": FIXED_R}
    print(json.dumps({"event": "stage", "name": "inward", "pose": mid_xy}), flush=True)
    movj(mid_xy)
    high = {"x": 305.0, "y": -20.0, "z": 45.0, "r": FIXED_R}
    print(json.dumps({"event": "stage", "name": "lift", "pose": high}), flush=True)
    movj(high)
    over = {"x": target["x"], "y": target["y"], "z": 45.0, "r": FIXED_R}
    print(json.dumps({"event": "stage", "name": "over_tl", "pose": over}), flush=True)
    movj(over)
    actual = movj(target)
    err_xy = float(np.linalg.norm(np.array([actual["x"], actual["y"]]) - probed[:2]))
    print(
        json.dumps(
            {
                "event": "arrived",
                "actual": actual,
                "xy_error_vs_probed_TL_mm": round(err_xy, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
