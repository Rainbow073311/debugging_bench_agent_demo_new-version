"""Hover tip onto projected TL using current J1-model camera_config (no refit)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from calibrate_camera_j1_offset import physical_tl_open_cv_index  # noqa: E402
from capture_basler_intrinsics import _capture, _open_camera  # noqa: E402
from capture_extrinsics import _bridge, _config, clamp_xy, detect_chessboard, move  # noqa: E402
from coordinate_transforms import PixelToWorld  # noqa: E402
from tip_offset import tcp_xy_from_tip_config  # noqa: E402

CFG = HERE / "camera_config.yaml"
STATE = HERE / "_extrinsic_recal.json"


def main() -> int:
    import json

    state = json.loads(STATE.read_text(encoding="utf-8"))
    p_tl = np.array(state["P_TL"][:3], float)
    cam_cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    bridge, config = _bridge(), _config(12)
    pose = move(bridge, config, {"x": 320.0, "y": 20.0, "z": 50.0, "r": 7.687})
    time.sleep(0.4)
    cam = _open_camera(cam_cfg["calibration"])
    try:
        for _ in range(3):
            _capture(cam)
        time.sleep(0.12)
        frame = _capture(cam)
    finally:
        cam.Close()
    corners = detect_chessboard(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tl_idx = physical_tl_open_cv_index(
        gray, corners, prefer_xy_mm=p_tl[:2], robot_pose=pose, config_path=CFG
    )
    ptw = PixelToWorld(str(CFG), robot_pose=pose)
    px = corners.reshape(-1, 2)[tl_idx]
    table = ptw.pixel_to_table(float(px[0]), float(px[1]), pose)
    tip_xy = (float(table[0]), float(table[1]))
    err = float(np.linalg.norm(np.array(tip_xy) - p_tl[:2]))
    solved = tcp_xy_from_tip_config(tip_xy)
    tcp_x, tcp_y = clamp_xy(float(solved["tcp_xy"][0]), float(solved["tcp_xy"][1]))
    target = {"x": round(tcp_x, 2), "y": round(tcp_y, 2), "z": -145.0, "r": 7.687}
    print(
        {
            "model": cam_cfg.get("extrinsics", {}).get("j1_model"),
            "t_mm": cam_cfg["extrinsics"]["T_end_to_camera"]["t_mm"],
            "err_vs_P_TL_mm": round(err, 3),
            "tip_xy": [round(tip_xy[0], 3), round(tip_xy[1], 3)],
            "P_TL": [round(float(v), 3) for v in p_tl[:2]],
            "target": target,
        },
        flush=True,
    )
    pose1 = move(bridge, config, target)
    print(
        {
            "event": "hover",
            "pose": pose1,
            "note": "J1-model tip hover. Check tip on Camera-title LEFT. yes=lock, no=restore",
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
