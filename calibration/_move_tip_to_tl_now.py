"""Capture chessboard, find physical TL, move TIP onto it (tip_offset)."""
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

from capture_basler_intrinsics import _capture, _open_camera  # noqa: E402
from capture_extrinsics import (  # noqa: E402
    _bridge,
    _config,
    clamp_xy,
    detect_chessboard,
    get_pose,
    move,
)
from calibrate_camera_j1_offset import OUTER_IDX, physical_tl_open_cv_index  # noqa: E402
from coordinate_transforms import PixelToWorld  # noqa: E402
from tip_offset import j1_from_tcp_xy, load_tip_offset_params, tcp_xy_from_tip  # noqa: E402

OUT = HERE / "_camera_center_50mm_session"
HOVER_ABOVE_TABLE_MM = 8.0  # tip above table plane estimate; keep current Z if safer


def capture(cfg: dict) -> np.ndarray:
    cam = _open_camera(cfg["calibration"])
    try:
        for _ in range(3):
            _capture(cam)
        time.sleep(0.12)
        return _capture(cam)
    finally:
        cam.Close()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tip_p = load_tip_offset_params()
    print(
        json.dumps(
            {
                "event": "tip_params",
                "radius_xy_mm": tip_p["radius_xy_mm"],
                "delta_deg": tip_p["delta_deg"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    bridge, config = _bridge(), _config(15)
    pose0 = get_pose(bridge, config)
    print(json.dumps({"event": "start", "pose": pose0}, ensure_ascii=False), flush=True)

    cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    frame = capture(cfg)
    cv2.imwrite(str(OUT / "_probe_tl_before_raw.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    corners = detect_chessboard(frame)
    if corners is None:
        print(json.dumps({"ok": False, "error": "chessboard not found"}, ensure_ascii=False))
        return 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tl_idx = physical_tl_open_cv_index(gray, corners)
    pts = corners.reshape(-1, 2)
    tl_px = pts[tl_idx]
    print(
        json.dumps(
            {
                "event": "tl_detected",
                "tl_idx": int(tl_idx),
                "tl_px": [round(float(tl_px[0]), 1), round(float(tl_px[1]), 1)],
                "outers": {
                    str(i): [round(float(pts[i, 0]), 1), round(float(pts[i, 1]), 1)]
                    for i in OUTER_IDX
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    vis = frame.copy()
    cv2.drawChessboardCorners(vis, (9, 6), corners, True)
    for i in OUTER_IDX:
        col = (0, 0, 255) if i == tl_idx else (255, 128, 0)
        p = (int(pts[i, 0]), int(pts[i, 1]))
        cv2.circle(vis, p, 18, col, 3)
        cv2.putText(
            vis,
            "TL" if i == tl_idx else str(i),
            (p[0] + 12, p[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            col,
            3,
        )
    cv2.imwrite(str(OUT / "_probe_tl_before_marked.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ptw = PixelToWorld(str(HERE / "camera_config.yaml"))
    ptw.set_robot_pose(pose0)
    tl_table = ptw.pixel_to_table(float(tl_px[0]), float(tl_px[1]), pose0)
    if tl_table is None:
        print(json.dumps({"ok": False, "error": "TL projection failed"}, ensure_ascii=False))
        return 3

    tip_xy = (float(tl_table[0]), float(tl_table[1]))
    tcp = tcp_xy_from_tip(tip_xy)
    tcp_x, tcp_y = clamp_xy(float(tcp[0]), float(tcp[1]))
    # Stay at current Z (already near board); do not dive to table_z
    hover_z = float(pose0["z"])
    target = {
        "x": round(tcp_x, 2),
        "y": round(tcp_y, 2),
        "z": round(hover_z, 2),
        "r": float(pose0["r"]),
    }
    print(
        json.dumps(
            {
                "event": "move_plan",
                "mode": "TIP_on_TL",
                "tl_table_xy": [round(tip_xy[0], 3), round(tip_xy[1], 3)],
                "tcp_xy": [target["x"], target["y"]],
                "j1_proxy": round(j1_from_tcp_xy(tcp_x, tcp_y), 4),
                "target": target,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    move(bridge, config, target)
    time.sleep(0.4)
    pose1 = get_pose(bridge, config)
    print(
        json.dumps(
            {
                "event": "done",
                "pose": pose1,
                "marked": str(OUT / "_probe_tl_before_marked.jpg"),
                "note": "TCP commanded so tip_xy ≈ TL via tip_offset.json",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
