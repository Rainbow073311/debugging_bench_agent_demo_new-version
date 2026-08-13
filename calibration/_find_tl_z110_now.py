"""At current pose: photo -> physical TL -> move tip to TL at Z=-110."""
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

from calibrate_camera_j1_offset import OUTER_IDX, physical_tl_open_cv_index  # noqa: E402
from capture_basler_intrinsics import _capture, _open_camera  # noqa: E402
from capture_extrinsics import (  # noqa: E402
    _bridge,
    _config,
    clamp_xy,
    detect_chessboard,
    get_pose,
    move,
)
from coordinate_transforms import PixelToWorld  # noqa: E402
from tip_offset import tcp_xy_from_tip_config  # noqa: E402

OUT = HERE / "_find_tl_z110_session"
DEFAULT_Z = -110.0
CFG = HERE / "camera_config.yaml"


def capture(cfg: dict) -> np.ndarray:
    cam = _open_camera(cfg["calibration"])
    try:
        for _ in range(3):
            _capture(cam)
        time.sleep(0.12)
        return _capture(cam)
    finally:
        cam.Close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    target_z = float(argv[0]) if argv else DEFAULT_Z
    OUT.mkdir(parents=True, exist_ok=True)
    cam_cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    bridge, config = _bridge(), _config(15)
    pose0 = get_pose(bridge, config)
    print(
        json.dumps(
            {"event": "start", "pose": pose0, "target_z": target_z},
            ensure_ascii=False,
        ),
        flush=True,
    )

    frame = capture(cam_cfg)
    raw_path = OUT / "now_raw.jpg"
    cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    corners = detect_chessboard(frame)
    if corners is None:
        print(
            json.dumps(
                {"ok": False, "error": "chessboard not found", "raw": str(raw_path)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tl_idx = physical_tl_open_cv_index(
        gray, corners, robot_pose=pose0, config_path=CFG
    )
    pts = corners.reshape(-1, 2)
    tl_px = pts[tl_idx]
    print(
        json.dumps(
            {
                "event": "tl_detected",
                "tl_idx": int(tl_idx),
                "tl_px": [round(float(tl_px[0]), 1), round(float(tl_px[1]), 1)],
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
    marked_path = OUT / "now_marked.jpg"
    cv2.imwrite(str(marked_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ptw = PixelToWorld(str(CFG), robot_pose=pose0)
    table = ptw.pixel_to_table(float(tl_px[0]), float(tl_px[1]), pose0)
    if table is None:
        print(json.dumps({"ok": False, "error": "TL projection failed"}, ensure_ascii=False))
        return 3

    tip_xy = (float(table[0]), float(table[1]))
    solved = tcp_xy_from_tip_config(tip_xy)
    tcp_x, tcp_y = clamp_xy(float(solved["tcp_xy"][0]), float(solved["tcp_xy"][1]))
    target = {
        "x": round(tcp_x, 2),
        "y": round(tcp_y, 2),
        "z": target_z,
        "r": float(pose0.get("r", 7.687)),
    }
    print(
        json.dumps(
            {
                "event": "plan",
                "model": cam_cfg.get("extrinsics", {}).get("j1_model"),
                "tl_table_xy": [round(tip_xy[0], 3), round(tip_xy[1], 3)],
                "tcp_from_tip": solved,
                "target": target,
                "raw": str(raw_path),
                "marked": str(marked_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    pose1 = move(bridge, config, target)
    print(
        json.dumps(
            {
                "event": "hover",
                "pose": pose1,
                "note": f"Tip should be above Camera-title LEFT at Z={target_z}",
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
