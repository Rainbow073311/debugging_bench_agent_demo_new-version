"""Capture preview with Camera-title-left TL marking for user check."""
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

import calibrate_camera_j1_offset as m  # noqa: E402
from capture_basler_intrinsics import _capture, _open_camera  # noqa: E402
from capture_extrinsics import (  # noqa: E402
    PATTERN,
    _bridge,
    _config,
    detect_chessboard,
    get_pose,
    move,
)

OUT = HERE / "_extrinsic_recal_session"


def mark(frame, corners, tl_idx: int, s08: dict, s45: dict, label: str) -> Path:
    pts = corners.reshape(-1, 2).astype(float)
    vis = frame.copy()
    cv2.drawChessboardCorners(vis, PATTERN, corners, True)
    for i in m.OUTER_IDX:
        col = (0, 0, 255) if i == tl_idx else (255, 128, 0)
        pt = (int(pts[i, 0]), int(pts[i, 1]))
        cv2.circle(vis, pt, 30 if i == tl_idx else 18, col, 3)
        cv2.putText(
            vis,
            "TL" if i == tl_idx else str(i),
            (pt[0] + 12, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            col,
            3,
        )
    cv2.putText(
        vis,
        f"{label} rawIdx={tl_idx} s08={s08['score']:.2f} s45={s45['score']:.2f}",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        vis,
        "Must be under word Camera when paper reads LTR",
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "preview_tl_marked.jpg"
    cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def main() -> int:
    # Re-check previous raw if present
    old = OUT / "preview_tl_raw.jpg"
    if old.exists():
        frame = cv2.imread(str(old))
        corners = detect_chessboard(frame)
        if corners is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pts = corners.reshape(-1, 2).astype(float)
            board_c = pts.mean(0)
            s08 = m._long_edge_camera_score(gray, pts, 0, 8, board_c)
            s45 = m._long_edge_camera_score(gray, pts, 45, 53, board_c)
            tl = m.physical_tl_open_cv_index(gray, corners)
            print(
                json.dumps(
                    {"event": "old_raw_recheck", "tl": tl, "s08": s08, "s45": s45},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    bridge, config = _bridge(), _config(12)
    look = {"x": 320.0, "y": 20.0, "z": 50.0, "r": 7.687}
    try:
        pose = move(bridge, config, look)
    except Exception as exc:
        print(json.dumps({"move_warn": str(exc)}, ensure_ascii=False), flush=True)
        pose = get_pose(bridge, config)
    time.sleep(0.4)

    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    cam = _open_camera(cam_cfg["calibration"])
    try:
        for _ in range(3):
            _capture(cam)
        time.sleep(0.15)
        frame = _capture(cam)
    finally:
        cam.Close()

    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "preview_tl_raw.jpg"
    cv2.imwrite(str(raw), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    corners = detect_chessboard(frame)
    if corners is None:
        print(json.dumps({"ok": False, "error": "no board", "raw": str(raw)}, ensure_ascii=False))
        return 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = corners.reshape(-1, 2).astype(float)
    board_c = pts.mean(0)
    s08 = m._long_edge_camera_score(gray, pts, 0, 8, board_c)
    s45 = m._long_edge_camera_score(gray, pts, 45, 53, board_c)
    tl = m.physical_tl_open_cv_index(gray, corners)
    marked = mark(frame, corners, tl, s08, s45, "FIXED Camera-LEFT TL")
    print(
        json.dumps(
            {
                "ok": True,
                "tl_idx": int(tl),
                "tl_px": [round(float(pts[tl, 0]), 1), round(float(pts[tl, 1]), 1)],
                "score08": s08,
                "score45": s45,
                "cam_edge": "(0,8)" if s08["score"] >= s45["score"] else "(45,53)",
                "pose": pose,
                "raw": str(raw),
                "marked": str(marked),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
