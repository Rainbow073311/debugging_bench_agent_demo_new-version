"""Temp-apply extrinsic fit and hover tip on Camera-title TL for validation."""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from calibrate_camera_j1_offset import OUTER_IDX, physical_tl_open_cv_index  # noqa: E402
from capture_basler_intrinsics import _capture, _open_camera  # noqa: E402
from capture_extrinsics import (  # noqa: E402
    PATTERN,
    _bridge,
    _config,
    clamp_xy,
    detect_chessboard,
    get_pose,
    move,
)
from coordinate_transforms import PixelToWorld  # noqa: E402
from tip_offset import tcp_xy_from_tip_config  # noqa: E402

CFG = HERE / "camera_config.yaml"
STATE = HERE / "_extrinsic_recal.json"
OUT = HERE / "_extrinsic_recal_session"


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    fit = state.get("fit")
    if not fit:
        raise RuntimeError("no fit")
    print(
        json.dumps(
            {
                "fit": {
                    k: fit[k]
                    for k in (
                        "n_looks",
                        "rms_mm",
                        "max_mm",
                        "mean_abs_mm",
                        "j1_span_deg",
                        "t_mm",
                        "tl_idx_choice",
                    )
                }
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if float(fit["rms_mm"]) > 8:
        raise RuntimeError(f"rms too high: {fit['rms_mm']}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = HERE / f"camera_config.yaml.bak_pre_recal_{stamp}"
    shutil.copy2(CFG, bak)
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    old_t = list(data["extrinsics"]["T_end_to_camera"]["t_mm"])
    data["extrinsics"]["T_end_to_camera"]["R"] = fit["R"]
    data["extrinsics"]["T_end_to_camera"]["t_mm"] = fit["t_mm"]
    data["extrinsics"]["date"] = "candidate-recal-" + datetime.now().strftime("%Y%m%d")
    data["extrinsics"]["note_candidate"] = (
        f"from _extrinsic_recal fit rms={fit['rms_mm']}mm; TEMP until tip validate"
    )
    CFG.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(
        json.dumps(
            {"event": "temp_apply", "backup": str(bak), "t_old": old_t, "t_new": fit["t_mm"]},
            ensure_ascii=False,
        ),
        flush=True,
    )

    bridge, config = _bridge(), _config(12)
    look = {"x": 320.0, "y": 20.0, "z": 50.0, "r": 7.687}
    pose = move(bridge, config, look)
    time.sleep(0.4)

    cam_cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    cam = _open_camera(cam_cfg["calibration"])
    try:
        for _ in range(3):
            _capture(cam)
        time.sleep(0.12)
        frame = _capture(cam)
    finally:
        cam.Close()

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "validate_raw.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    corners = detect_chessboard(frame)
    if corners is None:
        raise RuntimeError("no board")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p_tl = np.array(state["P_TL"][:2], float)
    tl_idx = physical_tl_open_cv_index(
        gray, corners, prefer_xy_mm=p_tl, robot_pose=pose, config_path=CFG
    )
    tl_ink = physical_tl_open_cv_index(gray, corners)
    ptw = PixelToWorld(str(CFG), robot_pose=pose)
    px = corners.reshape(-1, 2)[tl_idx]
    table = ptw.pixel_to_table(float(px[0]), float(px[1]), pose)
    tip_xy = (float(table[0]), float(table[1]))
    err = float(np.linalg.norm(np.array(tip_xy) - p_tl))
    solved = tcp_xy_from_tip_config(tip_xy)
    tcp_x, tcp_y = clamp_xy(float(solved["tcp_xy"][0]), float(solved["tcp_xy"][1]))
    target = {"x": round(tcp_x, 2), "y": round(tcp_y, 2), "z": -145.0, "r": 7.687}

    vis = frame.copy()
    cv2.drawChessboardCorners(vis, PATTERN, corners, True)
    for i in OUTER_IDX:
        col = (0, 0, 255) if i == tl_idx else (255, 128, 0)
        p = corners.reshape(-1, 2)[i]
        pt = (int(p[0]), int(p[1]))
        cv2.circle(vis, pt, 20, col, 3)
        cv2.putText(
            vis,
            "TL" if i == tl_idx else str(i),
            (pt[0] + 8, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            col,
            2,
        )
    cv2.putText(
        vis,
        f"validate err_vs_P_TL={err:.2f}mm idx={tl_idx} ink={tl_ink}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    cv2.imwrite(str(OUT / "validate_marked.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(
        json.dumps(
            {
                "event": "validate_plan",
                "tl_idx": int(tl_idx),
                "ink_idx": int(tl_ink),
                "tip_xy": [round(tip_xy[0], 3), round(tip_xy[1], 3)],
                "P_TL": [round(float(v), 3) for v in p_tl],
                "err_mm": round(err, 3),
                "target": target,
                "backup": str(bak),
            },
            indent=2,
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
                "note": "Check tip on Camera-title LEFT corner. Reply yes to lock, no to restore.",
            },
            indent=2,
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
