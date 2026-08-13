"""Capture extra LOOK poses L9+ for extrinsic recal."""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capture_basler_intrinsics import _capture, _open_camera  # noqa: E402
from capture_extrinsics import (  # noqa: E402
    _bridge,
    _config,
    clamp_xy,
    detect_chessboard,
    move,
)

OUT = HERE / "_extrinsic_recal_session"


def recover(bridge, config) -> None:
    robot = bridge.Mg400(config)
    try:
        robot.connect_dashboard()
        robot.dash("ClearError()")
        robot.dash("Continue()")
        m = robot.dash("RobotMode()")
        resp = str(m.get("response", ""))
        if "{4}" in resp or "{9}" in resp:
            robot.dash("ClearError()")
            robot.dash("EnableRobot(0.5)")
        time.sleep(0.5)
        print("recover", robot.dash("RobotMode()"), flush=True)
    finally:
        robot.close()


def main() -> int:
    bridge, config = _bridge(), _config(12)
    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    candidates = [
        ("L9", 300.0, 60.0, 85.0),
        ("L10", 350.0, -40.0, 95.0),
        ("L11", 290.0, -10.0, 120.0),
        ("L12", 355.0, 30.0, 110.0),
        ("L13", 325.0, 70.0, 100.0),
        ("L14", 330.0, -50.0, 115.0),
    ]
    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    for label, x, y, z in candidates:
        x, y = clamp_xy(x, y)
        tgt = {"x": float(x), "y": float(y), "z": float(z), "r": 7.687}
        print(
            f"--- {label} j1~{math.degrees(math.atan2(y, x)):.1f} {tgt}",
            flush=True,
        )
        try:
            move(bridge, config, tgt)
        except Exception as exc:
            print("move fail", exc, flush=True)
            recover(bridge, config)
            try:
                move(bridge, config, tgt)
            except Exception as exc2:
                print("move fail2", exc2, flush=True)
                failed.append((label, "move"))
                continue
        time.sleep(0.5)
        cam = _open_camera(cam_cfg["calibration"])
        try:
            for _ in range(3):
                _capture(cam)
            time.sleep(0.12)
            frame = _capture(cam)
        finally:
            cam.Close()
        if detect_chessboard(frame) is None:
            OUT.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(OUT / f"_fail_{label}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            print("NO BOARD", label, flush=True)
            failed.append((label, "no_board"))
            continue
        r = subprocess.run(
            [sys.executable, "-u", str(HERE / "calibrate_extrinsics_recal.py"), "look", label],
            capture_output=True,
            text=True,
        )
        idx_m = re.search(r'"raw_tl_idx":\s*(\d+)', r.stdout or "")
        j1_m = re.search(r'"j1_deg":\s*([-\d.]+)', r.stdout or "")
        print(
            "look_ok" if r.returncode == 0 else "look_fail",
            label,
            "raw_tl_idx",
            idx_m.group(1) if idx_m else "?",
            "j1",
            j1_m.group(1) if j1_m else "?",
            flush=True,
        )
        if r.returncode != 0:
            print((r.stderr or r.stdout or "")[-500:], flush=True)
            failed.append((label, "look"))
            continue
        ok.append(label)

    r = subprocess.run(
        [sys.executable, "-u", str(HERE / "calibrate_extrinsics_recal.py"), "status"],
        capture_output=True,
        text=True,
    )
    print("STATUS", r.stdout, flush=True)
    print(json.dumps({"ok": ok, "failed": failed}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
