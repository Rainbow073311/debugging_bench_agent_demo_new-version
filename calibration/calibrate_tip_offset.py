"""Look -> Touch tip-offset calibration (R-independent).

The probe tip is fixed on the head and is NOT the robot TCP.
The camera also cannot see the tip when the tip is on a corner — that is fine.

For each chessboard inner corner (TL/TR/BL/BR):
  1) LOOK: move so the camera sees the board clearly (tip away from that corner).
     Capture + OpenCV detect + project corner to base -> P
  2) TOUCH: jog the TIP onto the same physical corner (camera may not see tip).
     Read TCP -> tip_offset_sample = P - TCP

Convention:
  tip_xyz = TCP_xyz + tip_offset
  To put the tip on target T, command TCP = T - tip_offset

Usage:
  python calibration/calibrate_tip_offset.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_basler_intrinsics import _capture, _open_camera
from capture_extrinsics import FIXED_R, _bridge, _config, detect_chessboard, get_pose
from coordinate_transforms import PixelToWorld
from eye_in_hand_xyz import compose_base_to_camera

PATTERN = (9, 6)
SQUARE_MM = 5.0
CORNERS = (
    ("TL", 0),
    ("TR", 8),
    ("BL", 45),
    ("BR", 53),
)
OUT_JSON = HERE / "_tip_offset.json"
OUT_DIR = HERE / "_tip_offset_session"


def _pause(msg: str) -> None:
    print(msg, flush=True)
    try:
        input()
    except EOFError:
        raise SystemExit("stdin closed")


def _round3(v) -> list[float]:
    return [round(float(x), 3) for x in v]


def _project_corner(
    frame: np.ndarray,
    corner_idx: int,
    pose: dict,
    cfg: dict,
) -> dict:
    corners = detect_chessboard(frame)
    if corners is None:
        raise RuntimeError("chessboard not detected — raise camera / re-center board")
    corners = corners.reshape(-1, 2)
    px = corners[corner_idx]

    ptw = PixelToWorld(str(HERE / "camera_config.yaml"))
    ptw.set_robot_pose(pose)
    table = ptw.pixel_to_table(float(px[0]), float(px[1]))

    K = np.array(cfg["intrinsics"]["camera_matrix"], float)
    D = np.array(cfg["intrinsics"]["dist_coeffs"], float)
    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(-1, 1, 2), K, D)
    if not ok:
        raise RuntimeError("solvePnP failed")
    R_cam, _ = cv2.Rodrigues(rvec)
    obj_pt = obj[corner_idx]
    cam_pt = (R_cam @ obj_pt + tvec.reshape(3)).ravel()

    T_ec = np.eye(4)
    T_ec[:3, :3] = np.array(cfg["extrinsics"]["T_end_to_camera"]["R"], float)
    T_ec[:3, 3] = np.array(cfg["extrinsics"]["T_end_to_camera"]["t_mm"], float)
    T_bc = compose_base_to_camera(pose, T_ec)
    pnp_base = (T_bc @ np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0]))[:3]

    # Prefer plane intersection (matches runtime); fall back to PnP 3D.
    if table is not None:
        P = np.array(table, float)
        source = "pixel_to_table"
    else:
        P = np.array(pnp_base, float)
        source = "pnp_base"

    return {
        "pixel": [float(px[0]), float(px[1])],
        "P_mm": P,
        "P_source": source,
        "pixel_to_table": None if table is None else _round3(table),
        "pnp_base": _round3(pnp_base),
        "corners": corners,
    }


def _mark(frame: np.ndarray, corners: np.ndarray, name: str, idx: int, path: Path) -> None:
    vis = frame.copy()
    cv2.drawChessboardCorners(vis, PATTERN, corners.reshape(-1, 1, 2), True)
    x, y = map(int, corners[idx])
    cv2.circle(vis, (x, y), 28, (0, 0, 255), 4)
    cv2.putText(vis, name, (x + 24, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
    cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge = _bridge()
    config = _config(12)
    cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))

    print("=" * 60, flush=True)
    print("Tip offset: Look -> Touch (camera need NOT see tip on touch)", flush=True)
    print(f"Fixed R should stay ~{FIXED_R} deg. Board must not move.", flush=True)
    print("tip = TCP + tip_offset", flush=True)
    print("=" * 60, flush=True)

    pose0 = get_pose(bridge, config)
    print(json.dumps({"event": "start_pose", "pose": pose0}, ensure_ascii=False), flush=True)
    if abs(float(pose0["r"]) - FIXED_R) > 0.5:
        print(
            f"WARNING: current R={pose0['r']:.3f} differs from FIXED_R={FIXED_R}",
            flush=True,
        )

    samples = []
    for name, idx in CORNERS:
        print("\n" + "-" * 60, flush=True)
        print(f"[{name}] LOOK phase", flush=True)
        print("  Move so camera clearly sees the chessboard (tip away from this corner).", flush=True)

        while True:
            _pause(f"  Press Enter when ready to CAPTURE for {name}...")
            look_pose = get_pose(bridge, config)
            cam = _open_camera(cfg["calibration"])
            try:
                _capture(cam)
                time.sleep(0.15)
                frame = _capture(cam)
            finally:
                cam.Close()

            raw_path = OUT_DIR / f"look_{name}.jpg"
            cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            try:
                proj = _project_corner(frame, idx, look_pose, cfg)
                break
            except Exception as exc:
                print(f"LOOK failed: {exc}", flush=True)
                print("Reposition for a clearer view, then retry.", flush=True)

        marked = OUT_DIR / f"look_{name}_marked.jpg"
        _mark(frame, proj["corners"], name, idx, marked)
        P = proj["P_mm"]
        print(
            json.dumps(
                {
                    "event": "look",
                    "corner": name,
                    "look_pose": look_pose,
                    "pixel": _round3(proj["pixel"]),
                    "P_mm": _round3(P),
                    "P_source": proj["P_source"],
                    "pixel_to_table": proj["pixel_to_table"],
                    "pnp_base": proj["pnp_base"],
                    "marked": str(marked),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        print(f"  Optical P ({proj['P_source']}): {_round3(P)}", flush=True)
        print(f"  Check image: {marked}", flush=True)

        print(f"\n[{name}] TOUCH phase", flush=True)
        print("  Jog TIP onto the SAME physical corner. Camera does not need to see tip.", flush=True)
        print("  Keep R fixed. Do not move the board.", flush=True)
        _pause(f"  Press Enter when TIP is ON {name} to READ TCP...")

        touch_pose = get_pose(bridge, config)
        tcp = np.array([touch_pose["x"], touch_pose["y"], touch_pose["z"]], float)
        tip_offset = P - tcp
        sample = {
            "corner": name,
            "corner_index": idx,
            "look_pose": look_pose,
            "touch_tcp": _round3(tcp),
            "P_mm": _round3(P),
            "P_source": proj["P_source"],
            "tip_offset_mm": _round3(tip_offset),
            "r_look": float(look_pose["r"]),
            "r_touch": float(touch_pose["r"]),
        }
        samples.append(sample)
        print(
            json.dumps({"event": "touch", **sample}, ensure_ascii=False),
            flush=True,
        )
        print(f"  TCP touch: {_round3(tcp)}", flush=True)
        print(f"  tip_offset sample: {_round3(tip_offset)}", flush=True)

    offsets = np.array([s["tip_offset_mm"] for s in samples], float)
    mean = offsets.mean(axis=0)
    std = offsets.std(axis=0)
    spread = offsets.max(axis=0) - offsets.min(axis=0)

    result = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "convention": "tip_xyz = TCP_xyz + tip_offset_mm (R-independent)",
        "command_tip_to_target": "TCP = target - tip_offset_mm",
        "fixed_r_deg": FIXED_R,
        "P_method_note": "Prefer pixel_to_table (runtime plane); PnP logged for cross-check",
        "samples": samples,
        "tip_offset_mm": _round3(mean),
        "tip_offset_std_mm": _round3(std),
        "tip_offset_spread_mm": _round3(spread),
        "status": "estimated",
        "next": (
            "1) Apply tip_offset when commanding probe motion. "
            "2) Correct old probe points: tip = TCP_probe + tip_offset, re-fit T_end_to_camera. "
            "3) Re-run this script once to refine."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"MEAN tip_offset_mm = {_round3(mean)}", flush=True)
    print(f"STD  tip_offset_mm = {_round3(std)}", flush=True)
    print(f"SPREAD (max-min)   = {_round3(spread)}", flush=True)
    print(f"Saved: {OUT_JSON}", flush=True)
    if float(np.linalg.norm(spread[:2])) > 3.0:
        print(
            "WARNING: XY spread > 3 mm across corners — check corner identity / board moved / R changed.",
            flush=True,
        )
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
