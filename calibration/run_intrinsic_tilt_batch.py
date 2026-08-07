"""Capture one 5-pose intrinsic batch with robot motion; user changes board tilt between batches."""
from __future__ import annotations

import importlib.util
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
from capture_basler_intrinsics import (  # noqa: E402
    ACTIVE_SESSION_FILE,
    _capture,
    _detect,
    _metrics,
    _open_camera,
    _save_pending,
)

BRIDGE_PATH = HERE.parent / "Inputdemo" / "scripts" / "mg400_bridge.py"
ROBOT_CONFIG_PATH = HERE.parent / "Inputdemo" / "config" / "mg400.json"
FIXED_R = 7.687
PATTERN = (9, 6)


def _bridge():
    spec = importlib.util.spec_from_file_location("mg400_bridge_batch", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    cfg = json.loads(ROBOT_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    cfg.update({"speed": 18, "autoEnable": False, "returnHome": False})
    return cfg


def move(bridge, config, pose: dict) -> dict:
    result = bridge.action_execute(
        {
            "config": config,
            "pose": pose,
            "trajectory": {
                "mode": "safe-lift-traverse-descend",
                "safeTravelZ": 50,
                "travelSpeed": 18,
                "descentSpeed": 10,
            },
        }
    )
    actual = result["robot"]["pose"]
    err = max(abs(float(actual[k]) - float(pose[k])) for k in ("x", "y", "z"))
    if result["robot"]["mode"]["code"] != 5 or err > 1.5:
        raise RuntimeError(f"not settled: target={pose} actual={actual} err={err}")
    return {k: float(actual[k]) for k in ("x", "y", "z", "r")}


def get_pose(bridge, config) -> dict:
    status = bridge.action_status({"config": config})
    pose = status["robot"]["pose"]
    return {k: float(pose[k]) for k in ("x", "y", "z", "r")}


def ensure_session() -> tuple[Path, dict, Path]:
    if ACTIVE_SESSION_FILE.exists():
        session_dir = Path(ACTIVE_SESSION_FILE.read_text(encoding="utf-8-sig").strip())
        session_file = session_dir / "session.json"
        if session_file.exists():
            session = json.loads(session_file.read_text(encoding="utf-8-sig"))
            if session.get("mode") == "tilt_batches_v1":
                (session_dir / "pending").mkdir(exist_ok=True)
                return session_dir, session, session_file

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = HERE / "calibration_sessions" / f"intrinsics_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "pending").mkdir(exist_ok=True)
    session = {
        "status": "collecting",
        "mode": "tilt_batches_v1",
        "camera_model": "Basler acA2440-75uc",
        "serial": "24771618",
        "resolution": [2448, 2048],
        "roi": {"width": 2448, "height": 2048, "offset_x": 0, "offset_y": 0},
        "pixel_format": "BayerRG8",
        "pattern_inner_corners": [9, 6],
        "square_size_mm": 5.0,
        "exposure_us": 400000,
        "gain": 0,
        "created_at": datetime.now().astimezone().isoformat(),
        "batches": [],
        "images": [],
        "note": "user tilts board between 5-shot robot batches",
    }
    session_file = session_dir / "session.json"
    session_file.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ACTIVE_SESSION_FILE.write_text(str(session_dir.resolve()) + "\n", encoding="utf-8")
    return session_dir, session, session_file


def grab_frame(cam_cfg: dict):
    cam = _open_camera(cam_cfg)
    try:
        return _capture(cam)
    finally:
        cam.Close()


def find_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners, det = _detect(gray, PATTERN)
    if found:
        return corners, det, _metrics(frame, corners, PATTERN)
    # Plain SB (non-exhaustive) copes better with low-contrast/glare frames.
    for label, g in (("sb_plain", gray), ("sb_clahe", cv2.createCLAHE(3.0, (8, 8)).apply(gray))):
        ok, c = cv2.findChessboardCornersSB(g, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            c = c.reshape(-1, 2)
            return c, label, _metrics(frame, c, PATTERN)
    # ROI fallback for off-center boards
    h, w = gray.shape
    for y0, y1, x0, x1 in [
        (0, int(h * 0.7), 0, int(w * 0.7)),
        (0, int(h * 0.7), int(w * 0.3), w),
        (int(h * 0.3), h, 0, int(w * 0.7)),
        (int(h * 0.3), h, int(w * 0.3), w),
        (0, h, 0, w),
    ]:
        ok, c = cv2.findChessboardCornersSB(
            gray[y0:y1, x0:x1],
            PATTERN,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if ok:
            c = c.reshape(-1, 2)
            c[:, 0] += x0
            c[:, 1] += y0
            return c, "roi_sb", _metrics(frame, c, PATTERN)
    return None, None, None


def main() -> int:
    batch_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_shots = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    base_z = float(sys.argv[3]) if len(sys.argv) > 3 else 25.0

    session_dir, session, session_file = ensure_session()
    if batch_index <= 0:
        batch_index = len(session.get("batches", [])) + 1

    bridge = _bridge()
    config = _config()
    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))["calibration"]

    pose0 = get_pose(bridge, config)
    print(json.dumps({"event": "start", "session": session_dir.name, "batch": batch_index, "pose": pose0}), flush=True)

    # Probe board at a safe working height near current XY
    probe_z = max(float(pose0["z"]), 15.0)
    probe = {"x": pose0["x"], "y": pose0["y"], "z": probe_z, "r": FIXED_R}
    move(bridge, config, probe)
    time.sleep(0.8)
    frame = grab_frame(cam_cfg)
    corners, det, metrics = find_board(frame)
    if corners is None:
        fail = session_dir / f"batch{batch_index:02d}_probe_fail.jpg"
        cv2.imwrite(str(fail), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(json.dumps({"event": "no_board", "saved": str(fail)}), flush=True)
        return 2

    center = corners.mean(axis=0)
    print(
        json.dumps(
            {
                "event": "board_found",
                "detector": det,
                "center_norm": [float(center[0] / frame.shape[1]), float(center[1] / frame.shape[0])],
                "metrics": {
                    k: metrics[k]
                    for k in (
                        "center_norm",
                        "board_bbox_norm",
                        "full_board_visible",
                        "sharpness",
                        "corner_spacing_px",
                        "angle_deg",
                    )
                },
            }
        ),
        flush=True,
    )

    # Approximate robot move to center board using pixel offset -> mm via spacing
    # At current height, ~corner_spacing_px corresponds to 5mm
    spacing = max(float(metrics["corner_spacing_px"]), 1.0)
    mm_per_px = 5.0 / spacing
    du = float(center[0] - frame.shape[1] / 2)
    dv = float(center[1] - frame.shape[0] / 2)
    # Eye-in-hand looking mostly down with fixed R~7.7deg: image +u roughly -robot X-ish / +v roughly +robot Y
    # Empirically from earlier session: moving robot +X moves board left in image (u decreases)
    # so to center board currently at +du (right of center), move robot +X by du*mm_per_px? 
    # Earlier: board at left (u small), needed smaller robot X. board world X smaller than lookat.
    # suggested: robot += (board_world - lookat) with board left => negative dX in world when u smaller.
    # If board center_u > image_center, board is to the right in image => need robot move that shifts view right.
    # From prior: delta board-look was negative X when board was left (smaller u).
    # board_u < center => board left => robot should decrease X? Prior: board u~630, center 1224, board world X smaller, suggested robot X smaller.
    # So robot_delta_x ≈ (center_u - board_u) * scale = -du * mm_per_px if du = board-center.
    # du = board - center; robot_dx ≈ -du * mm_per_px
    # For v: prior board v small (up), board world Y larger, suggested robot Y larger. dv=board-center negative => robot_dy ≈ -dv * mm_per_px
    base_x = float(probe["x"]) - du * mm_per_px
    base_y = float(probe["y"]) - dv * mm_per_px
    # Keep radius safe-ish (>= ~280)
    if (base_x**2 + base_y**2) ** 0.5 < 280:
        scale = 280 / ((base_x**2 + base_y**2) ** 0.5)
        base_x *= scale
        base_y *= scale

    # 5 poses: center + 4 offsets, slight Z variety within batch
    offsets = [
        (0, 0, 0),
        (12, 0, -8),
        (-12, 0, 8),
        (0, 12, -5),
        (0, -12, 10),
    ]
    # Radial reach shrinks as Z rises; keep every target inside a safe radius.
    max_radius = 348.0
    radius = (base_x**2 + base_y**2) ** 0.5
    if radius > max_radius:
        shrink = max_radius / radius
        base_x *= shrink
        base_y *= shrink
    targets = []
    for dx, dy, dz in offsets[:n_shots]:
        targets.append(
            {
                "x": round(base_x + dx, 2),
                "y": round(base_y + dy, 2),
                "z": round(base_z + dz, 2),
                "r": FIXED_R,
            }
        )

    batch = {
        "batch": batch_index,
        "started_at": datetime.now().astimezone().isoformat(),
        "base_pose": {"x": base_x, "y": base_y, "z": base_z, "r": FIXED_R},
        "shots": [],
    }
    saved = 0
    for i, target in enumerate(targets, 1):
        print(json.dumps({"event": "move", "shot": i, "target": target}), flush=True)
        try:
            actual = move(bridge, config, target)
        except Exception as exc:
            batch["shots"].append({"shot": i, "target": target, "status": "move_fail", "error": str(exc)})
            print(json.dumps({"event": "move_fail", "shot": i, "error": str(exc)}), flush=True)
            continue
        time.sleep(1.0)  # settle for 400ms exposure
        frame = grab_frame(cam_cfg)
        corners, det, metrics = find_board(frame)
        path = _save_pending(session_dir, frame)
        entry = {
            "shot": i,
            "file": path.name,
            "pose": actual,
            "target": target,
            "detector": det,
            "found": corners is not None,
        }
        if metrics:
            entry["metrics"] = {
                k: metrics[k]
                for k in (
                    "center_norm",
                    "board_bbox_norm",
                    "full_board_visible",
                    "sharpness",
                    "corner_spacing_px",
                    "angle_deg",
                    "mean_brightness",
                )
            }
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, PATTERN, corners.reshape(-1, 1, 2), True)
            marked = session_dir / "pending" / path.name.replace(".jpg", "_marked.jpg")
            cv2.imwrite(str(marked), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
            print(
                json.dumps(
                    {
                        "event": "saved",
                        "shot": i,
                        "file": path.name,
                        "found": True,
                        "sharpness": metrics["sharpness"],
                        "full_board_visible": metrics["full_board_visible"],
                        "center_norm": metrics["center_norm"],
                    }
                ),
                flush=True,
            )
        else:
            fail = session_dir / f"batch{batch_index:02d}_fail_{i:02d}.jpg"
            cv2.imwrite(str(fail), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            entry["fail_image"] = fail.name
            print(json.dumps({"event": "saved_no_board", "shot": i, "file": path.name}), flush=True)
        batch["shots"].append(entry)

    batch["finished_at"] = datetime.now().astimezone().isoformat()
    batch["saved_with_board"] = saved
    session.setdefault("batches", []).append(batch)
    session_file.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (session_dir / f"batch_{batch_index:02d}.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Park at readable hover
    park = {"x": round(base_x, 2), "y": round(base_y, 2), "z": 35.0, "r": FIXED_R}
    try:
        move(bridge, config, park)
    except Exception:
        pass

    print(
        json.dumps(
            {
                "event": "batch_done",
                "batch": batch_index,
                "saved_with_board": saved,
                "total_pending": len(list((session_dir / "pending").glob("pending_*.jpg"))),
                "session": str(session_dir),
                "next": "请换棋盘倾角，说「拍」继续下一组",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if saved >= 3 else 3


if __name__ == "__main__":
    raise SystemExit(main())
