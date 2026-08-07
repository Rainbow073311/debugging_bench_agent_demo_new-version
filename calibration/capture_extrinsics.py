"""Step 1: Capture eye-in-hand extrinsics dataset with a wide XYZ pose grid.

Fixed-R only. Builds a larger XY/Z span than the old ±8/±6 mm lattice so the
translation fit is observable, clamps to MG400 workspace, and rejects frames
with poor board detection / PnP reprojection.

Usage:
  python capture_extrinsics.py
  python capture_extrinsics.py --base-z 25 --xy-span 18 --z-span 30
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_basler_intrinsics import _open_camera, _capture  # noqa: E402

BRIDGE_PATH = HERE.parent / "Inputdemo" / "scripts" / "mg400_bridge.py"
ROBOT_CONFIG_PATH = HERE.parent / "Inputdemo" / "config" / "mg400.json"
FIXED_R = 7.687
PATTERN = (9, 6)
SQUARE_MM = 5.0
RADIUS_MIN = 255.0
RADIUS_MAX = 348.0
MAX_REPROJ_PX = 1.5


def _bridge():
    spec = importlib.util.spec_from_file_location("mg400_bridge_extrinsics", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(speed: float = 18.0) -> dict:
    cfg = json.loads(ROBOT_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    cfg.update({"speed": speed, "autoEnable": False, "returnHome": False})
    return cfg


def clamp_xy(x: float, y: float) -> tuple[float, float]:
    radius = (x * x + y * y) ** 0.5
    if radius < 1e-6:
        return x, y
    if radius < RADIUS_MIN:
        s = RADIUS_MIN / radius
        return x * s, y * s
    if radius > RADIUS_MAX:
        s = RADIUS_MAX / radius
        return x * s, y * s
    return x, y


def move(bridge, config: dict, pose: dict) -> dict:
    try:
        result = bridge.action_execute(
            {
                "config": config,
                "pose": pose,
                "trajectory": {
                    "mode": "safe-lift-traverse-descend",
                    "safeTravelZ": 50,
                    "travelSpeed": float(config.get("speed", 18)),
                    "descentSpeed": 10,
                },
            }
        )
        actual = result["robot"]["pose"]
        mode_code = result["robot"]["mode"]["code"]
    except bridge.Mg400Error:
        # Safe-trajectory envelope is conservative near the board (large radius
        # at high Z); fall back to a direct MovJ and wait for settle.
        cmd = bridge.action_command(
            {
                "config": config,
                "command": {"name": "move", "pose": pose, "motionCommand": "MovJ"},
            }
        )
        if not cmd.get("ok"):
            raise RuntimeError(f"MovJ fallback rejected: {cmd}")
        deadline = time.time() + 40.0
        actual, mode_code = None, None
        while time.time() < deadline:
            status = bridge.action_status({"config": config})
            actual = status["robot"]["pose"]
            mode_code = status["robot"]["mode"]["code"]
            settled = max(
                abs(float(actual[k]) - float(pose[k])) for k in ("x", "y", "z")
            )
            if mode_code == 5 and settled <= 1.5:
                break
            time.sleep(0.3)
    err = max(abs(float(actual[k]) - float(pose[k])) for k in ("x", "y", "z"))
    if mode_code != 5 or err > 1.5:
        raise RuntimeError(f"not settled: target={pose} actual={actual} err={err:.2f}")
    return {k: float(actual[k]) for k in ("x", "y", "z", "r")}


def get_pose(bridge, config: dict) -> dict:
    status = bridge.action_status({"config": config})
    pose = status["robot"]["pose"]
    return {k: float(pose[k]) for k in ("x", "y", "z", "r")}


def detect_chessboard(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, PATTERN, classic_flags)
    if found:
        return cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
        )
    for g in (gray, cv2.createCLAHE(3.0, (8, 8)).apply(gray)):
        ok, c = cv2.findChessboardCornersSB(g, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            return c.reshape(-1, 1, 2).astype(np.float32)
    # Close-range frames: squares get too large in pixels for the detectors,
    # so retry on a downscaled image and refine at full resolution.
    for scale in (0.5, 0.35):
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        found, corners = cv2.findChessboardCorners(small, PATTERN, classic_flags)
        if not found:
            found, corners = cv2.findChessboardCornersSB(small, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            corners = (corners.astype(np.float64) / scale).astype(np.float32)
            return cv2.cornerSubPix(
                gray,
                corners,
                (15, 15),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
            )
    return None


def pnp(corners, K, D):
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
    ok, rvec, tvec = cv2.solvePnP(
        objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    residual = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return rvec, tvec, rmse


def build_targets(base_x: float, base_y: float, base_z: float, xy_span: float, z_span: float):
    """3x3 XY lattice × 4 Z layers, then prune radius failures / duplicates."""
    xy_offsets = [
        (0, 0),
        (xy_span, 0),
        (-xy_span, 0),
        (0, xy_span),
        (0, -xy_span),
        (xy_span * 0.7, xy_span * 0.7),
        (xy_span * 0.7, -xy_span * 0.7),
        (-xy_span * 0.7, xy_span * 0.7),
        (-xy_span * 0.7, -xy_span * 0.7),
    ]
    z_offsets = [-z_span, -z_span * 0.35, z_span * 0.35, z_span]
    targets = []
    seen = set()
    for dz in z_offsets:
        for dx, dy in xy_offsets:
            x, y = clamp_xy(base_x + dx, base_y + dy)
            z = base_z + dz
            # Keep a little clearance above the typical table / fixture band.
            if z < -80:
                continue
            if z > 140:
                continue
            key = (round(x, 1), round(y, 1), round(z, 1))
            if key in seen:
                continue
            seen.add(key)
            targets.append({"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "r": FIXED_R})
    return targets


def center_over_board(bridge, config, cam_cfg, base_z: float) -> dict:
    """Iteratively translate robot until the board is near image center."""
    pose0 = get_pose(bridge, config)
    pose = {
        "x": pose0["x"],
        "y": pose0["y"],
        "z": max(float(base_z), 15.0),
        "r": FIXED_R,
    }
    pose["x"], pose["y"] = clamp_xy(pose["x"], pose["y"])
    move(bridge, config, pose)

    last_good = None
    for step in range(12):
        time.sleep(0.8)
        cam = _open_camera(cam_cfg)
        try:
            frame = _capture(cam)
        finally:
            cam.Close()
        corners = detect_chessboard(frame)
        if corners is None:
            if last_good is None:
                raise RuntimeError(
                    "chessboard not found at probe pose; place the board flat under the camera"
                )
            # Overshoot recovery: go halfway back toward last detection pose.
            pose = {
                "x": round(0.5 * (pose["x"] + last_good["x"]), 2),
                "y": round(0.5 * (pose["y"] + last_good["y"]), 2),
                "z": float(base_z),
                "r": FIXED_R,
            }
            move(bridge, config, pose)
            continue

        pts = corners.reshape(-1, 2)
        center = pts.mean(axis=0)
        nu = float(center[0] / frame.shape[1])
        nv = float(center[1] / frame.shape[0])
        grid = pts.reshape(PATTERN[1], PATTERN[0], 2)
        spacing = float(
            np.median(
                np.concatenate(
                    [
                        np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                        np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                    ]
                )
            )
        )
        mm_per_px = SQUARE_MM / max(spacing, 1.0)
        du = float(center[0] - frame.shape[1] / 2)
        dv = float(center[1] - frame.shape[0] / 2)
        last_good = dict(pose)
        print(
            json.dumps(
                {
                    "event": "center_step",
                    "step": step,
                    "center_norm": [round(nu, 3), round(nv, 3)],
                    "spacing_px": round(spacing, 2),
                    "pose": pose,
                }
            ),
            flush=True,
        )
        if abs(nu - 0.5) < 0.10 and abs(nv - 0.5) < 0.10:
            print(json.dumps({"event": "board_centered", "base": pose}), flush=True)
            return pose

        # Partial gain. With this mount, camera X ≈ +robot X and camera Y ≈ -robot Y,
        # so image-u corrections push robot +X and image-v corrections push robot -Y.
        gain = 0.40
        bx, by = clamp_xy(
            pose["x"] + gain * du * mm_per_px,
            pose["y"] - gain * dv * mm_per_px,
        )
        pose = {"x": round(bx, 2), "y": round(by, 2), "z": float(base_z), "r": FIXED_R}
        move(bridge, config, pose)

    if last_good is None:
        raise RuntimeError("failed to center over chessboard")
    print(json.dumps({"event": "board_centered_partial", "base": last_good}), flush=True)
    move(bridge, config, last_good)
    return last_good


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-z", type=float, default=25.0, help="nominal capture Z (mm)")
    parser.add_argument("--xy-span", type=float, default=18.0, help="half-span of XY lattice (mm)")
    parser.add_argument("--z-span", type=float, default=28.0, help="half-span of Z layers (mm)")
    parser.add_argument("--speed", type=float, default=18.0)
    args = parser.parse_args()

    bridge = _bridge()
    config = _config(args.speed)
    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    K = np.array(cam_cfg["intrinsics"]["camera_matrix"], dtype=np.float64)
    D = np.array(cam_cfg["intrinsics"]["dist_coeffs"], dtype=np.float64)

    pose0 = get_pose(bridge, config)
    print(json.dumps({"event": "start", "pose": pose0}), flush=True)
    if abs(((pose0["r"] - FIXED_R + 180) % 360) - 180) > 1.0:
        move(bridge, config, {**pose0, "r": FIXED_R})

    base = center_over_board(bridge, config, cam_cfg.get("calibration", {}), args.base_z)
    targets = build_targets(base["x"], base["y"], base["z"], args.xy_span, args.z_span)
    print(
        json.dumps(
            {
                "event": "plan",
                "targets": len(targets),
                "xy_span_mm": args.xy_span,
                "z_span_mm": args.z_span,
                "x_range": [min(t["x"] for t in targets), max(t["x"] for t in targets)],
                "y_range": [min(t["y"] for t in targets), max(t["y"] for t in targets)],
                "z_range": [min(t["z"] for t in targets), max(t["z"] for t in targets)],
            }
        ),
        flush=True,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = HERE / "calibration_sessions" / f"extrinsics_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i, target in enumerate(targets, 1):
        print(
            json.dumps({"event": "move", "i": i, "n": len(targets), "target": target}),
            flush=True,
        )
        try:
            rp = move(bridge, config, target)
        except Exception as exc:
            print(json.dumps({"event": "move_fail", "i": i, "error": str(exc)[:180]}), flush=True)
            continue

        time.sleep(0.9)
        cam = _open_camera(cam_cfg.get("calibration", {}))
        try:
            frame = _capture(cam)
        finally:
            cam.Close()
        if frame is None:
            print(json.dumps({"event": "capture_fail", "i": i}), flush=True)
            continue

        corners = detect_chessboard(frame)
        if corners is None:
            cv2.imwrite(
                str(session_dir / f"fail_{i:03d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            print(json.dumps({"event": "no_board", "i": i}), flush=True)
            continue

        solved = pnp(corners, K, D)
        if solved is None:
            print(json.dumps({"event": "pnp_fail", "i": i}), flush=True)
            continue
        rvec, tvec, rmse = solved
        if rmse > MAX_REPROJ_PX:
            cv2.imwrite(
                str(session_dir / f"fail_{i:03d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            print(
                json.dumps({"event": "pnp_reject", "i": i, "rmse_px": round(rmse, 3)}),
                flush=True,
            )
            continue

        vis = frame.copy()
        cv2.drawChessboardCorners(vis, PATTERN, corners, True)
        img_name = f"img_{i:03d}.jpg"
        cv2.imwrite(str(session_dir / img_name), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        samples.append(
            {
                "index": i,
                "robot_pose": {k: float(rp[k]) for k in ("x", "y", "z", "r")},
                "marker_in_camera": {
                    "rvec": rvec.reshape(-1).tolist(),
                    "t_mm": tvec.reshape(-1).tolist(),
                },
                "pnp_rmse_px": rmse,
                "image": img_name,
            }
        )
        print(
            json.dumps(
                {
                    "event": "ok",
                    "i": i,
                    "tz_mm": round(float(tvec[2][0]), 1),
                    "rmse_px": round(rmse, 3),
                    "pose": samples[-1]["robot_pose"],
                }
            ),
            flush=True,
        )

    if samples:
        arr = np.array([[s["robot_pose"][k] for k in ("x", "y", "z")] for s in samples])
        spans = np.ptp(arr, axis=0)
    else:
        spans = np.zeros(3)

    dataset = {
        "board": {"pattern": list(PATTERN), "square_size_mm": SQUARE_MM},
        "fixed_r_deg": FIXED_R,
        "intrinsics_session": cam_cfg.get("calibration", {}).get("session"),
        "capture": {
            "base": base,
            "xy_span_mm": args.xy_span,
            "z_span_mm": args.z_span,
            "planned_targets": len(targets),
            "accepted": len(samples),
            "robot_xyz_span_mm": [float(v) for v in spans],
        },
        "samples": samples,
    }
    out = session_dir / "dataset.json"
    out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Park at a readable hover above the board.
    try:
        move(
            bridge,
            config,
            {"x": base["x"], "y": base["y"], "z": max(base["z"] + 15.0, 35.0), "r": FIXED_R},
        )
    except Exception:
        pass

    print(
        json.dumps(
            {
                "event": "done",
                "accepted": len(samples),
                "planned": len(targets),
                "robot_xyz_span_mm": [round(float(v), 1) for v in spans],
                "dataset": str(out),
                "next": "Probe TL/TR/BL/BR into probe_points.json, then: "
                "python step2_calibrate.py <dataset.json> <probe_points.json>",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if len(samples) >= 12 else 3


if __name__ == "__main__":
    raise SystemExit(main())
