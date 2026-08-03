"""Capture one validated Basler intrinsic-calibration image.

The active session path is stored in
``calibration_sessions/.active_basler_intrinsics_session``.  A frame is only
committed when the full 9x6 inner-corner pattern is detected and basic image
quality / border checks pass.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pypylon import genicam, pylon


HERE = Path(__file__).resolve().parent
ACTIVE_SESSION_FILE = HERE / "calibration_sessions" / ".active_basler_intrinsics_session"


def _writable(camera: pylon.InstantCamera, name: str) -> bool:
    node = camera.GetNodeMap().GetNode(name)
    return node is not None and genicam.IsWritable(node)


def _set_enum(camera: pylon.InstantCamera, name: str, value: str) -> None:
    if not _writable(camera, name):
        return
    node = camera.GetNodeMap().GetNode(name)
    if value in list(node.Symbolics):
        node.SetValue(value)


def _open_camera(session: dict) -> pylon.InstantCamera:
    factory = pylon.TlFactory.GetInstance()
    matches = [
        info
        for info in factory.EnumerateDevices()
        if info.GetSerialNumber() == str(session["serial"])
    ]
    if not matches:
        raise RuntimeError(f"Basler serial {session['serial']} is not connected")

    camera = pylon.InstantCamera(factory.CreateDevice(matches[0]))
    camera.Open()
    try:
        _set_enum(camera, "ExposureAuto", "Off")
        _set_enum(camera, "GainAuto", "Off")
        _set_enum(camera, "BalanceWhiteAuto", "Off")

        # Shrink first so offsets and final ROI are writable on this camera.
        camera.Width.SetValue(camera.Width.GetMin())
        camera.Height.SetValue(camera.Height.GetMin())
        if _writable(camera, "CenterX"):
            camera.CenterX.SetValue(False)
        if _writable(camera, "CenterY"):
            camera.CenterY.SetValue(False)
        camera.OffsetX.SetValue(int(session["roi"]["offset_x"]))
        camera.OffsetY.SetValue(int(session["roi"]["offset_y"]))
        camera.Width.SetValue(int(session["roi"]["width"]))
        camera.Height.SetValue(int(session["roi"]["height"]))
        camera.PixelFormat.SetValue(str(session["pixel_format"]))
        camera.ExposureTime.SetValue(float(session["exposure_us"]))
        camera.Gain.SetValue(float(session["gain"]))
        return camera
    except Exception:
        camera.Close()
        raise


def _capture(camera: pylon.InstantCamera) -> np.ndarray:
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    frame = None
    try:
        # Discard initial frames so the image belongs to the fixed settings.
        for _ in range(3):
            result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            try:
                if result.GrabSucceeded():
                    frame = converter.Convert(result).GetArray().copy()
            finally:
                result.Release()
    finally:
        camera.StopGrabbing()
    if frame is None:
        raise RuntimeError("Basler did not return a valid frame")
    return frame


def _converter() -> pylon.ImageFormatConverter:
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def _detect(gray: np.ndarray, pattern: tuple[int, int]) -> tuple[bool, np.ndarray | None, str]:
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    if found:
        return True, corners.reshape(-1, 2), "findChessboardCornersSB"

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, classic_flags)
    if not found:
        return False, None, "none"
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return True, refined.reshape(-1, 2), "findChessboardCorners"


def _metrics(frame: np.ndarray, corners: np.ndarray, pattern: tuple[int, int]) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    grid = corners.reshape(pattern[1], pattern[0], 2)
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2).reshape(-1)
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2).reshape(-1)
    spacing = float(np.median(np.concatenate([horizontal, vertical])))
    center = corners.mean(axis=0)
    first_row = grid[0]
    direction = first_row[-1] - first_row[0]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
    border = {
        "left": float(corners[:, 0].min()),
        "top": float(corners[:, 1].min()),
        "right": float(width - 1 - corners[:, 0].max()),
        "bottom": float(height - 1 - corners[:, 1].max()),
    }
    # Inner corners need roughly one additional square on every side.
    full_board_visible = min(border.values()) >= 1.05 * spacing
    return {
        "corner_count": int(len(corners)),
        "mean_brightness": round(float(gray.mean()), 1),
        "saturated_pct": round(float(np.mean(gray >= 250) * 100.0), 3),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "center_norm": [round(float(center[0] / width), 4), round(float(center[1] / height), 4)],
        "angle_deg": round(angle, 2),
        "board_bbox_norm": [
            round(x / width, 4),
            round(y / height, 4),
            round(w / width, 4),
            round(h / height, 4),
        ],
        "corner_spacing_px": round(spacing, 2),
        "border_px": {key: round(value, 1) for key, value in border.items()},
        "full_board_visible": bool(full_board_visible),
    }


def _validate_and_commit(
    session_dir: Path,
    session_file: Path,
    session: dict,
    frame: np.ndarray,
    pose_label: str,
) -> dict:
    pattern = tuple(int(value) for value in session["pattern_inner_corners"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners, detector = _detect(gray, pattern)
    if not found or corners is None:
        return {"accepted": False, "reason": "54 inner corners not detected"}

    metrics = _metrics(frame, corners, pattern)
    reasons = []
    if not metrics["full_board_visible"]:
        reasons.append("outer chessboard border is clipped or lacks margin")
    if metrics["sharpness"] < 60.0:
        reasons.append("image is too blurry")
    if metrics["saturated_pct"] > 5.0:
        reasons.append("too many saturated pixels")
    if reasons:
        return {"accepted": False, "reason": "; ".join(reasons), **metrics}

    index = len(session.get("images", [])) + 1
    filename = f"calib_{index:03d}.jpg"
    destination = session_dir / filename
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing image: {destination}")
    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
        raise RuntimeError(f"Failed to save {destination}")

    entry = {
        "file": filename,
        **metrics,
        "accepted": True,
        "pose_label": pose_label,
        "detector": detector,
        "captured_at": datetime.now().astimezone().isoformat(),
    }
    session.setdefault("images", []).append(entry)
    session_file.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"accepted": True, "path": str(destination), **entry}


def _save_pending(session_dir: Path, frame: np.ndarray) -> Path:
    pending_dir = session_dir / "pending"
    pending_dir.mkdir(exist_ok=True)
    existing = sorted(pending_dir.glob("pending_*.jpg"))
    index = len(existing) + 1
    destination = pending_dir / f"pending_{index:03d}.jpg"
    while destination.exists():
        index += 1
        destination = pending_dir / f"pending_{index:03d}.jpg"
    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
        raise RuntimeError(f"Failed to save {destination}")
    return destination


def _draw_live_overlay(frame: np.ndarray, pending_count: int, status: str, ok: bool) -> np.ndarray:
    display = frame.copy()
    color = (60, 220, 60) if ok else (40, 40, 240)
    cv2.rectangle(display, (0, 0), (display.shape[1], 115), (0, 0, 0), -1)
    cv2.putText(
        display,
        f"Raw captures: {pending_count}   SPACE/C: save only   Q/ESC: close",
        (22, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        status[:100],
        (22, 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        color,
        2,
        cv2.LINE_AA,
    )
    return display


def _live_capture(camera: pylon.InstantCamera, session_dir: Path, session_file: Path, session: dict) -> int:
    converter = _converter()
    window = f"Basler calibration {session['serial']} - fixed settings"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1224, 1024)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    frame = None
    pending_dir = session_dir / "pending"
    pending_count = len(list(pending_dir.glob("pending_*.jpg"))) if pending_dir.exists() else 0
    status = "Ready - capture now, validate all images later"
    status_ok = True
    last_capture_time = 0.0
    try:
        while camera.IsGrabbing():
            result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            try:
                if result.GrabSucceeded():
                    frame = converter.Convert(result).GetArray().copy()
            finally:
                result.Release()
            if frame is None:
                continue

            display = _draw_live_overlay(
                frame, pending_count, status, status_ok
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (32, ord("c"), ord("C")):
                now = time.monotonic()
                if now - last_capture_time < 0.8:
                    continue
                destination = _save_pending(session_dir, frame)
                last_capture_time = now
                pending_count += 1
                status_ok = True
                status = f"SAVED RAW {destination.name} - not validated yet"
                print(
                    json.dumps(
                        {"saved_raw": True, "path": str(destination)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-label", default="manual_varied_pose")
    parser.add_argument(
        "--live",
        action="store_true",
        help="keep preview open; SPACE/C validates and saves a frame",
    )
    args = parser.parse_args()

    if not ACTIVE_SESSION_FILE.exists():
        raise RuntimeError(f"Active session pointer is missing: {ACTIVE_SESSION_FILE}")
    session_dir = Path(ACTIVE_SESSION_FILE.read_text(encoding="utf-8-sig").strip())
    session_file = session_dir / "session.json"
    session = json.loads(session_file.read_text(encoding="utf-8-sig"))
    pattern = tuple(int(value) for value in session["pattern_inner_corners"])

    camera = _open_camera(session)
    try:
        if args.live:
            return _live_capture(camera, session_dir, session_file, session)
        frame = _capture(camera)
    finally:
        camera.Close()

    result_info = _validate_and_commit(
        session_dir, session_file, session, frame, args.pose_label
    )
    print(json.dumps(result_info, ensure_ascii=False, indent=2))
    return 0 if result_info["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
