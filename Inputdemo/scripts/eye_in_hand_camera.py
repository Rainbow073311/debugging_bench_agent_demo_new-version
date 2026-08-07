#!/usr/bin/env python
"""Hardware bridge for the guarded XYZ-only eye-in-hand capture workflow."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.coordinate_transforms import PixelToWorld  # noqa: E402


def read_payload() -> dict[str, Any]:
    text = sys.stdin.read().strip()
    return json.loads(text) if text else {}


def calibration_path(payload: dict[str, Any]) -> Path:
    value = payload.get("calibrationFile")
    return Path(value).resolve() if value else REPO_ROOT / "calibration" / "camera_config.yaml"


def calibration_status(payload: dict[str, Any]) -> str:
    path = calibration_path(payload)
    if not path.exists():
        return "missing"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(data.get("extrinsics", {}).get("status", "missing"))


def output_dir(payload: dict[str, Any]) -> Path:
    value = payload.get("outputDir")
    target = Path(value).resolve() if value else REPO_ROOT / "Inputdemo" / "data" / "eye_in_hand_captures"
    target.mkdir(parents=True, exist_ok=True)
    return target


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class BaslerCameraAdapter:
    def __init__(self, settings: dict[str, Any]):
        from calibration.capture_basler_intrinsics import _open_camera

        self._camera = _open_camera(settings)

    def read(self):
        from calibration.capture_basler_intrinsics import _capture

        return True, _capture(self._camera)

    def release(self):
        self._camera.Close()


def open_camera(payload: dict[str, Any]):
    config_file = calibration_path(payload)
    data = {}
    if config_file.exists():
        data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        calibration = data.get("calibration", {})
        if str(calibration.get("camera_model", "")).lower().startswith("basler"):
            return BaslerCameraAdapter(calibration)

    index = int(payload.get("cameraIndex", 0))
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Cannot open camera index {index}.")
    if data:
        resolution = data.get("calibration", {}).get("resolution", [])
        if len(resolution) == 2:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
    for _ in range(5):
        camera.read()
    return camera


def require_pose(payload: dict[str, Any]) -> dict[str, float]:
    pose = payload.get("robotPose")
    if not isinstance(pose, dict):
        raise ValueError("robotPose is required for every captured image.")
    clean = {}
    for key in ("x", "y", "z"):
        value = float(pose[key])
        if not np.isfinite(value):
            raise ValueError(f"robotPose.{key} must be finite.")
        clean[key] = value
    if "r" in pose:
        clean["r"] = float(pose["r"])
    return clean


def save_frame(frame: np.ndarray, payload: dict[str, Any], label: str, index: int | None = None) -> dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"_{index}" if index is not None else ""
    path = output_dir(payload) / f"{label}_{stamp}{suffix}.jpg"
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Failed to save camera image to {path}.")
    return {
        "path": str(path),
        "timestamp": utc_now(),
        "robotPose": require_pose(payload),
        "sharpness": sharpness(frame),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
    }


def capture_one(payload: dict[str, Any]) -> dict[str, Any]:
    camera = open_camera(payload)
    try:
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("Camera returned no frame.")
        result = save_frame(frame, payload, str(payload.get("label") or "capture"))
        return {"ok": True, **result}
    finally:
        camera.release()


def capture_burst(payload: dict[str, Any]) -> dict[str, Any]:
    count = 3
    interval = max(0.0, float(payload.get("burstIntervalMs", 350))) / 1000.0
    camera = open_camera(payload)
    captures = []
    try:
        for index in range(count):
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(f"Camera returned no frame for burst image {index + 1}.")
            captures.append(save_frame(frame, payload, "close", index + 1))
            if index + 1 < count:
                time.sleep(interval)
    finally:
        camera.release()
    selected = max(captures, key=lambda item: item["sharpness"])
    return {"ok": True, "captures": captures, "selected": selected}


def board_candidates(image: np.ndarray) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    image_area = float(height * width)

    # The current fixture uses a red PCB on a green ESD mat. Color isolation
    # keeps the black cable and dense component edges from breaking the board
    # outline. Fall back to the generic edge detector for other board colors.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_mask = (
        (((hue < 15) | (hue > 170)) & (saturation > 70) & (value > 35))
        .astype(np.uint8)
        * 255
    )
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        iterations=1,
    )
    red_contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    red_candidates = []
    for contour in red_contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if not 0.03 <= area_ratio <= 0.90:
            continue
        rect = cv2.minAreaRect(contour)
        box_area = float(rect[1][0] * rect[1][1])
        rectangularity = area / box_area if box_area > 0 else 0.0
        if rectangularity < 0.65:
            continue
        M = cv2.moments(contour)
        cx = float(M["m10"] / M["m00"]) if M["m00"] else float(rect[0][0])
        cy = float(M["m01"] / M["m00"]) if M["m00"] else float(rect[0][1])
        box = cv2.boxPoints(rect)
        red_candidates.append({
            "centerPixel": {"u": cx, "v": cy},
            "cornersPixel": [
                {"u": float(point[0]), "v": float(point[1])} for point in box
            ],
            "areaRatio": area_ratio,
            "rectangularity": rectangularity,
            "score": area_ratio * rectangularity,
            "detector": "red_pcb_hsv",
        })
    if red_candidates:
        return sorted(red_candidates, key=lambda item: item["score"], reverse=True)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if not 0.03 <= area_ratio <= 0.90:
            continue
        rect = cv2.minAreaRect(contour)
        box_area = float(rect[1][0] * rect[1][1])
        if box_area <= 0 or area / box_area < 0.60:
            continue
        box = cv2.boxPoints(rect)
        center = rect[0]
        candidates.append({
            "centerPixel": {"u": float(center[0]), "v": float(center[1])},
            "cornersPixel": [{"u": float(point[0]), "v": float(point[1])} for point in box],
            "areaRatio": area_ratio,
            "rectangularity": area / box_area,
            "score": area_ratio * (area / box_area),
            "detector": "edge_rectangle",
        })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)

def coarse_localize(payload: dict[str, Any]) -> dict[str, Any]:
    capture = payload.get("capture") or {}
    image_path = Path(str(capture.get("path") or ""))
    if not image_path.exists():
        raise ValueError("Global capture path does not exist.")
    robot_pose = capture.get("robotPose")
    if not robot_pose:
        raise ValueError("Global capture is missing its associated robot XYZ pose.")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Global capture cannot be decoded as an image.")
    candidates = board_candidates(image)
    if len(candidates) != 1:
        return {
            "ok": True,
            "status": "NOT_FOUND" if not candidates else "AMBIGUOUS",
            "candidateCount": len(candidates),
            "candidates": candidates,
        }

    mapper = PixelToWorld(str(calibration_path(payload)), robot_pose=robot_pose)
    center = candidates[0]["centerPixel"]
    point = mapper.pixel_to_table(center["u"], center["v"])
    if point is None:
        raise RuntimeError("PCB center ray does not intersect the configured table plane.")

    close_z = float(payload["closeZ"])
    hypothetical_close_pose = {
        "x": float(robot_pose["x"]),
        "y": float(robot_pose["y"]),
        "z": close_z,
        "r": float(robot_pose["r"]),
    }
    principal = (float(mapper.K[0, 2]), float(mapper.K[1, 2]))
    close_view_center = mapper.pixel_to_table(
        principal[0], principal[1], robot_pose=hypothetical_close_pose
    )
    if close_view_center is None:
        raise RuntimeError("Camera principal ray at close Z does not intersect the table plane.")

    # ── annotate global image with detected PCB ─────────────────────────
    annotated = image.copy()
    best = candidates[0]
    center_u, center_v = int(round(best["centerPixel"]["u"])), int(round(best["centerPixel"]["v"]))
    cv2.drawMarker(annotated, (center_u, center_v), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.circle(annotated, (center_u, center_v), 30, (0, 0, 255), 3)
    corners = [(int(round(p["u"])), int(round(p["v"]))) for p in best.get("cornersPixel", [])]
    if len(corners) == 4:
        cv2.polylines(annotated, [np.array(corners, dtype=np.int32)], True, (0, 255, 0), 3)
    cv2.putText(annotated, "PCB center", (center_u + 30, center_v - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(annotated, f"detector: {best['detector']}  score: {best['score']:.3f}",
                (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    annotated_path = image_path.parent / f"global_localized_{image_path.stem}.jpg"
    cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {
        "ok": True,
        "status": "UNIQUE",
        "candidateCount": 1,
        "candidate": candidates[0],
        "basePoint": {"x": point[0], "y": point[1], "z": point[2]},
        "recommendedEndXY": {
            "x": float(robot_pose["x"]) + point[0] - close_view_center[0] + 15,
            "y": float(robot_pose["y"]) + point[1] - close_view_center[1],
        },
        "robotPose": robot_pose,
        "sourcePath": str(image_path),
        "annotatedPath": str(annotated_path),
    }


def pixel_to_base(payload: dict[str, Any]) -> dict[str, Any]:
    pixel = payload.get("pixel") or {}
    if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
        u, v = float(pixel[0]), float(pixel[1])
    else:
        u, v = float(pixel["x"]), float(pixel["y"])
    if not np.isfinite([u, v]).all():
        raise ValueError("pixel x/y must be finite.")
    robot_pose = require_pose(payload)
    mapper = PixelToWorld(str(calibration_path(payload)), robot_pose=robot_pose)
    point = mapper.pixel_to_table(u, v)
    if point is None:
        raise RuntimeError("Selected close-image pixel does not intersect the calibrated PCB plane.")
    return {
        "ok": True,
        "pixel": {"x": u, "y": v},
        "robotPose": robot_pose,
        "basePoint": {"x": point[0], "y": point[1], "z": point[2]},
    }


def camera_center_offset(payload: dict[str, Any]) -> dict[str, Any]:
    """Offset to add to a target table point so the camera view centers on it.

    Derived purely from calibration: casts the image-center ray from the given
    robot Z, intersects the table plane, and reports (robot_xy - view_center_xy).
    Replaces the old hand-tuned probe->camera offset.
    """
    robot_pose = require_pose(payload)
    path = calibration_path(payload)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    width, height = data.get("calibration", {}).get("resolution", [2448, 2048])
    mapper = PixelToWorld(str(path), robot_pose=robot_pose)
    center = mapper.pixel_to_table(float(width) / 2.0, float(height) / 2.0)
    if center is None:
        raise RuntimeError("Image-center ray does not intersect the calibrated table plane.")
    dx = float(robot_pose["x"]) - float(center[0])
    dy = float(robot_pose["y"]) - float(center[1])
    return {
        "ok": True,
        "robotPose": robot_pose,
        "viewCenter": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "offset": {"dx": dx, "dy": dy},
    }


def health(payload: dict[str, Any]) -> dict[str, Any]:
    path = calibration_path(payload)
    status = calibration_status(payload)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    table_plane_configured = "table_z_mm" in (data or {}).get("table_homography", {})
    correction = (data or {}).get("xy_pose_correction", {})
    runtime_safety = (data or {}).get("runtime_safety", {})
    camera_ready = False
    if status == "calibrated" and table_plane_configured:
        camera = open_camera(payload)
        try:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("Camera health check returned no frame.")
            camera_ready = True
        finally:
            camera.release()
    return {
        "ok": True,
        "cameraIndex": int(payload.get("cameraIndex", 0)),
        "cameraReady": camera_ready,
        "calibrationFile": str(path),
        "calibrationStatus": status,
        "tablePlaneConfigured": table_plane_configured,
        "mountMode": "eye_in_hand_xyz",
        "robotAxesUsed": ["x", "y", "z"],
        "robotAxesIgnored": ["r"],
        "fixedR": correction.get("fixed_r_deg", runtime_safety.get("fixed_r_deg")),
        "rToleranceDeg": correction.get("r_tolerance_deg"),
        "validRobotXYZMin": correction.get("valid_robot_xyz_min"),
        "validRobotXYZMax": correction.get("valid_robot_xyz_max"),
        "safeHeightZ": runtime_safety.get("safe_height_z_mm"),
    }


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    payload = read_payload()
    actions = {
        "health": health,
        "capture": capture_one,
        "capture-burst": capture_burst,
        "coarse-localize": coarse_localize,
        "pixel-to-base": pixel_to_base,
        "camera-center-offset": camera_center_offset,
    }
    if action not in actions:
        raise ValueError(f"Unsupported camera action: {action}")
    print(json.dumps(actions[action](payload), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error), "type": type(error).__name__}, ensure_ascii=False))
        raise SystemExit(1)
