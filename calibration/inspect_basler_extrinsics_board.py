"""Capture a read-only board image and label the four P0 candidates."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from capture_basler_intrinsics import (
    ACTIVE_SESSION_FILE as ACTIVE_INTRINSICS_FILE,
    _capture,
    _detect,
    _open_camera,
)


HERE = Path(__file__).resolve().parent
ACTIVE_EXTRINSICS_FILE = (
    HERE / "calibration_sessions" / ".active_basler_extrinsics_session"
)


def main() -> int:
    intrinsic_dir = Path(
        ACTIVE_INTRINSICS_FILE.read_text(encoding="utf-8-sig").strip()
    )
    intrinsic_session = json.loads(
        (intrinsic_dir / "session.json").read_text(encoding="utf-8-sig")
    )
    extrinsic_dir = Path(
        ACTIVE_EXTRINSICS_FILE.read_text(encoding="utf-8-sig").strip()
    )
    pattern = tuple(
        int(value) for value in intrinsic_session["pattern_inner_corners"]
    )

    camera = _open_camera(intrinsic_session)
    try:
        frame = _capture(camera)
    finally:
        camera.Close()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners, detector = _detect(gray, pattern)
    if not found or corners is None:
        raw_path = extrinsic_dir / "board_orientation_check_failed.jpg"
        cv2.imwrite(str(raw_path), frame)
        raise RuntimeError(f"Chessboard corners not detected; raw frame saved to {raw_path}")

    grid = corners.reshape(pattern[1], pattern[0], 2)
    inner_object = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    homography, _ = cv2.findHomography(
        inner_object.astype(np.float32), corners.astype(np.float32)
    )
    if homography is None:
        raise RuntimeError("Could not extrapolate the chessboard outer boundary")
    # The printed 10x7-square board extends one full square beyond the 9x6
    # inner-corner grid on every side.
    outer_object = np.asarray(
        [[-1, -1], [pattern[0], -1], [-1, pattern[1]], [pattern[0], pattern[1]]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    outer_image = cv2.perspectiveTransform(outer_object, homography).reshape(-1, 2)
    candidates = dict(zip(("1", "2", "3", "4"), outer_image))
    annotated = frame.copy()
    cv2.drawChessboardCorners(
        annotated, pattern, corners.reshape(-1, 1, 2), True
    )
    colors = {
        "1": (0, 0, 255),
        "2": (0, 255, 0),
        "3": (255, 0, 0),
        "4": (0, 255, 255),
    }
    for label, point in candidates.items():
        x, y = (int(round(value)) for value in point)
        cv2.circle(annotated, (x, y), 24, colors[label], 6)
        cv2.putText(
            annotated,
            label,
            (x + 30, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            colors[label],
            5,
            cv2.LINE_AA,
        )
    output = extrinsic_dir / "board_orientation_check.jpg"
    cv2.imwrite(str(output), annotated, [cv2.IMWRITE_JPEG_QUALITY, 96])
    metadata_path = extrinsic_dir / "board_orientation_check.json"
    previous = {}
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    metadata = {
        "image": output.name,
        "detector": detector,
        "pattern": list(pattern),
        "candidates_px": {
            label: [float(point[0]), float(point[1])]
            for label, point in candidates.items()
        },
        "status": previous.get("status", "awaiting_user_p0_label"),
    }
    if previous.get("status") == "confirmed":
        metadata["p0_outer_label"] = previous["p0_outer_label"]
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "path": str(output), **metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
