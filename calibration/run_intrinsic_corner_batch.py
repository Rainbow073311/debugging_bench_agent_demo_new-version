"""Push the chessboard to image corners/edges for distortion coverage."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_basler_intrinsics import _save_pending  # noqa: E402
from run_intrinsic_tilt_batch import (  # noqa: E402
    FIXED_R,
    PATTERN,
    _bridge,
    _config,
    ensure_session,
    find_board,
    grab_frame,
    move,
)

IMAGE_W, IMAGE_H = 2448, 2048
# Board center targets in the image: 4 corners + 4 edge midpoints.
TARGETS_PX = [
    (500, 500),
    (1948, 500),
    (500, 1548),
    (1948, 1548),
    (1224, 450),
    (1224, 1598),
    (430, 1024),
    (2018, 1024),
]
RADIUS_MIN, RADIUS_MAX = 255.0, 350.0


def clamp_radius(x: float, y: float) -> tuple[float, float]:
    radius = (x * x + y * y) ** 0.5
    if radius < RADIUS_MIN:
        s = RADIUS_MIN / radius
        return x * s, y * s
    if radius > RADIUS_MAX:
        s = RADIUS_MAX / radius
        return x * s, y * s
    return x, y


def main() -> int:
    base_z = float(sys.argv[1]) if len(sys.argv) > 1 else 22.0
    session_dir, session, session_file = ensure_session()
    batch_index = len(session.get("batches", [])) + 1

    bridge = _bridge()
    config = _config()
    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))["calibration"]

    # Start from an inner-radius base so +X/+Y offsets stay reachable.
    start = {"x": 305.0, "y": -8.0, "z": base_z, "r": FIXED_R}
    move(bridge, config, start)
    time.sleep(0.8)
    frame = grab_frame(cam_cfg)
    corners, det, metrics = find_board(frame)
    if corners is None:
        fail = session_dir / f"batch{batch_index:02d}_probe_fail.jpg"
        cv2.imwrite(str(fail), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(json.dumps({"event": "no_board", "saved": str(fail)}), flush=True)
        return 2

    center = corners.mean(axis=0)
    spacing = max(float(metrics["corner_spacing_px"]), 1.0)
    mm_per_px = 5.0 / spacing
    print(
        json.dumps(
            {
                "event": "board_found",
                "detector": det,
                "center_px": [float(center[0]), float(center[1])],
                "spacing_px": spacing,
            }
        ),
        flush=True,
    )

    batch = {
        "batch": batch_index,
        "kind": "corner_coverage",
        "started_at": datetime.now().astimezone().isoformat(),
        "base": start,
        "shots": [],
    }
    saved = 0
    for i, (ut, vt) in enumerate(TARGETS_PX, 1):
        # Robot +X moves the board toward -u in the image; same for Y/v.
        dx = -(ut - float(center[0])) * mm_per_px
        dy = -(vt - float(center[1])) * mm_per_px
        tx, ty = clamp_radius(start["x"] + dx, start["y"] + dy)
        target = {"x": round(tx, 2), "y": round(ty, 2), "z": base_z, "r": FIXED_R}
        print(json.dumps({"event": "move", "shot": i, "target_px": [ut, vt], "target": target}), flush=True)
        try:
            actual = move(bridge, config, target)
        except Exception as exc:
            batch["shots"].append({"shot": i, "target": target, "status": "move_fail", "error": str(exc)})
            print(json.dumps({"event": "move_fail", "shot": i, "error": str(exc)[:160]}), flush=True)
            continue
        time.sleep(1.0)
        frame = grab_frame(cam_cfg)
        corners2, det2, metrics2 = find_board(frame)
        path = _save_pending(session_dir, frame)
        entry = {"shot": i, "file": path.name, "pose": actual, "found": corners2 is not None}
        if metrics2:
            entry["metrics"] = {
                k: metrics2[k]
                for k in ("center_norm", "full_board_visible", "sharpness", "corner_spacing_px")
            }
            saved += 1
            print(
                json.dumps(
                    {
                        "event": "saved",
                        "shot": i,
                        "file": path.name,
                        "center_norm": metrics2["center_norm"],
                        "full_board_visible": metrics2["full_board_visible"],
                        "sharpness": metrics2["sharpness"],
                    }
                ),
                flush=True,
            )
        else:
            print(json.dumps({"event": "saved_no_board", "shot": i, "file": path.name}), flush=True)
        batch["shots"].append(entry)

    batch["finished_at"] = datetime.now().astimezone().isoformat()
    batch["saved_with_board"] = saved
    session.setdefault("batches", []).append(batch)
    session_file.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (session_dir / f"batch_{batch_index:02d}.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        move(bridge, config, {"x": 320.0, "y": -8.0, "z": 35.0, "r": FIXED_R})
    except Exception:
        pass
    print(
        json.dumps(
            {
                "event": "batch_done",
                "batch": batch_index,
                "saved_with_board": saved,
                "total_pending": len(
                    [f for f in (session_dir / "pending").glob("pending_*.jpg") if "_marked" not in f.name]
                ),
            }
        ),
        flush=True,
    )
    return 0 if saved >= 5 else 3


if __name__ == "__main__":
    raise SystemExit(main())
