"""Collect the authorized fixed-R, multi-height 5 mm-board samples."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import capture_basler_extrinsics_sample as capture_sample


HERE = Path(__file__).resolve().parent
BRIDGE_PATH = HERE.parent / "Inputdemo" / "scripts" / "mg400_bridge.py"
ROBOT_CONFIG_PATH = HERE.parent / "Inputdemo" / "config" / "mg400.json"
SESSION_NAME = "basler_close_5mm_20260803"
FIXED_R = 7.686619
SAFE_Z = 50.0
STAGING_X = 315.0
BASE_X = 363.285553
BASE_Y = -20.915600

# Sample 1 was captured at the first target before this sequence was started.
# Every XY change is performed through SAFE_Z by mg400_bridge.
TARGETS = [
    (BASE_X, BASE_Y, -43.59),
    (373.285553, BASE_Y, -43.59),
    (383.285553, -10.915600, -43.59),
    (373.285553, -30.915600, -43.59),
    (340.0, BASE_Y, 25.0),
    (340.0, -0.915600, 25.0),
    (340.0, 9.084400, 25.0),
    (350.0, BASE_Y, 50.0),
    (350.0, -0.915600, 50.0),
    (350.0, 19.084400, 50.0),
]
SAFE_POSE = {"x": STAGING_X, "y": BASE_Y, "z": SAFE_Z, "r": FIXED_R}


def _load_bridge():
    spec = importlib.util.spec_from_file_location("mg400_bridge_close_5mm", BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BRIDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move(bridge, config: dict, pose: dict, safe_travel_z: float) -> dict:
    payload = {
        "config": config,
        "pose": pose,
        "trajectory": {
            "mode": "safe-lift-traverse-descend",
            "safeTravelZ": safe_travel_z,
            "travelSpeed": 12,
            "descentSpeed": 6,
        },
    }
    result = bridge.action_execute(payload)
    actual = result["robot"]["pose"]
    error = max(abs(float(actual[key]) - float(pose[key])) for key in pose)
    if result["robot"]["mode"]["code"] != 5 or error > 0.03:
        raise RuntimeError(f"Robot did not settle: pose={actual}, error={error}")
    return actual


def _move_via_staging(bridge, config: dict, target: dict) -> dict:
    """Cross the narrow Z=10 mm envelope only at the safe inner radius."""
    status = bridge.action_status({"config": config})["robot"]
    if status["mode"]["code"] != 5:
        raise RuntimeError(
            f"MG400 must be ENABLED_IDLE (5), got {status['mode']}"
        )
    current = status["pose"]
    current_z = float(current["z"])
    staging_here = {
        "x": STAGING_X,
        "y": BASE_Y,
        "z": current_z,
        "r": FIXED_R,
    }
    _move(bridge, config, staging_here, current_z)

    target_z = float(target["z"])
    staging_target = {**staging_here, "z": target_z}
    _move(bridge, config, staging_target, max(current_z, target_z))
    return _move(bridge, config, target, target_z)


def _validate_active_session() -> tuple[Path, int]:
    session_dir = Path(
        capture_sample.ACTIVE_EXTRINSICS_FILE.read_text(
            encoding="utf-8-sig"
        ).strip()
    )
    if session_dir.name != SESSION_NAME:
        raise RuntimeError(f"Refusing to use unexpected session: {session_dir}")
    session = json.loads(
        (session_dir / "session.json").read_text(encoding="utf-8-sig")
    )
    if float(session["board"]["square_size_mm"]) != 5.0:
        raise RuntimeError("Active session is not configured for the 5 mm board")
    dataset_path = session_dir / "samples.json"
    existing = 0
    if dataset_path.exists():
        existing = len(
            json.loads(dataset_path.read_text(encoding="utf-8-sig"))["samples"]
        )
    if not 1 <= existing <= len(TARGETS):
        raise RuntimeError(
            f"Expected 1..{len(TARGETS)} existing samples, got {existing}"
        )
    return session_dir, existing


def main() -> int:
    _, existing = _validate_active_session()
    if existing == len(TARGETS):
        print(json.dumps({"event": "sequence_already_complete", "samples": existing}))
        return 0

    bridge = _load_bridge()
    config = json.loads(ROBOT_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    config.update({"speed": 12, "autoEnable": False, "returnHome": False})
    completed = 0
    try:
        for index, (x, y, z) in enumerate(TARGETS[existing:], start=existing + 1):
            target = {"x": x, "y": y, "z": z, "r": FIXED_R}
            print(
                json.dumps({"event": "move_start", "sample": index, "target": target}),
                flush=True,
            )
            actual = _move_via_staging(bridge, config, target)
            print(
                json.dumps({"event": "move_complete", "sample": index, "pose": actual}),
                flush=True,
            )
            time.sleep(1.0)
            capture_sample.main(
                ["--dataset-file", "samples.json", "--image-prefix", "xyz5mm"]
            )
            completed += 1
            print(json.dumps({"event": "sample_complete", "sample": index}), flush=True)
    finally:
        actual = _move_via_staging(bridge, config, SAFE_POSE)
        print(json.dumps({"event": "safe_pose", "pose": actual}), flush=True)
    print(json.dumps({"event": "sequence_complete", "new_samples": completed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
