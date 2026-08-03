"""Run the explicitly authorized fixed-R external-calibration sample sequence."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import capture_basler_extrinsics_sample as capture_sample


HERE = Path(__file__).resolve().parent
BRIDGE_PATH = HERE.parent / "Inputdemo" / "scripts" / "mg400_bridge.py"
ROBOT_CONFIG_PATH = HERE.parent / "Inputdemo" / "config" / "mg400.json"
FIXED_R = 7.686619
TARGETS = [
    (345.5, -40.8, 47.5),
    (330.5, -40.8, 47.5),
    (330.5, -25.8, 47.5),
    (330.5, -55.8, 47.5),
    (345.5, -25.8, 47.5),
    (345.5, -55.8, 47.5),
    (330.5, -40.8, 65.0),
    (345.5, -40.8, 65.0),
    (360.5, -25.8, 65.0),
    (345.5, -40.8, 80.0),
]
SAFE_POSE = {"x": 345.5, "y": -40.8, "z": 80.0, "r": FIXED_R}


def _load_bridge():
    spec = importlib.util.spec_from_file_location("mg400_bridge_sequence", BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BRIDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _move(bridge, config: dict, pose: dict) -> dict:
    payload = {
        "config": config,
        "pose": pose,
        "trajectory": {
            "mode": "safe-lift-traverse-descend",
            "safeTravelZ": 90,
            "travelSpeed": 15,
            "descentSpeed": 8,
        },
    }
    result = bridge.action_execute(payload)
    actual = result["robot"]["pose"]
    error = max(abs(float(actual[key]) - float(pose[key])) for key in pose)
    if result["robot"]["mode"]["code"] != 5 or error > 0.02:
        raise RuntimeError(f"Robot did not settle: pose={actual}, error={error}")
    return actual


def main() -> int:
    bridge = _load_bridge()
    config = json.loads(ROBOT_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    config.update(
        {
            "speed": 15,
            "autoEnable": False,
            "returnHome": False,
        }
    )
    session_dir = Path(
        capture_sample.ACTIVE_EXTRINSICS_FILE.read_text(encoding="utf-8-sig").strip()
    )
    dataset_path = session_dir / "samples.json"
    existing = 0
    if dataset_path.exists():
        existing = len(json.loads(dataset_path.read_text(encoding="utf-8-sig"))["samples"])
    if existing > len(TARGETS):
        raise RuntimeError(f"Dataset already has {existing} samples; expected at most {len(TARGETS)}")
    if existing == len(TARGETS):
        print(json.dumps({"event": "sequence_already_complete", "samples": existing}))
        return 0
    completed = 0
    try:
        for target_index, (x, y, z) in enumerate(TARGETS[existing:], start=existing + 1):
            target = {"x": x, "y": y, "z": z, "r": FIXED_R}
            print(json.dumps({"event": "move_start", "sample": target_index, "target": target}), flush=True)
            pose = _move(bridge, config, target)
            print(json.dumps({"event": "move_complete", "sample": target_index, "pose": pose}), flush=True)
            time.sleep(1.0)
            capture_sample.main([])
            completed += 1
            print(json.dumps({"event": "sample_complete", "sample": target_index}), flush=True)
    finally:
        pose = _move(bridge, config, SAFE_POSE)
        print(json.dumps({"event": "safe_pose", "pose": pose}), flush=True)
    print(json.dumps({"event": "sequence_complete", "new_samples": completed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
