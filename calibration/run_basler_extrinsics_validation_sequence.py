"""Collect independent fixed-R validation frames without changing fit samples."""

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
SAFE_POSE = {"x": 345.5, "y": -40.8, "z": 80.0, "r": FIXED_R}
TARGETS = [
    {"x": 337.5, "y": -33.3, "z": 75.0, "r": FIXED_R},
    {"x": 352.5, "y": -48.3, "z": 75.0, "r": FIXED_R},
    {"x": 337.5, "y": -48.3, "z": 82.5, "r": FIXED_R},
]


def _load_bridge():
    spec = importlib.util.spec_from_file_location("mg400_bridge_validation", BRIDGE_PATH)
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
    config.update({"speed": 15, "autoEnable": False, "returnHome": False})
    completed = 0
    try:
        for index, target in enumerate(TARGETS, start=1):
            print(json.dumps({"event": "move_start", "index": index, "target": target}), flush=True)
            actual = _move(bridge, config, target)
            print(json.dumps({"event": "move_complete", "index": index, "pose": actual}), flush=True)
            time.sleep(1.0)
            capture_sample.main(
                [
                    "--dataset-file",
                    "validation_samples.json",
                    "--image-prefix",
                    "validation",
                ]
            )
            completed += 1
    finally:
        actual = _move(bridge, config, SAFE_POSE)
        print(json.dumps({"event": "safe_pose", "pose": actual}), flush=True)
    print(json.dumps({"event": "validation_complete", "samples": completed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
