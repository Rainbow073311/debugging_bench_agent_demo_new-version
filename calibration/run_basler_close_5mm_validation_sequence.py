"""Collect three independent holdout frames for the 5 mm-board model."""

from __future__ import annotations

import json
import time

import capture_basler_extrinsics_sample as capture_sample
import run_basler_close_5mm_extrinsics_sequence as sequence


TARGETS = [
    {"x": 378.285553, "y": -15.915600, "z": -43.59, "r": sequence.FIXED_R},
    {"x": 340.0, "y": 4.084400, "z": 25.0, "r": sequence.FIXED_R},
    {"x": 350.0, "y": 9.084400, "z": 50.0, "r": sequence.FIXED_R},
]


def main() -> int:
    session_dir, _ = sequence._validate_active_session()
    dataset_path = session_dir / "validation_samples.json"
    existing = 0
    if dataset_path.exists():
        existing = len(
            json.loads(dataset_path.read_text(encoding="utf-8-sig"))["samples"]
        )
    if existing > len(TARGETS):
        raise RuntimeError(f"Unexpected validation sample count: {existing}")
    if existing == len(TARGETS):
        print(json.dumps({"event": "validation_already_complete", "samples": existing}))
        return 0

    bridge = sequence._load_bridge()
    config = json.loads(
        sequence.ROBOT_CONFIG_PATH.read_text(encoding="utf-8-sig")
    )
    config.update({"speed": 12, "autoEnable": False, "returnHome": False})
    completed = 0
    try:
        for index, target in enumerate(TARGETS[existing:], start=existing + 1):
            print(
                json.dumps({"event": "move_start", "sample": index, "target": target}),
                flush=True,
            )
            actual = sequence._move_via_staging(bridge, config, target)
            print(
                json.dumps({"event": "move_complete", "sample": index, "pose": actual}),
                flush=True,
            )
            time.sleep(1.0)
            capture_sample.main(
                [
                    "--dataset-file",
                    "validation_samples.json",
                    "--image-prefix",
                    "validation5mm",
                ]
            )
            completed += 1
    finally:
        actual = sequence._move_via_staging(
            bridge, config, sequence.SAFE_POSE
        )
        print(json.dumps({"event": "safe_pose", "pose": actual}), flush=True)
    print(json.dumps({"event": "validation_complete", "new_samples": completed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
