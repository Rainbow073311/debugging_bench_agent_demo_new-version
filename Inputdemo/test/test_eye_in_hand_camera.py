import importlib.util
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eye_in_hand_camera.py"
SPEC = importlib.util.spec_from_file_location("eye_in_hand_camera_test", SCRIPT)
camera_bridge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(camera_bridge)


def test_red_pcb_detector_ignores_green_background_and_black_cable():
    image = np.full((800, 1000, 3), (90, 120, 70), dtype=np.uint8)
    cv2.rectangle(image, (220, 160), (780, 620), (30, 30, 190), -1)
    cv2.rectangle(image, (470, 620), (530, 799), (20, 20, 20), -1)

    candidates = camera_bridge.board_candidates(image)

    assert len(candidates) == 1
    assert candidates[0]["detector"] == "red_pcb_hsv"
    assert abs(candidates[0]["centerPixel"]["u"] - 500) < 5
    assert abs(candidates[0]["centerPixel"]["v"] - 390) < 5
