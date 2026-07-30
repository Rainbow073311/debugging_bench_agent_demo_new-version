"""Read-only verification for XYZ-only eye-in-hand pixel projection.

The script opens the camera and reads the MG400 status, but it never sends a
motion command.  Any Z motion remains the responsibility of the existing safe
trajectory/probe-threshold workflow.
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
import requests

from coordinate_transforms import PixelToWorld


CAMERA_INDEX = 1
CAMERA_RESOLUTION = (2560, 1440)
ROBOT_GATEWAY_URL = "http://127.0.0.1:8010"
ARUCO_DICT = cv2.aruco.DICT_6X6_250
ARUCO_ID = 0

MG400_CONFIG = {
    "mode": "mg400",
    "ip": "192.168.2.6",
    "dashboardPort": 29999,
    "motionPort": 30003,
    "feedbackPort": 30004,
    "speed": 30,
    "timeoutMs": 5000,
    "autoEnable": True,
    "motionCommand": "MovJ",
}


def get_robot_status():
    response = requests.post(
        f"{ROBOT_GATEWAY_URL}/v1/robot/status",
        json={"config": MG400_CONFIG},
        timeout=5,
    )
    response.raise_for_status()
    data = response.json()
    robot = data.get("robot", {})
    pose = robot.get("pose", {})
    if not all(key in pose for key in ("x", "y", "z")):
        raise RuntimeError("robot status did not include X/Y/Z")
    return robot


def detect_aruco_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or ARUCO_ID not in ids:
        return None
    index = list(ids.flatten()).index(ARUCO_ID)
    points = corners[index].reshape(4, 2)
    return float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))


def main():
    config_path = os.path.join(os.path.dirname(__file__), "camera_config.yaml")
    converter = PixelToWorld(config_path)

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    if not camera.isOpened():
        raise RuntimeError("camera could not be opened")

    try:
        frame = None
        for _ in range(40):
            ok, candidate = camera.read()
            if ok:
                frame = candidate
            time.sleep(0.02)
        if frame is None:
            raise RuntimeError("camera did not return a frame")

        # Bind the robot pose read immediately after the selected image.
        robot = get_robot_status()
        mode = robot.get("mode", {})
        if mode.get("code") != 5:
            raise RuntimeError(
                "robot must be ENABLED_IDLE for a stable calibration image"
            )
        pose = robot["pose"]

        pixel = detect_aruco_center(frame)
        if pixel is None:
            cv2.imwrite("verify_debug.jpg", cv2.resize(frame, (1280, 720)))
            raise RuntimeError("ArUco marker 0 was not detected")

        world = converter.pixel_to_table(pixel[0], pixel[1], robot_pose=pose)
        if world is None:
            raise RuntimeError("camera ray did not intersect the configured table plane")

        print(f"robot XYZ: {pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}")
        print(f"robot R ignored: {pose.get('r', 'not provided')}")
        print(f"pixel: {pixel[0]:.1f}, {pixel[1]:.1f}")
        print(f"table coordinate: {world[0]:.3f}, {world[1]:.3f}, {world[2]:.3f}")
        print("read-only verification complete; no robot motion command was sent")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
