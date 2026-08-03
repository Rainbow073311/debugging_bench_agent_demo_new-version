"""Capture one synchronized Basler/MG400 XYZ eye-in-hand sample."""

from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

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
ROBOT_CONFIG_FILE = HERE.parent / "Inputdemo" / "config" / "mg400.json"
CAMERA_CONFIG_FILE = HERE / "camera_config.yaml"


def _dashboard(command: str, config: dict) -> str:
    timeout = float(config.get("timeoutMs", 5000)) / 1000.0
    with socket.create_connection(
        (config["ip"], int(config.get("dashboardPort", 29999))), timeout=timeout
    ) as connection:
        connection.sendall((command + "\r\n").encode("utf-8"))
        return connection.recv(2048).decode("utf-8", errors="replace").strip()


def _payload(response: str) -> list[str]:
    if not response.startswith("0,"):
        raise RuntimeError(f"MG400 command failed: {response}")
    try:
        return response.split(",{", 1)[1].split("}", 1)[0].split(",")
    except IndexError as error:
        raise RuntimeError(f"Could not parse MG400 response: {response}") from error


def _robot_status(config: dict) -> dict:
    mode_response = _dashboard("RobotMode()", config)
    mode = int(_payload(mode_response)[0])
    if mode != 5:
        raise RuntimeError(f"MG400 must be ENABLED_IDLE (5), got mode {mode}")
    pose_response = _dashboard("GetPose()", config)
    values = [float(value) for value in _payload(pose_response)[:4]]
    return {
        "mode": "ENABLED_IDLE",
        "pose": dict(zip(("x", "y", "z", "r"), values)),
        "raw": {"mode": mode_response, "pose": pose_response},
    }


def _canonical_grid(
    corners: np.ndarray,
    pattern: tuple[int, int],
    orientation: dict,
) -> np.ndarray:
    grid = corners.reshape(pattern[1], pattern[0], 2)
    candidate = orientation["candidates_px"]
    reference_x = np.asarray(candidate["2"]) - np.asarray(candidate["1"])
    reference_y = np.asarray(candidate["3"]) - np.asarray(candidate["1"])
    current_x = grid[0, -1] - grid[0, 0]
    current_y = grid[-1, 0] - grid[0, 0]
    score = float(np.dot(reference_x, current_x) + np.dot(reference_y, current_y))
    reverse_score = float(np.dot(reference_x, -current_x) + np.dot(reference_y, -current_y))
    if reverse_score > score:
        grid = grid[::-1, ::-1]
    return grid


def _object_points(pattern: tuple[int, int], square_mm: float) -> np.ndarray:
    # Confirmed outer P0 is label 3. Detector +X points 1->2; detector +Y
    # points 1->3. Board +X is 3->4 and board +Y is 3->1.
    detector = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    points = np.zeros((len(detector), 3), dtype=np.float32)
    points[:, 0] = (detector[:, 0] + 1.0) * square_mm
    points[:, 1] = (pattern[1] - detector[:, 1]) * square_mm
    return points


def _pose_delta(first: dict, second: dict) -> dict:
    return {
        key: abs(float(second[key]) - float(first[key]))
        for key in ("x", "y", "z", "r")
    }


def _load_extrinsic_session(extrinsic_dir: Path) -> tuple[dict, dict]:
    """Load geometry owned by this extrinsic session.

    A session-local board definition prevents a 5 mm board from silently
    inheriting the 10 mm board used for intrinsic calibration.
    """
    session_file = extrinsic_dir / "session.json"
    if session_file.exists():
        session = json.loads(session_file.read_text(encoding="utf-8-sig"))
        return session, dict(session["orientation"])

    # Backward compatibility for the completed legacy 10 mm session.
    reference = json.loads(
        (extrinsic_dir / "reference_points.json").read_text(encoding="utf-8-sig")
    )
    orientation = json.loads(
        (extrinsic_dir / "board_orientation_check.json").read_text(
            encoding="utf-8-sig"
        )
    )
    return {
        "board": reference["board"],
        "coordinate_convention": reference["coordinate_convention"],
        "board_pose_in_base_operational": reference[
            "board_pose_in_base_operational"
        ],
        "camera_serial": reference["camera_serial"],
    }, orientation


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Capture one synchronized fixed-R Basler/MG400 sample"
    )
    parser.add_argument("--dataset-file", default="samples.json")
    parser.add_argument("--image-prefix", default="extrinsic")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    extrinsic_dir = Path(
        ACTIVE_EXTRINSICS_FILE.read_text(encoding="utf-8-sig").strip()
    )
    intrinsic_dir = Path(
        ACTIVE_INTRINSICS_FILE.read_text(encoding="utf-8-sig").strip()
    )
    intrinsic_session = json.loads(
        (intrinsic_dir / "session.json").read_text(encoding="utf-8-sig")
    )
    extrinsic_session, orientation = _load_extrinsic_session(extrinsic_dir)
    if orientation.get("status") != "confirmed" or orientation.get("p0_outer_label") != "3":
        raise RuntimeError("Outer-board P0 orientation has not been confirmed as label 3")
    camera_config = yaml.safe_load(CAMERA_CONFIG_FILE.read_text(encoding="utf-8-sig"))
    matrix = np.asarray(camera_config["intrinsics"]["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(camera_config["intrinsics"]["dist_coeffs"], dtype=np.float64)
    robot_config = json.loads(ROBOT_CONFIG_FILE.read_text(encoding="utf-8-sig"))
    camera_serial = str(extrinsic_session["camera_serial"])
    if str(intrinsic_session["serial"]) != camera_serial:
        raise RuntimeError(
            "Intrinsic and extrinsic sessions refer to different cameras: "
            f"{intrinsic_session['serial']} != {camera_serial}"
        )
    board = extrinsic_session["board"]
    pattern = tuple(int(value) for value in board["pattern_inner_corners"])
    object_points = _object_points(pattern, float(board["square_size_mm"]))

    before = _robot_status(robot_config)
    camera = _open_camera(intrinsic_session)
    try:
        frame = _capture(camera)
    finally:
        camera.Close()
    after = _robot_status(robot_config)
    delta = _pose_delta(before["pose"], after["pose"])
    if max(delta[key] for key in ("x", "y", "z")) > 0.05 or delta["r"] > 0.05:
        raise RuntimeError(f"Robot moved during exposure: {delta}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners, detector = _detect(gray, pattern)
    if not found or corners is None:
        raise RuntimeError("Full 9x6 chessboard was not detected; sample not saved")
    grid = _canonical_grid(corners, pattern, orientation)
    image_points = grid.reshape(-1, 1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed; sample not saved")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)
    residual = image_points.reshape(-1, 2) - projected.reshape(-1, 2)
    reprojection_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if reprojection_rmse > 1.5:
        raise RuntimeError(
            f"PnP reprojection RMSE {reprojection_rmse:.3f}px exceeds 1.5px"
        )

    if Path(args.dataset_file).name != args.dataset_file:
        raise ValueError("--dataset-file must be a filename inside the active session")
    if not args.image_prefix.replace("_", "").isalnum():
        raise ValueError("--image-prefix may contain only letters, digits and underscores")
    dataset_file = extrinsic_dir / args.dataset_file
    if dataset_file.exists():
        dataset = json.loads(dataset_file.read_text(encoding="utf-8-sig"))
    else:
        source_file = extrinsic_dir / "samples.json"
        if source_file.exists():
            source = json.loads(source_file.read_text(encoding="utf-8-sig"))
            dataset = {
                key: source[key]
                for key in (
                    "target_plane_z_mm",
                    "marker_pose_in_base",
                    "board_coordinate_convention",
                    "camera_serial",
                    "fixed_r_reference_deg",
                )
            }
            dataset.update({"status": "validation", "samples": []})
        else:
            board_pose = extrinsic_session["board_pose_in_base_operational"]
            dataset = {
                "status": "collecting",
                "target_plane_z_mm": float(board_pose["translation_mm"][2]),
                "marker_pose_in_base": {
                    "translation_mm": board_pose["translation_mm"],
                    "R": board_pose["R"],
                },
                "board": board,
                "board_coordinate_convention": extrinsic_session[
                    "coordinate_convention"
                ],
                "camera_serial": camera_serial,
                "fixed_r_reference_deg": float(
                    extrinsic_session.get("fixed_r_reference_deg", before["pose"]["r"])
                ),
                "samples": [],
            }
    fixed_r = float(dataset["fixed_r_reference_deg"])
    r_difference = abs(((float(before["pose"]["r"]) - fixed_r + 180.0) % 360.0) - 180.0)
    if r_difference > 0.5:
        raise RuntimeError(
            f"R changed by {r_difference:.3f} deg from fixed reference {fixed_r:.3f}; sample not saved"
        )

    index = len(dataset["samples"]) + 1
    filename = f"{args.image_prefix}_{index:03d}.jpg"
    image_path = extrinsic_dir / filename
    if image_path.exists():
        raise RuntimeError(f"Refusing to overwrite {image_path}")
    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
        raise RuntimeError(f"Could not save {image_path}")
    sample = {
        "file": filename,
        "captured_at": datetime.now().astimezone().isoformat(),
        "robot_pose": before["pose"],
        "robot_pose_after": after["pose"],
        "pose_delta_during_capture": delta,
        "marker_pose_in_camera": {
            "rvec": rvec.reshape(-1).tolist(),
            "t_mm": tvec.reshape(-1).tolist(),
        },
        "detector": detector,
        "reprojection_rmse_px": reprojection_rmse,
    }
    dataset["samples"].append(sample)
    dataset_file.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "accepted": True,
                "sample_index": index,
                "image": str(image_path),
                "robot_pose": before["pose"],
                "reprojection_rmse_px": reprojection_rmse,
                "marker_t_camera_mm": tvec.reshape(-1).tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
