import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_basler_intrinsics import _capture, _open_camera
from capture_extrinsics import (
    FIXED_R,
    _bridge,
    _config,
    clamp_xy,
    detect_chessboard,
    get_pose,
    move,
)


def main() -> int:
    bridge = _bridge()
    config = _config()
    cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    pose = {"x": 320.0, "y": -15.0, "z": 30.0, "r": FIXED_R}
    move(bridge, config, pose)

    for step in range(10):
        time.sleep(0.7)
        cam = _open_camera(cfg["calibration"])
        try:
            frame = _capture(cam)
        finally:
            cam.Close()
        corners = detect_chessboard(frame)
        if corners is None:
            print("lost at", pose, flush=True)
            pose = {"x": 320.0, "y": -15.0, "z": 35.0, "r": FIXED_R}
            try:
                move(bridge, config, pose)
            except Exception as exc:
                print("recover fail", exc, flush=True)
            continue

        pts = corners.reshape(-1, 2)
        ctr = pts.mean(axis=0)
        nu, nv = float(ctr[0] / 2448), float(ctr[1] / 2048)
        grid = pts.reshape(6, 9, 2)
        spacing = float(
            np.median(
                np.concatenate(
                    [
                        np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                        np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                    ]
                )
            )
        )
        mpp = 5.0 / spacing
        du = float(ctr[0] - 1224)
        dv = float(ctr[1] - 1024)
        print(
            f"step{step} center=({nu:.3f},{nv:.3f}) spacing={spacing:.1f} "
            f"pose=({pose['x']:.1f},{pose['y']:.1f},{pose['z']:.1f})",
            flush=True,
        )
        if abs(nu - 0.5) < 0.08 and abs(nv - 0.5) < 0.08:
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, (9, 6), corners, True)
            cv2.imwrite(str(HERE / "_extrinsic_centered.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print("CENTERED", get_pose(bridge, config), flush=True)
            return 0

        gain = 0.40
        bx, by = clamp_xy(pose["x"] + gain * du * mpp, pose["y"] - gain * dv * mpp)
        pose = {"x": round(bx, 2), "y": round(by, 2), "z": 30.0, "r": FIXED_R}
        move(bridge, config, pose)

    print("partial", get_pose(bridge, config), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
