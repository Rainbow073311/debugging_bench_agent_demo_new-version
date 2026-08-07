"""Capture photo, detect chessboard TL, move robot to that position."""
import cv2, json, numpy as np, requests, sys, time, yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_basler_intrinsics import _open_camera, _converter
from coordinate_transforms import PixelToWorld

API = "http://localhost:3000/api/mg400"
PATTERN = (9, 6)
FIXED_R = 7.687

def get_pose():
    return requests.get(f"{API}/status", timeout=5).json()["robot"]["pose"]

def move_and_wait(target):
    resp = requests.post(f"{API}/command", json={"name": "move", "pose": target}, timeout=30).json()
    if not resp.get("ok"):
        raise RuntimeError(f"Move failed: {resp}")
    for _ in range(40):
        time.sleep(0.5)
        rp = get_pose()
        if rp is None: continue
        if all(abs(float(rp[k]) - target[k]) < 1.5 for k in ("x","y","z")):
            return rp
    raise TimeoutError(f"Robot did not reach {target}")

# 1. Capture frame
print("Capturing...")
cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
cam = _open_camera(cfg.get("calibration", {}))
conv = _converter()
cam.StartGrabbing()
grab = cam.RetrieveResult(5000, 2)
frame = conv.Convert(grab).GetArray() if grab.GrabSucceeded() else None
grab.Release(); cam.StopGrabbing(); cam.Close()
if frame is None:
    print("ERROR: capture failed"); sys.exit(1)
print(f"Captured: {frame.shape[1]}x{frame.shape[0]}")

# 2. Detect chessboard
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
found, corners = cv2.findChessboardCornersSB(gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
if not found:
    # Save debug image
    cv2.imwrite(str(HERE / "_tl_debug_fail.jpg"), frame)
    print("ERROR: chessboard not found. Saved _tl_debug_fail.jpg"); sys.exit(1)
corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
print(f"Chessboard found: {len(corners)} corners")

# OpenCV chessboard corner ordering: origin is the first corner found.
# For a 9x6 board, corners[0] is one of the 4 outer corners.
# OpenCV's findChessboardCornersSB ordering: starts from a specific corner,
# let's determine which is TL by checking the corner indices.
# corners are row-major: row0=[0..8], row1=[9..17], ... row5=[45..53]
# The origin corner (index 0) is the top-left in OpenCV's coordinate system.
# But which physical corner this maps to depends on the camera orientation.

# Let's identify all 4 corners:
tl_px = corners[0][0]      # row 0, col 0
tr_px = corners[8][0]      # row 0, col 8
bl_px = corners[45][0]     # row 5, col 0
br_px = corners[53][0]     # row 5, col 8

print(f"TL corner pixel: ({tl_px[0]:.1f}, {tl_px[1]:.1f})")
print(f"TR corner pixel: ({tr_px[0]:.1f}, {tr_px[1]:.1f})")
print(f"BL corner pixel: ({bl_px[0]:.1f}, {bl_px[1]:.1f})")
print(f"BR corner pixel: ({br_px[0]:.1f}, {br_px[1]:.1f})")

# 3. Get robot pose and project TL to world
pose = get_pose()
print(f"Robot pose: X={pose['x']:.2f} Y={pose['y']:.2f} Z={pose['z']:.2f} R={pose['r']:.4f}")

ptw = PixelToWorld(str(HERE / "camera_config.yaml"))
ptw.set_robot_pose(pose)

tl_world = ptw.pixel_to_table(tl_px[0], tl_px[1])
if tl_world is None:
    print("ERROR: TL projects outside table"); sys.exit(1)
print(f"TL world: X={tl_world[0]:.2f} Y={tl_world[1]:.2f} Z={tl_world[2]:.2f}")

# 4. Annotate and save
vis = frame.copy()
cv2.drawChessboardCorners(vis, PATTERN, corners, True)
cv2.circle(vis, (int(tl_px[0]), int(tl_px[1])), 15, (0,0,255), 3)
cv2.putText(vis, "TL", (int(tl_px[0])+20, int(tl_px[1])-20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
cv2.imwrite(str(HERE / "_tl_detected.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("Saved _tl_detected.jpg")

# 5. Move robot to TL
target = {"x": tl_world[0], "y": tl_world[1], "z": tl_world[2] + 10, "r": FIXED_R}
print(f"Moving to TL + 10mm hover: X={target['x']:.2f} Y={target['y']:.2f} Z={target['z']:.2f}")
move_and_wait(target)
print("Done — robot is at TL hover position")
