"""Step 3: Verify extrinsics accuracy.
Captures a chessboard image, projects 5 key points to robot base,
prints predicted coordinates for user to probe-verify.
"""
import cv2, json, numpy as np, yaml, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_basler_intrinsics import _open_camera, _converter as _conv
from coordinate_transforms import PixelToWorld

SQUARE_MM = 5.0
PATTERN = (9, 6)


def main():
    # Get robot pose
    try:
        resp = urllib.request.urlopen("http://localhost:3000/api/mg400/status", timeout=5)
        status = json.loads(resp.read())
        robot_pose = status["robot"]["pose"]
    except Exception as e:
        print(f"Cannot read robot pose: {e}"); sys.exit(1)

    r = float(robot_pose["r"])
    print(f"Robot: X={robot_pose['x']:.1f} Y={robot_pose['y']:.1f} "
          f"Z={robot_pose['z']:.1f} R={r:.4f}")

    # Capture
    cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text())
    cam = _open_camera(cfg.get("calibration", {}))
    conv = _conv()
    cam.StartGrabbing()
    grab = cam.RetrieveResult(5000, 2)
    if not grab.GrabSucceeded():
        print("Capture failed!"); cam.Close(); sys.exit(1)
    frame = conv.Convert(grab).GetArray()
    grab.Release(); cam.StopGrabbing(); cam.Close()

    # Detect chessboard
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    if not found:
        print("Chessboard not detected!"); sys.exit(1)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    print(f"Detected {len(corners)} corners")

    # PnP
    objp = np.zeros((54, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQUARE_MM
    K = np.array(cfg["intrinsics"]["camera_matrix"])
    D = np.array(cfg["intrinsics"]["dist_coeffs"])
    ret, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
    R_cam, _ = cv2.Rodrigues(rvec)
    print(f"solvePnP OK, tvec_z={tvec[2][0]:.1f}mm")

    # 5 calibration points (object space)
    pts_obj = np.array([
        [0, 0, 0], [40, 0, 0], [0, 25, 0], [40, 25, 0], [20, 12.5, 0]
    ], np.float32)
    labels = ["1_TL", "2_TR", "3_BL", "4_BR", "5_Center"]
    colors = [(0, 220, 220), (0, 140, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255)]

    # Transform to base
    mapper = PixelToWorld(str(HERE / "camera_config.yaml"), robot_pose=robot_pose)
    T_base_to_cam = mapper.base_to_camera(robot_pose)

    points_base = []
    for pt_obj in pts_obj:
        pt_cam = (R_cam @ pt_obj.reshape(3, 1) + tvec).ravel()
        pt_cam_h = np.array([pt_cam[0], pt_cam[1], pt_cam[2], 1.0])
        pt_base = T_base_to_cam @ pt_cam_h
        points_base.append((float(pt_base[0]), float(pt_base[1]), float(pt_base[2])))

    # Draw
    out = frame.copy()
    pts_img, _ = cv2.projectPoints(pts_obj, rvec, tvec, K, D)
    for i, (p, c) in enumerate(zip(pts_img, colors)):
        px, py = int(p[0][0]), int(p[0][1])
        cv2.circle(out, (px, py), 16, c, -1)
        cv2.circle(out, (px, py), 18, (255, 255, 255), 3)
        cv2.putText(out, str(i + 1), (px - 12, py + 12),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 4)
        cv2.putText(out, str(i + 1), (px - 12, py + 12),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 2)
    cv2.drawChessboardCorners(out, PATTERN, corners, True)
    cv2.imwrite(str(HERE / "_verify.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\n===== Probe These 5 Points =====")
    for label, (x, y, z) in zip(labels, points_base):
        print(f"  {label}: X={x:.2f}  Y={y:.2f}  Z={z:.2f}")

    # Save
    result = {
        "robot_pose": robot_pose,
        "points": {labels[i]: {"x": round(p[0], 2), "y": round(p[1], 2), "z": round(p[2], 2)}
                   for i, p in enumerate(points_base)},
        "image": str(HERE / "_verify.jpg"),
    }
    (HERE / "_verify.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nAnnotated image: {HERE / '_verify.jpg'}")


if __name__ == "__main__":
    main()
