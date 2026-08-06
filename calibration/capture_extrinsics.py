"""Step 1: Capture extrinsics dataset -- 12 poses, direct MovJ moves."""
import cv2, json, numpy as np, requests, time, yaml, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
API = "http://localhost:3000/api/mg400"
FIXED_R = 7.687
PATTERN = (9, 6)
SQUARE_MM = 5.0

sys.path.insert(0, str(HERE))
from capture_basler_intrinsics import _open_camera, _converter as _conv


def get_pose():
    return requests.get(f"{API}/status", timeout=5).json()["robot"]["pose"]


def move_and_wait(target):
    """Send MovJ, poll until pose reached."""
    resp = requests.post(f"{API}/command", json={
        "name": "move", "pose": target
    }, timeout=30).json()
    if not resp.get("ok"):
        raise RuntimeError(f"Move failed: {resp}")

    for _ in range(40):  # 20s max
        time.sleep(0.5)
        rp = get_pose()
        if rp is None:
            continue
        if all(abs(float(rp[k]) - target[k]) < 1.5 for k in ("x", "y", "z")):
            return rp
    raise TimeoutError(f"Robot did not reach {target}")


def capture_frame():
    cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    cam = _open_camera(cfg.get("calibration", {}))
    conv = _conv()
    cam.StartGrabbing()
    grab = cam.RetrieveResult(5000, 2)
    if not grab.GrabSucceeded():
        cam.Close(); return None
    frame = conv.Convert(grab).GetArray()
    grab.Release(); cam.StopGrabbing(); cam.Close()
    return frame


def detect_chessboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


def main():
    pose = get_pose()
    if pose is None:
        print("ERROR: cannot read robot pose"); return
    cx, cy, cz = float(pose["x"]), float(pose["y"]), float(pose["z"])
    r = float(pose["r"])
    print(f"Start: X={cx:.1f} Y={cy:.1f} Z={cz:.1f} R={r:.4f}")

    if abs(((r - FIXED_R + 180) % 360) - 180) > 1.0:
        print(f"WARNING: R={r:.4f} != {FIXED_R}. Setting now...")
        move_and_wait({"x": cx, "y": cy, "z": cz, "r": FIXED_R})
        pose = get_pose()
        cx, cy, cz = float(pose["x"]), float(pose["y"]), float(pose["z"])

    # 3 heights x 4 XY = 12 poses. Z must stay above -155 (table)
    targets = []
    for dz in [0, -6, 6]:
        for dx, dy in [(0, 0), (8, 0), (-8, 0), (0, 8)]:
            tz = cz + dz
            targets.append({"x": cx + dx, "y": cy + dy, "z": tz, "r": FIXED_R})
    print(f"{len(targets)} targets to capture")

    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = HERE / "calibration_sessions" / f"extrinsics_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    K = np.array(cfg["intrinsics"]["camera_matrix"], dtype=np.float64)
    D = np.array(cfg["intrinsics"]["dist_coeffs"], dtype=np.float64)
    objp = np.zeros((54, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * SQUARE_MM

    samples = []
    for i, t in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] move to X={t['x']:.1f} Y={t['y']:.1f} Z={t['z']:.1f}", end=" ")
        try:
            rp = move_and_wait(t)
        except Exception as e:
            print(f"SKIP: {e}"); continue

        frame = capture_frame()
        if frame is None:
            print("SKIP: capture"); continue

        corners = detect_chessboard(frame)
        if corners is None:
            print("SKIP: no board")
            cv2.imwrite(str(session_dir / f"fail_{i+1:03d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92]); continue

        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
        print(f"OK tz={tvec[2][0]:.0f}mm")

        vis = frame.copy()
        cv2.drawChessboardCorners(vis, PATTERN, corners, True)
        cv2.imwrite(str(session_dir / f"img_{i+1:03d}.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        samples.append({
            "index": i + 1,
            "robot_pose": {k: float(rp[k]) for k in ("x", "y", "z", "r")},
            "marker_in_camera": {
                "rvec": rvec.reshape(-1).tolist(),
                "t_mm": tvec.reshape(-1).tolist(),
            },
            "image": f"img_{i+1:03d}.jpg",
        })

    dataset = {"board": {"pattern": list(PATTERN), "square_size_mm": SQUARE_MM},
               "fixed_r_deg": FIXED_R, "samples": samples}
    out = session_dir / "dataset.json"
    out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone: {len(samples)}/{len(targets)} → {out}")


if __name__ == "__main__":
    main()
