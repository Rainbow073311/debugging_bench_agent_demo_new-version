import sys
from pathlib import Path

import cv2
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_basler_intrinsics import _capture, _open_camera
from capture_extrinsics import _bridge, _config, detect_chessboard, get_pose


def main() -> int:
    bridge = _bridge()
    config = _config()
    print("pose", get_pose(bridge, config), flush=True)
    cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    cam = _open_camera(cfg["calibration"])
    try:
        frame = _capture(cam)
    finally:
        cam.Close()
    cv2.imwrite(str(HERE / "_probe_live.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    corners = detect_chessboard(frame)
    if corners is None:
        print("FAIL: board not found in live view", flush=True)
        return 2
    corners = corners.reshape(-1, 2)
    labels = [("TL", 0), ("TR", 8), ("BL", 45), ("BR", 53)]
    colors = {"TL": (0, 0, 255), "TR": (0, 165, 255), "BL": (0, 255, 0), "BR": (255, 0, 0)}
    vis = frame.copy()
    for name, idx in labels:
        x, y = map(float, corners[idx])
        side = "left" if x < 1224 else "right"
        vert = "upper" if y < 1024 else "lower"
        print(f"{name}: idx={idx} px=({x:.0f},{y:.0f}) image=({side}, {vert})", flush=True)
        cv2.circle(vis, (int(x), int(y)), 30, colors[name], -1)
        cv2.putText(vis, name, (int(x) + 25, int(y) + 12), cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 5)
        cv2.putText(vis, name, (int(x) + 25, int(y) + 12), cv2.FONT_HERSHEY_DUPLEX, 1.8, colors[name], 3)
    tl, tr, bl = corners[0], corners[8], corners[45]
    cv2.arrowedLine(vis, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 165, 255), 6, tipLength=0.08)
    cv2.arrowedLine(vis, (int(tl[0]), int(tl[1])), (int(bl[0]), int(bl[1])), (0, 255, 0), 6, tipLength=0.08)
    out = HERE / "_probe_corners_labeled.jpg"
    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
