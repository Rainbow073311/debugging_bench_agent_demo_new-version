"""Quick Basler camera preview — shows live view, press 'q' to quit, 's' to save."""
import sys, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from capture_basler_intrinsics import _open_camera, _converter
except ImportError:
    print("Cannot import Basler modules. Is pypylon installed?")
    sys.exit(1)

import cv2

# Load camera config
cfg_path = Path(__file__).resolve().parent / "camera_config.yaml"
cfg = yaml.safe_load(open(cfg_path))
calib = cfg.get("calibration", {})
serial = calib.get("serial", "")
print(f"Opening Basler serial={serial}...")

camera = _open_camera(calib)

# Enable auto white balance
try:
    camera.BalanceWhiteAuto.SetValue("Continuous")
    print("Auto white balance: Continuous")
except Exception:
    try:
        camera.BalanceWhiteAuto.SetValue("Once")
        print("Auto white balance: Once")
    except Exception as e:
        print(f"Auto white balance not available: {e}")
if camera is None:
    print(f"Basler camera serial={serial} not found!")
    sys.exit(1)

converter = _converter()
camera.StartGrabbing()
print(f"Camera: {camera.GetDeviceInfo().GetModelName()} serial={camera.GetDeviceInfo().GetSerialNumber()}")
print("Press 'q' to quit, 's' to save snapshot")

cv2.namedWindow("Basler Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Basler Preview", 1280, 960)

while camera.IsGrabbing():
    grab = camera.RetrieveResult(5000, 2)
    if not grab.GrabSucceeded():
        continue
    frame = converter.Convert(grab).GetArray()
    grab.Release()

    cv2.imshow("Basler Preview", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        path = "basler_snapshot.jpg"
        cv2.imwrite(path, frame)
        print(f"Saved: {path} ({frame.shape[1]}x{frame.shape[0]})")

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
print("Done.")
