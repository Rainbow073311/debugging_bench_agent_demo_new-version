import json
from pathlib import Path

import numpy as np
import yaml

old = yaml.safe_load(Path("calibration/camera_config.yaml").read_text(encoding="utf-8"))
new = json.loads(
    Path("calibration/calibration_sessions/intrinsics_20260806_175412/intrinsics.json").read_text(
        encoding="utf-8"
    )
)
Ko = np.array(old["intrinsics"]["camera_matrix"], float)
Kn = np.array(new["camera_matrix"], float)

print("OLD fx,fy,cx,cy =", Ko[0, 0], Ko[1, 1], Ko[0, 2], Ko[1, 2])
print("NEW fx,fy,cx,cy =", Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2])
print(f"dfx = {(Kn[0, 0] - Ko[0, 0]) / Ko[0, 0] * 100:.1f}%")
print(f"dfy = {(Kn[1, 1] - Ko[1, 1]) / Ko[1, 1] * 100:.1f}%")
print(f"dcx = {Kn[0, 2] - Ko[0, 2]:.1f} px, dcy = {Kn[1, 2] - Ko[1, 2]:.1f} px")
print(f"X focal scale new/old = {Kn[0, 0] / Ko[0, 0]:.4f}")
print(f"Y focal scale new/old = {Kn[1, 1] / Ko[1, 1]:.4f}")

print("\npixel  Zc  dX_mm  dY_mm  |dXY|_mm")
for u, v in [(1224, 1024), (600, 400), (1800, 1600), (200, 200), (2200, 1800)]:
    for Z in (120, 160, 200, 250):
        Xo = (u - Ko[0, 2]) * Z / Ko[0, 0]
        Yo = (v - Ko[1, 2]) * Z / Ko[1, 1]
        Xn = (u - Kn[0, 2]) * Z / Kn[0, 0]
        Yn = (v - Kn[1, 2]) * Z / Kn[1, 1]
        dX, dY = Xn - Xo, Yn - Yo
        print(f"({u},{v}) {Z}  {dX:7.2f} {dY:7.2f} {np.hypot(dX, dY):7.2f}")

Z = 180.0
print(f"\nPrincipal-point shift alone at Zc={Z:.0f} mm (using old f):")
print(f"  dX ~= {-(Kn[0, 2] - Ko[0, 2]) * Z / Ko[0, 0]:.2f} mm")
print(f"  dY ~= {-(Kn[1, 2] - Ko[1, 2]) * Z / Ko[1, 1]:.2f} mm")
print(
    "Scale-only: a true 20 mm X offset would be recovered as "
    f"{20 * Ko[0, 0] / Kn[0, 0]:.2f} mm if only fx changed to the new value"
)
