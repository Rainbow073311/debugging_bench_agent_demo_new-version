"""Apply preferred camera J1 fit into camera_config.yaml files."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIT_PATH = HERE / "_camera_j1_offset.json"
TARGETS = [
    HERE / "camera_config.yaml",
    ROOT / "Vlm agent" / "Debugging-agent-v2" / "calibration" / "camera_config.yaml",
]


def main() -> None:
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))["fit"]
    t_ec = fit["T_end_to_camera"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in TARGETS:
        if not path.exists():
            print("skip missing", path)
            continue
        bak = path.with_name(path.name + f".bak_pre_j1_{stamp}")
        shutil.copy2(path, bak)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        old = (data.get("extrinsics") or {}).get("T_end_to_camera")
        ext = data.setdefault("extrinsics", {})
        ext["T_end_to_camera"] = {"R": t_ec["R"], "t_mm": t_ec["t_mm"]}
        ext["mount_mode"] = "eye_in_hand_xyz"
        ext["status"] = "calibrated"
        ext["robot_axes_used"] = ["x", "y", "z"]
        ext["robot_axes_ignored"] = ["r"]
        ext["j1_model"] = "Trans(TCP) @ Rz(J1) @ T_end_to_camera"
        ext["j1_source"] = "pose.j1_deg or atan2(TCP_y, TCP_x)"
        ext["date"] = datetime.now().strftime("%Y-%m-%d")
        ext["method"] = "camera_j1_offset LOOK L1-L4 physical-TL + Kabsch3d"
        ext["source_file"] = "calibration/_camera_j1_offset.json"
        ext["quality"] = {
            "n_looks": fit["n"],
            "j1_span_deg": fit["j1_span_deg"],
            "mean_residual_mm": fit["mean_residual_mm"],
            "mean_residual_xy_mm": fit["mean_residual_xy_mm"],
            "max_residual_xy_mm": fit["max_residual_xy_mm"],
            "excluded_looks": fit.get("excluded_looks"),
        }
        ext["previous_T_end_to_camera"] = old
        ext["transform_convention"] = (
            "T_end_to_camera maps camera coords into J1-rotating end-effector coords; "
            "T_base_cam = Trans(XYZ) @ Rz(J1) @ T_end_to_camera"
        )
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print("updated", path)
        print("  backup", bak.name)
        print("  t_mm", t_ec["t_mm"])


if __name__ == "__main__":
    main()
