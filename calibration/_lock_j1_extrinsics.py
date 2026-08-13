"""Lock the current J1-model T_end_to_camera after user tip-hover yes."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CFG = HERE / "camera_config.yaml"
STATE = HERE / "_extrinsic_recal.json"
VLM = HERE.parent / "Vlm agent" / "Debugging-agent-v2" / "calibration" / "camera_config.yaml"


def main() -> int:
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    state = json.loads(STATE.read_text(encoding="utf-8"))
    fit = state.get("fit") or {}

    ext = data.setdefault("extrinsics", {})
    ext["status"] = "calibrated"
    ext["locked"] = True
    ext["date"] = datetime.now().strftime("%Y-%m-%d")
    ext["method"] = "physical-TL PnP + J1 end-frame Kabsch (robust LOOK subset)"
    ext["source_file"] = "calibration/_extrinsic_recal.json"
    ext["j1_model"] = "Trans(TCP) @ Rz(J1) @ T_end_to_camera"
    ext["j1_source"] = "pose.j1_deg or atan2(TCP_y, TCP_x)"
    ext["transform_convention"] = (
        "T_end_to_camera maps camera coords into J1-rotating end-effector coords; "
        "T_base_cam = Trans(XYZ) @ Rz(J1) @ T_end_to_camera"
    )
    ext.pop("note_candidate", None)
    ext["note"] = (
        "LOCKED — J1 eye-in-hand; tip hover validate err_vs_P_TL~0.94mm on 2026-08-13; "
        f"fit rms={fit.get('rms_mm')}mm n={fit.get('n_looks')} dropped={fit.get('dropped')}"
    )
    ext["quality"] = {
        "n_looks": fit.get("n_looks"),
        "j1_span_deg": fit.get("j1_span_deg"),
        "rms_mm": fit.get("rms_mm"),
        "mean_abs_mm": fit.get("mean_abs_mm"),
        "max_mm": fit.get("max_mm"),
        "dropped_looks": fit.get("dropped"),
        "validate_tip_vs_P_TL_mm": 0.936,
        "tl_idx_choice": fit.get("tl_idx_choice"),
    }
    CFG.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    state["status"] = "locked"
    state["locked_at"] = datetime.now().isoformat(timespec="seconds")
    state["locked_note"] = (
        "User yes on tip hover Camera-title LEFT; J1 model T_ec locked"
    )
    STATE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if VLM.exists():
        bak = VLM.with_name(
            VLM.name + f".bak_pre_lock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(VLM, bak)
        d2 = yaml.safe_load(VLM.read_text(encoding="utf-8")) or {}
        d2["extrinsics"] = data["extrinsics"]
        VLM.write_text(
            yaml.safe_dump(d2, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print({"vlm_synced": True, "backup": bak.name}, flush=True)
    else:
        print({"vlm_synced": False}, flush=True)

    print(
        json.dumps(
            {
                "locked": True,
                "t_mm": ext["T_end_to_camera"]["t_mm"],
                "j1_model": ext["j1_model"],
                "rms_mm": fit.get("rms_mm"),
                "validate_mm": 0.936,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
