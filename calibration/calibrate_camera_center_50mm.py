"""Calibrate camera image-center XY offset from a known 50 mm line (LOOK, not tip).

Same physical ruler as tip_delta_50mm.  For each placement:
  A/B = move so the IMAGE CENTER sits on each endpoint of the 50 mm segment.
  Record TCP + true J1 (GetAngle).

Model (same family as tip, independent parameters):
  cam_xy = TCP_xy + radius * (sin(J1+δ), -cos(J1+δ))
  ||cam_B - cam_A|| should equal 50 mm

Do NOT reuse tip δ=6.15 — camera has its own (radius, δ).

Usage:
  python calibration/calibrate_camera_center_50mm.py record <label> A
  python calibration/calibrate_camera_center_50mm.py record <label> B
  python calibration/calibrate_camera_center_50mm.py fit
  python calibration/calibrate_camera_center_50mm.py apply [--force]
  python calibration/calibrate_camera_center_50mm.py status
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capture_basler_intrinsics import _capture, _open_camera  # noqa: E402
from capture_extrinsics import _bridge, _config, get_pose  # noqa: E402
from calibrate_tip_yaw_delta import get_angle  # noqa: E402
from tip_offset import tip_offset_xy  # noqa: E402

OUT = HERE / "_camera_center_50mm.json"
OUT_DIR = HERE / "_camera_center_50mm_session"
LOCKED_JSON = HERE / "camera_center_offset.json"
LENGTH_MM = 50.0
IMG_W, IMG_H = 2448, 2048


def load_state() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {
        "length_mm": LENGTH_MM,
        "model": "cam_xy = TCP_xy + r*(sin(J1+δ), -cos(J1+δ))",
        "placements": {},
        "status": "collecting",
    }


def save_state(state: dict) -> None:
    OUT.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_look() -> dict:
    """Capture current pose + J1 and a debug frame marked with image center."""
    bridge, config = _bridge(), _config(12)
    pose = get_pose(bridge, config)
    if pose is None:
        raise RuntimeError("GetPose failed")
    ang = get_angle(bridge, config)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    cam = _open_camera(cam_cfg["calibration"])
    try:
        for _ in range(2):
            _capture(cam)
        time.sleep(0.1)
        frame = _capture(cam)
    finally:
        cam.Close()

    cx, cy = IMG_W // 2, IMG_H // 2
    marked = frame.copy()
    cv2.drawMarker(marked, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
    cv2.circle(marked, (cx, cy), 18, (0, 255, 255), 2)
    cv2.putText(
        marked,
        "IMAGE CENTER — put this on ruler endpoint",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    stamp = datetime.now().strftime("%H%M%S")
    raw_path = OUT_DIR / f"look_{stamp}_raw.jpg"
    marked_path = OUT_DIR / f"look_{stamp}_marked.jpg"
    cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(marked_path), marked, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return {
        "tcp_xy": [round(float(pose["x"]), 3), round(float(pose["y"]), 3)],
        "tcp_z": round(float(pose["z"]), 3),
        "tcp_r": round(float(pose["r"]), 3),
        "j1_deg": round(float(ang["j1"]), 4),
        "j2_deg": round(float(ang["j2"]), 4),
        "j3_deg": round(float(ang["j3"]), 4),
        "j4_deg": round(float(ang["j4"]), 4),
        "time": datetime.now().isoformat(timespec="seconds"),
        "raw": str(raw_path),
        "marked": str(marked_path),
        "note": "Assumes image-center ray hits the ruler endpoint on the table.",
    }


def cam_of(sample: dict, delta_deg: float, radius: float) -> np.ndarray:
    dx, dy = tip_offset_xy(sample["j1_deg"], radius_xy_mm=radius, delta_deg=delta_deg)
    return np.array(sample["tcp_xy"], float) + np.array([dx, dy], float)


def placement_error(placement: dict, delta_deg: float, radius: float) -> dict | None:
    a, b = placement.get("A"), placement.get("B")
    if not a or not b:
        return None
    ca, cb = cam_of(a, delta_deg, radius), cam_of(b, delta_deg, radius)
    dist = float(np.linalg.norm(cb - ca))
    return {
        "dist_mm": dist,
        "err_mm": dist - LENGTH_MM,
        "cam_a": ca.tolist(),
        "cam_b": cb.tolist(),
        "j1_a": a["j1_deg"],
        "j1_b": b["j1_deg"],
    }


def fit_params(state: dict) -> dict:
    pairs = []
    for label, pl in state["placements"].items():
        if pl.get("A") and pl.get("B"):
            pairs.append((label, pl))
    if not pairs:
        raise ValueError("need at least one complete A/B placement")

    best = None
    # Search radius and delta (camera may differ from tip 27/6.15)
    for radius in np.linspace(20.0, 80.0, 61):
        for delta in np.linspace(-180.0, 180.0, 361):
            errs = []
            details = {}
            for label, pl in pairs:
                info = placement_error(pl, float(delta), float(radius))
                if info is None:
                    continue
                errs.append(info["err_mm"])
                details[label] = info
            if not errs:
                continue
            rms = float(np.sqrt(np.mean(np.square(errs))))
            mean_abs = float(np.mean(np.abs(errs)))
            row = {
                "radius_xy_mm": float(radius),
                "delta_deg": float(delta),
                "rms_mm": rms,
                "mean_abs_err_mm": mean_abs,
                "per_placement": details,
            }
            if best is None or rms < best["rms_mm"]:
                best = row

    # Refine locally
    r0, d0 = best["radius_xy_mm"], best["delta_deg"]
    for radius in np.linspace(r0 - 2.0, r0 + 2.0, 21):
        for delta in np.linspace(d0 - 2.0, d0 + 2.0, 41):
            errs = []
            details = {}
            for label, pl in pairs:
                info = placement_error(pl, float(delta), float(radius))
                errs.append(info["err_mm"])
                details[label] = info
            rms = float(np.sqrt(np.mean(np.square(errs))))
            if rms < best["rms_mm"]:
                best = {
                    "radius_xy_mm": float(radius),
                    "delta_deg": float(delta),
                    "rms_mm": rms,
                    "mean_abs_err_mm": float(np.mean(np.abs(errs))),
                    "per_placement": details,
                }

    j1_all = []
    for _, pl in pairs:
        j1_all.extend([pl["A"]["j1_deg"], pl["B"]["j1_deg"]])
    j1_span = float(max(j1_all) - min(j1_all)) if j1_all else 0.0

    return {
        "length_mm": LENGTH_MM,
        "n_placements": len(pairs),
        "radius_xy_mm": round(best["radius_xy_mm"], 3),
        "delta_deg": round(best["delta_deg"], 3),
        "fit_rms_mm": round(best["rms_mm"], 4),
        "mean_abs_err_mm": round(best["mean_abs_err_mm"], 4),
        "j1_span_deg": round(j1_span, 3),
        "per_placement": {
            k: {
                "dist_mm": round(v["dist_mm"], 3),
                "err_mm": round(v["err_mm"], 3),
                "j1_a": v["j1_a"],
                "j1_b": v["j1_b"],
            }
            for k, v in best["per_placement"].items()
        },
        "warning": None
        if j1_span >= 15.0
        else "J1 span still small; move the 50mm line to more different arm angles",
        "note": "Camera params are independent of tip_offset (do not copy tip δ).",
    }


def apply_fit(state: dict, *, force: bool = False) -> dict:
    fit = state.get("fit")
    if not fit:
        raise ValueError("no fit yet; run fit first")
    radius = float(fit["radius_xy_mm"])
    delta = float(fit["delta_deg"])
    now = datetime.now().isoformat(timespec="seconds")
    payload = {
        "status": "calibrated",
        "model": "j1_tangential_xy",
        "radius_xy_mm": radius,
        "delta_deg": delta,
        "z_mm": 0.0,
        "formula": (
            "dx = radius_xy_mm * sin(J1_deg + delta_deg); "
            "dy = -radius_xy_mm * cos(J1_deg + delta_deg)"
        ),
        "convention": (
            "cam_xy = TCP_xy + [dx, dy] when image-center ray hits table point; "
            "to put optical center on target P, command TCP = P - [dx, dy] (J1-consistent)"
        ),
        "j1_proxy": "atan2(TCP_y, TCP_x) degrees on MG400",
        "locked": True,
        "locked_date": datetime.now().strftime("%Y-%m-%d"),
        "source": (
            f"50mm LOOK fit P1-P4 {now}; "
            f"rms={fit.get('fit_rms_mm')}mm j1_span={fit.get('j1_span_deg')}deg"
        ),
        "fit": {
            "n_placements": fit.get("n_placements"),
            "fit_rms_mm": fit.get("fit_rms_mm"),
            "mean_abs_err_mm": fit.get("mean_abs_err_mm"),
            "j1_span_deg": fit.get("j1_span_deg"),
            "per_placement": fit.get("per_placement"),
        },
        "note": "Independent of tip_offset. Runtime ultra-close uses this file.",
    }
    if LOCKED_JSON.exists() and not force:
        existing = json.loads(LOCKED_JSON.read_text(encoding="utf-8"))
        if existing.get("locked"):
            raise RuntimeError(
                f"{LOCKED_JSON.name} is locked "
                f"(r={existing.get('radius_xy_mm')}, δ={existing.get('delta_deg')}); "
                "pass --force to overwrite"
            )
    LOCKED_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    yaml_section = {
        "status": "calibrated",
        "locked": True,
        "model": "j1_tangential_xy",
        "radius_xy_mm": radius,
        "delta_deg": delta,
        "z_mm": 0.0,
        "formula": payload["formula"],
        "convention": payload["convention"],
        "j1_proxy": payload["j1_proxy"],
        "locked_file": "calibration/camera_center_offset.json",
        "source": payload["source"],
        "note": "LOCKED — optical-center XY vs TCP; evaluate from J1; independent of tip_offset",
    }
    written = [str(LOCKED_JSON)]
    for rel in (
        HERE / "camera_config.yaml",
        HERE.parent / "Vlm agent" / "Debugging-agent-v2" / "calibration" / "camera_config.yaml",
    ):
        if not rel.exists():
            continue
        data = yaml.safe_load(rel.read_text(encoding="utf-8")) or {}
        section = data.get("camera_center_offset") or {}
        if section.get("locked") and not force:
            continue
        data["camera_center_offset"] = yaml_section
        rel.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append(str(rel))

    state["status"] = "applied"
    state["applied"] = {
        "time": now,
        "radius_xy_mm": radius,
        "delta_deg": delta,
        "files": written,
    }
    save_state(state)
    return {"ok": True, "applied": state["applied"], "payload": payload}


def status(state: dict) -> None:
    placements = state.get("placements") or {}
    complete = {
        k: bool(v.get("A") and v.get("B")) for k, v in placements.items()
    }
    print(
        json.dumps(
            {
                "status": state.get("status"),
                "n_labels": len(placements),
                "complete": complete,
                "fit": state.get("fit"),
                "hint": "Need >=3 complete placements, J1 span >=15–20 deg preferred",
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def main(argv: list[str]) -> int:
    state = load_state()
    if len(argv) < 2 or argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd = argv[1]
    if cmd == "status":
        status(state)
        return 0
    if cmd == "record":
        if len(argv) < 4 or argv[3].upper() not in ("A", "B"):
            print("Usage: record <label> A|B")
            return 1
        label, end = argv[2], argv[3].upper()
        sample = read_look()
        pl = state["placements"].setdefault(label, {})
        pl[end] = sample
        save_state(state)
        print(
            json.dumps({"recorded": {"label": label, "end": end, **sample}}, indent=2, ensure_ascii=False),
            flush=True,
        )
        print(
            f"OK {label}/{end}. Put IMAGE CENTER on the other endpoint, then record {label} {'B' if end=='A' else 'A'}.",
            flush=True,
        )
        return 0
    if cmd == "fit":
        fit = fit_params(state)
        state["fit"] = fit
        state["status"] = "fitted"
        save_state(state)
        print(json.dumps(fit, indent=2, ensure_ascii=False), flush=True)
        return 0
    if cmd == "apply":
        force = "--force" in argv[2:] or "force" in argv[2:]
        result = apply_fit(state, force=force)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
