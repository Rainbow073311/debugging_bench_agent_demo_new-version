"""Estimate tip yaw bias δ from known 50 mm line tip-touches (no camera).

For each placement the user puts the tip on both endpoints of a physical 50 mm
segment (e.g. the printed verification ruler).  We record TCP + J1 at each end.

  tip = TCP + 27 * (sin(J1+δ), -cos(J1+δ))
  ||tip_B - tip_A|| should equal 50 mm

δ is chosen to minimize distance error over all placements.  J1 must differ
across / within placements or δ is unobservable.

Usage:
  python calibration/calibrate_tip_delta_50mm.py record <label> A
  python calibration/calibrate_tip_delta_50mm.py record <label> B
  python calibration/calibrate_tip_delta_50mm.py fit
  python calibration/calibrate_tip_delta_50mm.py status
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_extrinsics import _bridge, _config, get_pose
from calibrate_tip_yaw_delta import get_angle
from tip_offset import tip_offset_xy, load_tip_offset_params

OUT = HERE / "_tip_delta_50mm.json"
LENGTH_MM = 50.0


def _params():
    try:
        p = load_tip_offset_params(HERE / "camera_config.yaml")
        return float(p["radius_xy_mm"]), float(p.get("delta_deg", 0.0))
    except Exception:
        return 27.0, 0.0


def load_state() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"length_mm": LENGTH_MM, "placements": {}, "status": "collecting"}


def save_state(state: dict) -> None:
    OUT.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_touch() -> dict:
    bridge, config = _bridge(), _config(12)
    pose = get_pose(bridge, config)
    ang = get_angle(bridge, config)
    return {
        "tcp_xy": [round(float(pose["x"]), 3), round(float(pose["y"]), 3)],
        "tcp_z": round(float(pose["z"]), 3),
        "tcp_r": round(float(pose["r"]), 3),
        "j1_deg": round(float(ang["j1"]), 4),
        "j2_deg": round(float(ang["j2"]), 4),
        "j3_deg": round(float(ang["j3"]), 4),
        "j4_deg": round(float(ang["j4"]), 4),
        "time": datetime.now().isoformat(timespec="seconds"),
    }


def tip_of(sample: dict, delta_deg: float, radius: float) -> np.ndarray:
    dx, dy = tip_offset_xy(sample["j1_deg"], radius_xy_mm=radius, delta_deg=delta_deg)
    return np.array(sample["tcp_xy"], float) + np.array([dx, dy], float)


def placement_error(placement: dict, delta_deg: float, radius: float) -> dict | None:
    a, b = placement.get("A"), placement.get("B")
    if not a or not b:
        return None
    ta, tb = tip_of(a, delta_deg, radius), tip_of(b, delta_deg, radius)
    dist = float(np.linalg.norm(tb - ta))
    j1_span = abs(float(b["j1_deg"]) - float(a["j1_deg"]))
    tcp_dist = float(
        np.linalg.norm(np.array(b["tcp_xy"], float) - np.array(a["tcp_xy"], float))
    )
    return {
        "tip_dist_mm": dist,
        "err_mm": dist - LENGTH_MM,
        "tcp_dist_mm": tcp_dist,
        "j1_span_deg": j1_span,
        "direction_xy": (tb - ta).tolist(),
    }


def fit_delta(state: dict) -> dict:
    radius, _ = _params()
    pairs = []
    for label, pl in state["placements"].items():
        if pl.get("A") and pl.get("B"):
            pairs.append((label, pl))
    if len(pairs) < 1:
        raise ValueError("need at least one complete A/B placement")

    sweep = np.linspace(-90.0, 90.0, 721)  # 0.25 deg
    best = None
    curve = []
    for d in sweep:
        errs = []
        details = {}
        for label, pl in pairs:
            info = placement_error(pl, float(d), radius)
            assert info is not None
            errs.append(info["err_mm"] ** 2)
            details[label] = info
        rms = float(math.sqrt(sum(errs) / len(errs)))
        mean_abs = float(np.mean([abs(details[l]["err_mm"]) for l, _ in pairs]))
        row = {"delta_deg": float(d), "rms_mm": rms, "mean_abs_err_mm": mean_abs}
        curve.append(row)
        if best is None or rms < best["rms_mm"]:
            best = {**row, "per_placement": details}

    # refine
    d0 = best["delta_deg"]
    fine = np.linspace(d0 - 2.0, d0 + 2.0, 401)
    for d in fine:
        errs = []
        details = {}
        for label, pl in pairs:
            info = placement_error(pl, float(d), radius)
            errs.append(info["err_mm"] ** 2)
            details[label] = info
        rms = float(math.sqrt(sum(errs) / len(errs)))
        mean_abs = float(np.mean([abs(v["err_mm"]) for v in details.values()]))
        if rms < best["rms_mm"]:
            best = {
                "delta_deg": float(d),
                "rms_mm": rms,
                "mean_abs_err_mm": mean_abs,
                "per_placement": details,
            }

    j1_all = []
    for _, pl in pairs:
        j1_all.extend([pl["A"]["j1_deg"], pl["B"]["j1_deg"]])
    j1_span = float(max(j1_all) - min(j1_all)) if j1_all else 0.0

    dx, dy = tip_offset_xy(float(np.mean(j1_all)), radius_xy_mm=radius, delta_deg=best["delta_deg"])
    return {
        "length_mm": LENGTH_MM,
        "radius_xy_mm": radius,
        "n_placements": len(pairs),
        "delta_deg": round(best["delta_deg"], 3),
        "fit_rms_mm": round(best["rms_mm"], 4),
        "mean_abs_err_mm": round(best["mean_abs_err_mm"], 4),
        "j1_span_all_deg": round(j1_span, 3),
        "tip_offset_xy_at_mean_j1_mm": [round(dx, 3), round(dy, 3)],
        "per_placement": {
            k: {
                "tip_dist_mm": round(v["tip_dist_mm"], 3),
                "err_mm": round(v["err_mm"], 3),
                "tcp_dist_mm": round(v["tcp_dist_mm"], 3),
                "j1_span_deg": round(v["j1_span_deg"], 3),
            }
            for k, v in best["per_placement"].items()
        },
        "warning": (
            None
            if j1_span >= 8.0
            else "J1 span still small; move the 50mm line to more different arm angles"
        ),
    }


def write_delta_to_config(delta_deg: float, *, force: bool = False) -> None:
    locked_json = HERE / "tip_offset.json"
    if locked_json.exists() and not force:
        data = json.loads(locked_json.read_text(encoding="utf-8"))
        if data.get("locked"):
            raise RuntimeError(
                f"{locked_json} is locked (delta_deg={data.get('delta_deg')}); "
                "pass force=True / --force to overwrite"
            )
    for rel in (
        HERE / "camera_config.yaml",
        HERE.parent / "Vlm agent" / "Debugging-agent-v2" / "calibration" / "camera_config.yaml",
        locked_json,
    ):
        if rel.suffix == ".json":
            payload = {
                "status": "calibrated",
                "model": "j1_tangential_xy",
                "radius_xy_mm": 27.0,
                "delta_deg": float(delta_deg),
                "z_mm": 0.0,
                "locked": True,
                "locked_date": datetime.now().strftime("%Y-%m-%d"),
                "source": f"50mm line fit force-update {datetime.now().isoformat(timespec='seconds')}",
                "note": "Do not change without recalibration. Runtime Inputdemo must use this file.",
            }
            rel.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            continue
        if not rel.exists():
            continue
        data = yaml.safe_load(rel.read_text(encoding="utf-8"))
        tip = data.setdefault("tip_offset", {})
        if tip.get("locked") and not force:
            continue
        tip["delta_deg"] = float(delta_deg)
        tip["status"] = "calibrated"
        tip["locked"] = True
        tip["delta_source"] = f"50mm line fit {datetime.now().isoformat(timespec='seconds')}"
        rel.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    if args[0] == "record":
        if len(args) < 3:
            print("Usage: record <label> A|B")
            return 1
        label, end = args[1], args[2].upper()
        if end not in ("A", "B"):
            print("end must be A or B")
            return 1
        sample = read_touch()
        state = load_state()
        pl = state["placements"].setdefault(label, {})
        pl[end] = sample
        state["status"] = "collecting"
        # auto fit if any complete pair
        complete = sum(1 for p in state["placements"].values() if p.get("A") and p.get("B"))
        if complete >= 1:
            try:
                state["fit"] = fit_delta(state)
                state["status"] = "fitted"
            except Exception as exc:
                state["fit_error"] = str(exc)
        save_state(state)
        print(
            json.dumps(
                {"recorded": {"label": label, "end": end, **sample}, "fit": state.get("fit")},
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    if args[0] == "fit":
        state = load_state()
        fit = fit_delta(state)
        state["fit"] = fit
        state["status"] = "fitted"
        save_state(state)
        print(json.dumps(fit, indent=2, ensure_ascii=False), flush=True)
        return 0

    if args[0] == "apply":
        state = load_state()
        fit = state.get("fit") or fit_delta(state)
        write_delta_to_config(fit["delta_deg"])
        print(json.dumps({"applied_delta_deg": fit["delta_deg"], "fit": fit}, indent=2), flush=True)
        return 0

    if args[0] == "status":
        print(json.dumps(load_state(), indent=2, ensure_ascii=False), flush=True)
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
