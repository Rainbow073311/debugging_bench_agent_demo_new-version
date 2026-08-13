"""Estimate tip XY yaw bias δ from chessboard tip-touches (no camera).

Model (XY only):
  tip_xy = TCP_xy + 27 * (sin(J1_deg + δ), -cos(J1_deg + δ))
  tip points must form a rigid 5 mm chessboard grid.

δ = 0 means pure -J1 tangential. J1 must vary across touches
(use corners that span XY so atan2/J1 changes by several degrees).

Usage (interactive):
  python calibration/calibrate_tip_yaw_delta.py

Or record one touch then fit later:
  python calibration/calibrate_tip_yaw_delta.py record TL
  python calibration/calibrate_tip_yaw_delta.py fit
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from capture_extrinsics import _bridge, _config, get_pose

OUT = HERE / "_tip_yaw_delta.json"
TIP_XY_MM = 27.0

# Inner-corner board coords (mm), OpenCV TL origin, 9x6 @ 5 mm
BOARD_XY = {
    "TL": np.array([0.0, 0.0]),
    "TR": np.array([40.0, 0.0]),
    "BL": np.array([0.0, 25.0]),
    "BR": np.array([40.0, 25.0]),
}


def _robot():
    bridge = _bridge()
    config = _config(12)
    return bridge, config


def get_angle(bridge, config) -> dict:
    """Return J1..J4 in degrees via dashboard GetAngle()."""
    robot = bridge.Mg400(config)
    try:
        robot.connect_dashboard()
        result = robot.dash("GetAngle()")
        if not result.get("ok"):
            raise RuntimeError(result)
        values = result["response"].split(",{", 1)[1].split("}", 1)[0].split(",")
        return {
            "j1": float(values[0]),
            "j2": float(values[1]),
            "j3": float(values[2]),
            "j4": float(values[3]),
        }
    finally:
        robot.close()


def tip_offset_xy(j1_deg: float, delta_deg: float, radius_mm: float = TIP_XY_MM) -> np.ndarray:
    ang = math.radians(j1_deg + delta_deg)
    return radius_mm * np.array([math.sin(ang), -math.cos(ang)], dtype=np.float64)


def fit_rigid_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Fit dst ≈ R(yaw) @ src + t. Returns yaw_deg, t, rms."""
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    a = src - c_src
    b = dst - c_dst
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = c_dst - r @ c_src
    pred = (r @ src.T).T + t
    rms = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, t, rms


def evaluate_delta(samples: list[dict], delta_deg: float) -> dict:
    names = [s["corner"] for s in samples]
    src = np.array([BOARD_XY[n] for n in names], float)
    tips = []
    for s in samples:
        j1 = float(s["j1_deg"])
        tcp = np.array(s["tcp_xy"], float)
        tips.append(tcp + tip_offset_xy(j1, delta_deg))
    tips = np.array(tips, float)
    yaw, t, rms = fit_rigid_2d(src, tips)
    # Span check vs nominal board
    if "TL" in names and "TR" in names:
        i, j = names.index("TL"), names.index("TR")
        x_span = float(np.linalg.norm(tips[j] - tips[i]))
    else:
        x_span = None
    if "TL" in names and "BL" in names:
        i, j = names.index("TL"), names.index("BL")
        y_span = float(np.linalg.norm(tips[j] - tips[i]))
    else:
        y_span = None
    return {
        "delta_deg": delta_deg,
        "rms_mm": rms,
        "board_yaw_deg": yaw,
        "board_t_xy": [float(t[0]), float(t[1])],
        "tip_x_span_mm": x_span,
        "tip_y_span_mm": y_span,
    }


def fit_delta(samples: list[dict], sweep: np.ndarray | None = None) -> dict:
    if len(samples) < 3:
        raise ValueError("need at least 3 touched corners")
    if sweep is None:
        sweep = np.linspace(-60.0, 60.0, 481)  # 0.25 deg
    scores = [evaluate_delta(samples, float(d)) for d in sweep]
    best = min(scores, key=lambda s: s["rms_mm"])
    # Refine locally
    d0 = best["delta_deg"]
    fine = np.linspace(d0 - 2.0, d0 + 2.0, 161)
    scores_fine = [evaluate_delta(samples, float(d)) for d in fine]
    best = min(scores_fine, key=lambda s: s["rms_mm"])

    j1s = np.array([s["j1_deg"] for s in samples], float)
    j1_span = float(j1s.max() - j1s.min())
    # Nominal offset at mean J1
    j1_mean = float(j1s.mean())
    off = tip_offset_xy(j1_mean, best["delta_deg"])
    return {
        "tip_xy_radius_mm": TIP_XY_MM,
        "delta_deg": round(best["delta_deg"], 3),
        "fit_rms_mm": round(best["rms_mm"], 4),
        "j1_span_deg": round(j1_span, 3),
        "j1_mean_deg": round(j1_mean, 3),
        "tip_offset_xy_at_mean_j1_mm": [round(float(off[0]), 3), round(float(off[1]), 3)],
        "board_yaw_deg": round(best["board_yaw_deg"], 3),
        "board_t_xy": [round(v, 3) for v in best["board_t_xy"]],
        "tip_x_span_mm": None if best["tip_x_span_mm"] is None else round(best["tip_x_span_mm"], 3),
        "tip_y_span_mm": None if best["tip_y_span_mm"] is None else round(best["tip_y_span_mm"], 3),
        "warning": (
            None
            if j1_span >= 3.0
            else "J1 span < 3 deg — δ may be weakly observable; touch corners farther apart in XY"
        ),
        "curve_min": best,
    }


def load_state() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"samples": [], "status": "collecting"}


def save_state(state: dict) -> None:
    OUT.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_corner(name: str) -> dict:
    name = name.upper()
    if name not in BOARD_XY:
        raise ValueError(f"corner must be one of {list(BOARD_XY)}")
    bridge, config = _robot()
    pose = get_pose(bridge, config)
    angles = get_angle(bridge, config)
    j1 = angles["j1"]
    # Cross-check polar angle of TCP
    theta = math.degrees(math.atan2(pose["y"], pose["x"]))
    sample = {
        "corner": name,
        "board_xy_mm": BOARD_XY[name].tolist(),
        "tcp_xy": [round(float(pose["x"]), 3), round(float(pose["y"]), 3)],
        "tcp_z": round(float(pose["z"]), 3),
        "tcp_r": round(float(pose["r"]), 3),
        "j1_deg": round(j1, 4),
        "j2_deg": round(angles["j2"], 4),
        "j3_deg": round(angles["j3"], 4),
        "j4_deg": round(angles["j4"], 4),
        "tcp_polar_deg": round(theta, 4),
        "j1_vs_polar_deg": round(j1 - theta, 4),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    state = load_state()
    # Replace prior sample for same corner
    state["samples"] = [s for s in state["samples"] if s["corner"] != name]
    state["samples"].append(sample)
    state["status"] = "collecting"
    state["tip_xy_radius_mm"] = TIP_XY_MM
    if len(state["samples"]) >= 3:
        try:
            state["fit"] = fit_delta(state["samples"])
            state["status"] = "fitted"
        except Exception as exc:
            state["fit_error"] = str(exc)
    save_state(state)
    return {"sample": sample, "state": state}


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("interactive", "run"):
        print("Tip yaw δ from chessboard touches", flush=True)
        print(f"Model: tip = TCP + {TIP_XY_MM}*(sin(J1+δ), -cos(J1+δ))", flush=True)
        print("Touch TL/TR/BL/BR with TIP. Prefer large XY span (J1 changes).", flush=True)
        order = ["TL", "TR", "BR", "BL"]
        for name in order:
            input(f"\nPut TIP on {name}, then press Enter...")
            out = record_corner(name)
            s = out["sample"]
            print(
                json.dumps(
                    {
                        "recorded": name,
                        "tcp_xy": s["tcp_xy"],
                        "j1_deg": s["j1_deg"],
                        "tcp_polar_deg": s["tcp_polar_deg"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if out["state"].get("fit"):
                print("fit so far:", json.dumps(out["state"]["fit"], ensure_ascii=False), flush=True)
        state = load_state()
        print("\nFINAL", json.dumps(state.get("fit"), indent=2, ensure_ascii=False), flush=True)
        print("saved", OUT, flush=True)
        return 0

    if args[0] == "record":
        name = args[1] if len(args) > 1 else "TL"
        out = record_corner(name)
        print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
        return 0

    if args[0] == "fit":
        state = load_state()
        fit = fit_delta(state["samples"])
        state["fit"] = fit
        state["status"] = "fitted"
        save_state(state)
        print(json.dumps(fit, indent=2, ensure_ascii=False), flush=True)
        return 0

    if args[0] == "status":
        print(json.dumps(load_state(), indent=2, ensure_ascii=False), flush=True)
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
