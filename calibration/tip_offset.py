"""Tip XY offset vs MG400 J1 (R-independent).

Model (delta_deg = 0 => pure -J1 tangential):
  dx = radius_mm * sin(J1_deg + delta_deg)
  dy = -radius_mm * cos(J1_deg + delta_deg)
  tip_xy = TCP_xy + [dx, dy]

On MG400, J1 ≈ atan2(TCP_y, TCP_x) (deg).  Planning therefore solves the
implicit map TCP -> tip without requiring a live GetAngle() read, then can
optionally refine with a measured J1 after motion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "camera_config.yaml"


def load_tip_offset_params(config_path: str | Path | None = None) -> dict:
    """Load locked tip offset. Prefer tip_offset.json, else camera_config.yaml."""
    json_path = HERE / "tip_offset.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("status") in ("calibrated", "approx_calibrated"):
            return {
                "radius_xy_mm": float(data.get("radius_xy_mm", 27.0)),
                "delta_deg": float(data.get("delta_deg", 0.0)),
                "z_mm": float(data.get("z_mm", 0.0)),
                "model": data.get("model", "j1_tangential_xy"),
                "locked": bool(data.get("locked", False)),
            }

    path = Path(config_path) if config_path else DEFAULT_CONFIG
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tip = data.get("tip_offset") or {}
    if tip.get("status") not in ("approx_calibrated", "calibrated"):
        raise ValueError("tip_offset is not calibrated in camera_config.yaml")
    return {
        "radius_xy_mm": float(tip.get("radius_xy_mm", 27.0)),
        "delta_deg": float(tip.get("delta_deg", 0.0)),
        "z_mm": float(tip.get("z_mm", 0.0)),
        "model": tip.get("model", "j1_tangential_xy"),
        "locked": bool(tip.get("locked", False)),
    }


def j1_from_tcp_xy(x_mm: float, y_mm: float) -> float:
    """MG400 base angle proxy: J1 ≈ atan2(y, x) in degrees."""
    return math.degrees(math.atan2(float(y_mm), float(x_mm)))


def tip_offset_xy(
    j1_deg: float,
    *,
    radius_xy_mm: float = 27.0,
    delta_deg: float = 0.0,
) -> tuple[float, float]:
    """Return (dx, dy) for tip = TCP + offset at this J1."""
    ang = math.radians(float(j1_deg) + float(delta_deg))
    return (radius_xy_mm * math.sin(ang), -radius_xy_mm * math.cos(ang))


def tip_xy_from_tcp(
    tcp_xy: Sequence[float],
    j1_deg: float | None = None,
    *,
    radius_xy_mm: float = 27.0,
    delta_deg: float = 0.0,
) -> tuple[float, float]:
    x, y = float(tcp_xy[0]), float(tcp_xy[1])
    if j1_deg is None:
        j1_deg = j1_from_tcp_xy(x, y)
    dx, dy = tip_offset_xy(j1_deg, radius_xy_mm=radius_xy_mm, delta_deg=delta_deg)
    return (x + dx, y + dy)


def tcp_xy_from_tip(
    tip_xy: Sequence[float],
    *,
    radius_xy_mm: float = 27.0,
    delta_deg: float = 0.0,
    j1_hint_deg: float | None = None,
    iterations: int = 8,
    tol_mm: float = 1e-4,
) -> dict:
    """Solve TCP_xy so that tip_xy_from_tcp(TCP, J1(TCP)) == tip_xy.

    Uses J1≈atan2(TCP_y, TCP_x). Returns dict with tcp_xy, j1_deg, offset_xy, err_mm.
    """
    tx, ty = float(tip_xy[0]), float(tip_xy[1])
    if j1_hint_deg is None:
        j1 = j1_from_tcp_xy(tx, ty)
    else:
        j1 = float(j1_hint_deg)

    x = y = 0.0
    dx = dy = 0.0
    for _ in range(max(1, iterations)):
        dx, dy = tip_offset_xy(j1, radius_xy_mm=radius_xy_mm, delta_deg=delta_deg)
        x, y = tx - dx, ty - dy
        j1_new = j1_from_tcp_xy(x, y)
        if abs(j1_new - j1) < 1e-6 and math.hypot((x + dx) - tx, (y + dy) - ty) < tol_mm:
            j1 = j1_new
            break
        j1 = j1_new

    tip_chk = (x + dx, y + dy)
    err = math.hypot(tip_chk[0] - tx, tip_chk[1] - ty)
    return {
        "tcp_xy": (x, y),
        "j1_deg": j1,
        "offset_xy": (dx, dy),
        "tip_check_xy": tip_chk,
        "err_mm": err,
        "radius_xy_mm": radius_xy_mm,
        "delta_deg": delta_deg,
    }


def tcp_xy_from_tip_config(
    tip_xy: Sequence[float],
    config_path: str | Path | None = None,
    **kwargs,
) -> dict:
    params = load_tip_offset_params(config_path)
    return tcp_xy_from_tip(
        tip_xy,
        radius_xy_mm=params["radius_xy_mm"],
        delta_deg=params["delta_deg"],
        **kwargs,
    )


def offset_table(
    j1_min_deg: float = -30.0,
    j1_max_deg: float = 30.0,
    step_deg: float = 1.0,
    *,
    radius_xy_mm: float = 27.0,
    delta_deg: float = 0.0,
) -> list[dict]:
    """Optional lookup table: each J1 -> (dx, dy). Prefer closed-form over table."""
    rows = []
    j1 = j1_min_deg
    while j1 <= j1_max_deg + 1e-9:
        dx, dy = tip_offset_xy(j1, radius_xy_mm=radius_xy_mm, delta_deg=delta_deg)
        rows.append(
            {
                "j1_deg": round(j1, 3),
                "dx_mm": round(dx, 4),
                "dy_mm": round(dy, 4),
                "dir_deg": round(math.degrees(math.atan2(dy, dx)), 3),
            }
        )
        j1 += step_deg
    return rows
