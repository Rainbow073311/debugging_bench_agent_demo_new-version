"""Calibrate camera mount in the J1-rotating end frame (like tip).

Model:
  T_base_camera = Trans(TCP_xyz) @ Rz(J1) @ T_end_to_camera
  T_end_to_camera is fixed in the arm-head frame (does NOT use tip delta).

Protocol (chessboard fixed after touch):
  1) tip-touch TL once -> P_TL
  2) LOOK: any pose where the full board is visible and TL is detected
     (TL need NOT be at image center). Change TCP so true J1 spans >=20 deg.
  3) fit T_end_to_camera from PnP TL rays / board origin in camera

Usage:
  python calibration/calibrate_camera_j1_offset.py status
  python calibration/calibrate_camera_j1_offset.py touch-tl
  python calibration/calibrate_camera_j1_offset.py look L1
  python calibration/calibrate_camera_j1_offset.py fit
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
from capture_extrinsics import (  # noqa: E402
    FIXED_R,
    PATTERN,
    SQUARE_MM,
    _bridge,
    _config,
    detect_chessboard,
    get_pose,
    pnp,
)
from calibrate_tip_yaw_delta import get_angle  # noqa: E402
from tip_offset import tip_xy_from_tcp, load_tip_offset_params  # noqa: E402
from eye_in_hand_xyz import make_transform, _fit_rigid_translation_model  # noqa: E402

OUT = HERE / "_camera_j1_offset.json"
OUT_DIR = HERE / "_camera_j1_session"
OUTER_IDX = (0, 8, 45, 53)  # OpenCV outer inner-corners for 9x6
IMG_W, IMG_H = 2448, 2048


def _ink_mask(gray: np.ndarray, corners_xy: np.ndarray) -> np.ndarray:
    """Binary ink outside the chessboard (paper title / ruler)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hull = cv2.convexHull(corners_xy.astype(np.float32))
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    # Cover outer chessboard squares beyond inner corners (~1.3 square).
    sq = float(
        np.mean(
            [
                np.linalg.norm(corners_xy[8] - corners_xy[0]) / 8.0,
                np.linalg.norm(corners_xy[45] - corners_xy[0]) / 5.0,
            ]
        )
    )
    rad = max(25, int(round(1.3 * sq)))
    mask = cv2.dilate(mask, np.ones((rad, rad), np.uint8))
    return cv2.bitwise_and(bw, cv2.bitwise_not(mask))


def _ink_centroid(gray: np.ndarray, corners_xy: np.ndarray) -> np.ndarray | None:
    """Rough centroid of dark ink on paper outside the chessboard (title/ruler)."""
    ink = _ink_mask(gray, corners_xy)
    ys, xs = np.where(ink > 0)
    if len(xs) < 50:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _warp_long_edge_strip(
    gray: np.ndarray, pts: np.ndarray, i0: int, i1: int, board_c: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Warp gray strip outside one long edge (beyond outer squares) + local ink mask."""
    a, b = pts[i0], pts[i1]
    mid = 0.5 * (a + b)
    outward = mid - board_c
    n = float(np.linalg.norm(outward))
    if n < 1e-6:
        return None, None
    outward /= n
    el = float(np.linalg.norm(b - a))
    sq = el / 8.0
    inset = 1.05 * sq
    depth = 3.0 * sq
    p0 = a - 0.08 * (b - a) + outward * inset
    p1 = b + 0.08 * (b - a) + outward * inset
    p2 = p1 + outward * depth
    p3 = p0 + outward * depth
    src = np.array([p0, p1, p2, p3], dtype=np.float32)
    Ww, Hh = max(32, int(round(el * 1.16))), max(24, int(round(depth)))
    dst = np.array(
        [[0.0, 0.0], [Ww - 1.0, 0.0], [Ww - 1.0, Hh - 1.0], [0.0, Hh - 1.0]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(gray, M, (Ww, Hh), flags=cv2.INTER_LINEAR)
    if crop.size == 0:
        return None, None
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return crop, bw


def _long_edge_camera_score(
    gray: np.ndarray, pts: np.ndarray, i0: int, i1: int, board_c: np.ndarray
) -> dict:
    """Score how much a long-edge margin looks like Camera-title text (not ruler).

    Camera title: medium ink density, letter gaps (transitions), not Otsu-saturated.
    Ruler / empty: thin ticks or saturated false ink on bright paper.
    """
    crop, bw = _warp_long_edge_strip(gray, pts, i0, i1, board_c)
    if bw is None:
        return {"score": -1.0, "dens": 0.0, "occ": 0.0, "trans": 0.0, "longest": 0}
    dens = float((bw > 0).mean())
    prof = (bw > 0).max(axis=0).astype(np.float64)
    occ = float((prof > 0.5).mean())
    trans = float(np.sum(np.abs(np.diff(prof))))
    runs: list[int] = []
    n = 0
    for v in prof:
        if v > 0.5:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    longest = int(max(runs) if runs else 0)
    longest_frac = float(longest) / float(max(1, len(prof)))

    # Saturated mask = bad Otsu on nearly blank / ruler strip.
    if occ > 0.97 or dens > 0.45:
        score = dens * 0.05
    else:
        # Prefer title-like: some ink, gaps (trans), long-ish text run but not full width.
        score = (
            dens * 3.0
            + min(trans, 40.0) * 0.04
            + min(longest_frac, 0.85) * 0.8
            - abs(occ - 0.75) * 0.5
        )
    return {
        "score": float(score),
        "dens": dens,
        "occ": occ,
        "trans": trans,
        "longest": longest,
    }


def _long_edge_ink_density(
    gray: np.ndarray, pts: np.ndarray, i0: int, i1: int, board_c: np.ndarray
) -> float:
    """Back-compat: ink density in Camera-title margin strip."""
    return float(_long_edge_camera_score(gray, pts, i0, i1, board_c)["dens"])


def physical_tl_open_cv_index(
    gray: np.ndarray,
    corners,
    *,
    prefer_xy_mm: tuple[float, float] | list[float] | None = None,
    robot_pose: dict | None = None,
    config_path: str | Path | None = None,
) -> int:
    """Return OpenCV corner index of the printed-sheet physical TL.

    Definition (human reading order on the paper):
      Rotate the sheet so \"Camera Calibration Checkerboard\" reads left→right
      above the grid. Physical TL = the inner corner under that title on the
      LEFT (under the word \"Camera\").

    OpenCV's index-0 flips with viewpoint; this returns whichever current
    outer index sits on that physical corner.

    Prefer order:
      1) If prefer_xy_mm is given (locked tip P_TL), nearest projected outer corner.
      2) Else: Camera-title long edge (text-like margin, not verification ruler),
         then the left endpoint when +up points from board toward that title.
    """
    pts = corners.reshape(-1, 2).astype(np.float64)

    if prefer_xy_mm is not None and robot_pose is not None:
        try:
            from coordinate_transforms import PixelToWorld

            path = Path(config_path) if config_path else HERE / "camera_config.yaml"
            ptw = PixelToWorld(str(path), robot_pose=robot_pose)
            target = np.array([float(prefer_xy_mm[0]), float(prefer_xy_mm[1])], dtype=np.float64)
            best_i, best_e = None, None
            for i in OUTER_IDX:
                table = ptw.pixel_to_table(float(pts[i, 0]), float(pts[i, 1]), robot_pose)
                if table is None:
                    continue
                err = float(np.linalg.norm(np.array(table[:2], float) - target))
                if best_e is None or err < best_e:
                    best_i, best_e = int(i), err
            if best_i is not None:
                return best_i
        except Exception:
            pass

    board_c = pts.mean(axis=0)
    # 9x6 pattern: long edges are always (0,8) and (45,53).
    long_edges = ((0, 8), (45, 53))
    scores = {
        edge: _long_edge_camera_score(gray, pts, edge[0], edge[1], board_c)
        for edge in long_edges
    }
    cam_edge = max(long_edges, key=lambda e: scores[e]["score"])
    if scores[cam_edge]["score"] < 0.02:
        raise RuntimeError("cannot locate Camera-title margin ink for TL disambiguation")

    mid = 0.5 * (pts[cam_edge[0]] + pts[cam_edge[1]])
    up = mid - board_c  # board → Camera-title side
    n = float(np.linalg.norm(up))
    if n < 1e-6:
        raise RuntimeError("degenerate Camera-title edge geometry")
    up /= n
    # Image coords (x right, y down): when +up points to title above the board
    # in reading orientation, +left is (up_y, -up_x).
    left = np.array([up[1], -up[0]], dtype=np.float64)
    i_left, i_right = cam_edge[0], cam_edge[1]
    if float(np.dot(pts[i_right] - board_c, left)) > float(np.dot(pts[i_left] - board_c, left)):
        i_left, i_right = i_right, i_left
    return int(i_left)


def reorder_corners_origin_at(corners, tl_idx: int):
    """Permute 9x6 corners so physical TL becomes index 0 (object origin)."""
    w, h = PATTERN
    pts = corners.reshape(h, w, 2)
    if tl_idx == 0:
        out = pts
    elif tl_idx == 8:
        out = pts[:, ::-1, :]
    elif tl_idx == 45:
        out = pts[::-1, :, :]
    elif tl_idx == 53:
        out = pts[::-1, ::-1, :]
    else:
        raise ValueError(f"tl_idx must be one of {OUTER_IDX}, got {tl_idx}")
    return out.reshape(-1, 1, 2).astype(np.float32)


def _load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "fixed_r_deg": FIXED_R,
        "P_TL": None,
        "touch": None,
        "looks": {},
        "fit": None,
    }


def _save(state: dict) -> None:
    OUT.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _j1_proxy(x: float, y: float) -> float:
    return math.degrees(math.atan2(y, x))


def _rot_z(j1_deg: float) -> np.ndarray:
    a = math.radians(j1_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def status(state: dict) -> None:
    looks = state.get("looks") or {}
    j1s = [float(v["j1_deg"]) for v in looks.values() if "j1_deg" in v]
    span = (max(j1s) - min(j1s)) if j1s else 0.0
    print(
        json.dumps(
            {
                "P_TL": state.get("P_TL"),
                "n_looks": len(looks),
                "look_labels": sorted(looks.keys()),
                "j1_span_deg": round(span, 2),
                "j1_list": [round(j, 2) for j in sorted(j1s)],
                "fit": state.get("fit"),
                "ready_to_fit": bool(state.get("P_TL")) and len(looks) >= 3 and span >= 12.0,
                "hint": (
                    "OK to fit"
                    if (state.get("P_TL") and len(looks) >= 3 and span >= 12.0)
                    else "Need P_TL + >=3 looks with J1 span >=12 deg (prefer >=20). "
                    "TL only needs to be visible/detected — not at image center."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def touch_tl(state: dict) -> None:
    bridge = _bridge()
    config = _config()
    pose = get_pose(bridge, config)
    if pose is None:
        raise RuntimeError("GetPose failed — enable robot / check dashboard connection")
    angles = get_angle(bridge, config)
    params = load_tip_offset_params()
    tip_xy = tip_xy_from_tcp(
        (pose["x"], pose["y"]),
        j1_deg=float(angles["j1"]),
        radius_xy_mm=float(params["radius_xy_mm"]),
        delta_deg=float(params["delta_deg"]),
    )
    p_tl = [float(tip_xy[0]), float(tip_xy[1]), float(pose["z"])]
    state["touch"] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "tcp": pose,
        "angles": angles,
        "tip_params": params,
        "tip_xy": [float(tip_xy[0]), float(tip_xy[1])],
    }
    state["P_TL"] = p_tl
    _save(state)
    print(
        json.dumps(
            {
                "event": "touch-tl",
                "tcp": pose,
                "j1_true": angles["j1"],
                "P_TL": p_tl,
                "note": "Board must NOT move. Next LOOK: board visible is enough; TL need not be centered.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def look(state: dict, label: str) -> None:
    if not state.get("P_TL"):
        raise RuntimeError("Run touch-tl first so P_TL is known")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge = _bridge()
    config = _config()
    pose = get_pose(bridge, config)
    if pose is None:
        raise RuntimeError("GetPose failed — enable robot / check dashboard")
    angles = get_angle(bridge, config)
    j1 = float(angles["j1"])

    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    K = np.array(cam_cfg["intrinsics"]["camera_matrix"], dtype=np.float64)
    D = np.array(cam_cfg["intrinsics"]["dist_coeffs"], dtype=np.float64)

    cam = _open_camera(cam_cfg["calibration"])
    try:
        for _ in range(3):
            _capture(cam)
        time.sleep(0.15)
        frame = _capture(cam)
    finally:
        cam.Close()

    raw = OUT_DIR / f"look_{label}_raw.jpg"
    cv2.imwrite(str(raw), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    corners = detect_chessboard(frame)
    if corners is None:
        raise RuntimeError(f"chessboard not found — saved {raw}; show more of the board and retry")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw_tl_idx = physical_tl_open_cv_index(gray, corners)
    corners = reorder_corners_origin_at(corners, raw_tl_idx)

    pnp_res = pnp(corners, K, D)
    if pnp_res is None:
        raise RuntimeError("solvePnP failed")
    rvec, tvec, rmse = pnp_res
    # Board object origin is physical TL (0,0,0) -> TL in camera frame is tvec
    tl_cam = tvec.reshape(3).astype(float)

    pts = corners.reshape(-1, 2)
    u, v = float(pts[0][0]), float(pts[0][1])
    cx, cy = IMG_W / 2.0, IMG_H / 2.0
    center_err_px = float(math.hypot(u - cx, v - cy))

    marked = frame.copy()
    for p in pts:
        cv2.circle(marked, (int(p[0]), int(p[1])), 3, (80, 180, 255), 1)
    # also mark the four outers before confusion: emphasize canonical TL
    cv2.drawMarker(marked, (int(u), int(v)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 36, 3)
    cv2.putText(
        marked,
        f"{label} physTL=({u:.0f},{v:.0f}) rawIdx={raw_tl_idx} J1={j1:.2f} pnp={rmse:.2f}px",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
    )
    marked_path = OUT_DIR / f"look_{label}_marked.jpg"
    cv2.imwrite(str(marked_path), marked, [cv2.IMWRITE_JPEG_QUALITY, 92])

    p = np.array(state["P_TL"], dtype=np.float64)
    tcp = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
    # Express (P - TCP) in the J1 end frame: y_end = Rz(-J1) @ (P - TCP)
    y_end = _rot_z(-j1) @ (p - tcp)

    rec = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "tcp": pose,
        "angles": angles,
        "j1_deg": j1,
        "j1_proxy_atan2_deg": _j1_proxy(pose["x"], pose["y"]),
        "tl_pixel": [u, v],
        "opencv_raw_tl_index": raw_tl_idx,
        "image_center_err_px": center_err_px,
        "tl_in_camera_mm": tl_cam.tolist(),
        "p_minus_tcp_in_end_mm": y_end.tolist(),
        "pnp_rmse_px": float(rmse),
        "raw": str(raw),
        "marked": str(marked_path),
    }
    state.setdefault("looks", {})[label] = rec
    _save(state)
    print(json.dumps({"event": "look", **rec}, indent=2, ensure_ascii=False), flush=True)
    print(
        "OK — physical TL locked via title-side corner (not raw OpenCV index 0).",
        flush=True,
    )


def reprocess(state: dict) -> None:
    """Recompute physical-TL / PnP for all saved looks from raw images (no robot)."""
    if not state.get("P_TL"):
        raise RuntimeError("missing P_TL")
    looks = state.get("looks") or {}
    if not looks:
        raise RuntimeError("no looks to reprocess")

    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    K = np.array(cam_cfg["intrinsics"]["camera_matrix"], dtype=np.float64)
    D = np.array(cam_cfg["intrinsics"]["dist_coeffs"], dtype=np.float64)
    p_tl = np.array(state["P_TL"], dtype=np.float64)

    for label, rec in sorted(looks.items()):
        raw_path = Path(rec["raw"])
        if not raw_path.exists():
            alt = OUT_DIR / f"look_{label}_raw.jpg"
            if not alt.exists():
                raise RuntimeError(f"missing raw image for {label}")
            raw_path = alt
        frame = cv2.imread(str(raw_path))
        if frame is None:
            raise RuntimeError(f"cannot read {raw_path}")
        corners = detect_chessboard(frame)
        if corners is None:
            raise RuntimeError(f"{label}: chessboard not detected in {raw_path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_tl_idx = physical_tl_open_cv_index(gray, corners)
        corners = reorder_corners_origin_at(corners, raw_tl_idx)
        pnp_res = pnp(corners, K, D)
        if pnp_res is None:
            raise RuntimeError(f"{label}: solvePnP failed")
        _, tvec, rmse = pnp_res
        tl_cam = tvec.reshape(3).astype(float)
        pts = corners.reshape(-1, 2)
        u, v = float(pts[0][0]), float(pts[0][1])
        pose = rec["tcp"]
        j1 = float(rec["j1_deg"])
        tcp = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
        y_end = _rot_z(-j1) @ (p_tl - tcp)

        marked = frame.copy()
        for p in pts:
            cv2.circle(marked, (int(p[0]), int(p[1])), 3, (80, 180, 255), 1)
        cv2.drawMarker(marked, (int(u), int(v)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 36, 3)
        cv2.putText(
            marked,
            f"{label} physTL=({u:.0f},{v:.0f}) rawIdx={raw_tl_idx} J1={j1:.2f} pnp={rmse:.2f}px",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )
        marked_path = OUT_DIR / f"look_{label}_marked.jpg"
        cv2.imwrite(str(marked_path), marked, [cv2.IMWRITE_JPEG_QUALITY, 92])

        rec.update(
            {
                "tl_pixel": [u, v],
                "opencv_raw_tl_index": raw_tl_idx,
                "image_center_err_px": float(math.hypot(u - IMG_W / 2.0, v - IMG_H / 2.0)),
                "tl_in_camera_mm": tl_cam.tolist(),
                "p_minus_tcp_in_end_mm": y_end.tolist(),
                "pnp_rmse_px": float(rmse),
                "raw": str(raw_path),
                "marked": str(marked_path),
                "reprocessed": datetime.now().isoformat(timespec="seconds"),
            }
        )
        print(
            json.dumps(
                {
                    "label": label,
                    "raw_idx": raw_tl_idx,
                    "tl_pixel": [round(u, 1), round(v, 1)],
                    "pnp_rmse_px": round(float(rmse), 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    state["looks"] = looks
    _save(state)
    print(json.dumps({"event": "reprocess", "n": len(looks)}, ensure_ascii=False), flush=True)


def fit(state: dict) -> None:
    looks = list((state.get("looks") or {}).values())
    if not state.get("P_TL"):
        raise RuntimeError("missing P_TL")
    if len(looks) < 3:
        raise RuntimeError(f"need >=3 looks, have {len(looks)}")
    j1s = np.array([float(r["j1_deg"]) for r in looks], dtype=np.float64)
    span = float(j1s.max() - j1s.min())
    if span < 12.0:
        raise RuntimeError(f"J1 span only {span:.1f} deg — move to more different arm angles")

    # y_end = R_ec @ tl_cam + t_ec
    xs = np.array([r["tl_in_camera_mm"] for r in looks], dtype=np.float64)
    ys = np.array([r["p_minus_tcp_in_end_mm"] for r in looks], dtype=np.float64)

    # Full 3D Kabsch can be rank-deficient with few views; fall back to XY Kabsch + tz.
    method = "kabsch3d"
    try:
        R, t, singular = _fit_rigid_translation_model(
            xs, ys, minimum_observability_ratio=1e-8
        )
    except ValueError:
        method = "kabsch2d_xy_plus_tz"
        x2 = xs[:, :2]
        y2 = ys[:, :2]
        cx, cy = x2.mean(axis=0), y2.mean(axis=0)
        X = x2 - cx
        Y = y2 - cy
        u, svals, vt = np.linalg.svd(Y.T @ X)
        R2 = u @ vt
        if np.linalg.det(R2) < 0:
            u[:, -1] *= -1
            R2 = u @ vt
        t2 = cy - R2 @ cx
        R = np.eye(3)
        R[:2, :2] = R2
        # tz so that mean(yz) ≈ mean((R@x)_z) + tz  -> tz = mean(yz - xz) with R≈I on z
        xz = (R @ xs.T).T[:, 2]
        tz = float(np.mean(ys[:, 2] - xz))
        t = np.array([t2[0], t2[1], tz], dtype=np.float64)
        singular = np.array([svals[0], svals[1] if len(svals) > 1 else 0.0, 0.0])

    residuals = [float(np.linalg.norm(y - (R @ x + t))) for x, y in zip(xs, ys)]
    resid_xy = [
        float(np.linalg.norm((y - (R @ x + t))[:2])) for x, y in zip(xs, ys)
    ]
    T = make_transform(R, t)
    radius_xy = float(np.linalg.norm(t[:2]))
    bearing = float(math.degrees(math.atan2(t[0], -t[1]))) if radius_xy > 1e-6 else 0.0

    fit_rec = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "n": len(looks),
        "j1_span_deg": round(span, 3),
        "method": method,
        "T_end_to_camera": {
            "R": R.tolist(),
            "t_mm": [round(float(v), 4) for v in t.tolist()],
        },
        "p_camera_local_xyz_mm": [round(float(v), 3) for v in t.tolist()],
        "radius_xy_mm": round(radius_xy, 3),
        "bearing_from_minus_y_deg": round(bearing, 3),
        "mean_residual_mm": round(float(np.mean(residuals)), 3),
        "max_residual_mm": round(float(np.max(residuals)), 3),
        "mean_residual_xy_mm": round(float(np.mean(resid_xy)), 3),
        "max_residual_xy_mm": round(float(np.max(resid_xy)), 3),
        "residuals_mm": [round(v, 3) for v in residuals],
        "residuals_xy_mm": [round(v, 3) for v in resid_xy],
        "singular_values": [round(float(v), 4) for v in np.asarray(singular).tolist()],
        "model": "T_base_cam = Trans(TCP_xyz) @ Rz(J1_true) @ T_end_to_camera",
        "note": (
            "Independent of tip delta. Optional: add 1-2 looks at different HEIGHT "
            "(same J1 ok) to strengthen Z observability, then re-fit."
        ),
    }
    state["fit"] = fit_rec
    _save(state)
    print(json.dumps({"event": "fit", **fit_rec}, indent=2, ensure_ascii=False), flush=True)


def main(argv: list[str]) -> int:
    state = _load()
    if len(argv) < 2 or argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd = argv[1]
    if cmd == "status":
        status(state)
    elif cmd == "touch-tl":
        touch_tl(state)
    elif cmd == "look":
        if len(argv) < 3:
            raise SystemExit("usage: look <label>")
        look(state, argv[2])
    elif cmd == "fit":
        fit(state)
    elif cmd == "reprocess":
        reprocess(state)
        fit(state)
    else:
        raise SystemExit(f"unknown command: {cmd}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
