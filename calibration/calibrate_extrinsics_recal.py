"""Recalibrate eye-in-hand T_end_to_camera (Trans(TCP) @ Rz(J1) @ T_ec).

Protocol:
  1) tip-touch physical TL (title-side) -> P_TL  [board must not move]
  2) LOOK L1..Ln at varied J1 (board visible; TL need not be centered)
  3) fit   -> T_ec from PnP TL vs P_TL
  4) status / apply (apply only after validate)

Usage:
  python calibration/calibrate_extrinsics_recal.py approach-tl
  python calibration/calibrate_extrinsics_recal.py touch-tl
  python calibration/calibrate_extrinsics_recal.py look <label>
  python calibration/calibrate_extrinsics_recal.py fit
  python calibration/calibrate_extrinsics_recal.py status
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

from calibrate_camera_j1_offset import (  # noqa: E402
    OUTER_IDX,
    physical_tl_open_cv_index,
    reorder_corners_origin_at,
)
from calibrate_tip_yaw_delta import get_angle  # noqa: E402
from capture_basler_intrinsics import _capture, _open_camera  # noqa: E402
from capture_extrinsics import (  # noqa: E402
    PATTERN,
    _bridge,
    _config,
    clamp_xy,
    detect_chessboard,
    get_pose,
    move,
)
from coordinate_transforms import PixelToWorld  # noqa: E402
from tip_offset import (  # noqa: E402
    load_tip_offset_params,
    tip_xy_from_tcp,
    tcp_xy_from_tip_config,
)

OUT = HERE / "_extrinsic_recal.json"
OUT_DIR = HERE / "_extrinsic_recal_session"
SQUARE_MM = 5.0


def load_state() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": "T_base_cam = Trans(TCP_xyz) @ T_ec  (R ignored)",
        "P_TL": None,
        "touch": None,
        "looks": {},
        "fit": None,
        "status": "collecting",
    }


def save_state(state: dict) -> None:
    OUT.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def capture_frame(cam_cfg: dict) -> np.ndarray:
    cam = _open_camera(cam_cfg["calibration"])
    try:
        for _ in range(3):
            _capture(cam)
        time.sleep(0.12)
        return _capture(cam)
    finally:
        cam.Close()


def detect_physical_tl(frame: np.ndarray):
    corners = detect_chessboard(frame)
    if corners is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tl_idx = physical_tl_open_cv_index(gray, corners)
    return corners, tl_idx, corners.reshape(-1, 2)[tl_idx]


def approach_tl() -> dict:
    """Move TIP above projected physical TL using current extrinsics + tip_offset."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge, config = _bridge(), _config(15)
    pose0 = get_pose(bridge, config)
    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    frame = capture_frame(cam_cfg)
    cv2.imwrite(str(OUT_DIR / "approach_raw.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    det = detect_physical_tl(frame)
    if det is None:
        raise RuntimeError("chessboard not found")
    corners, tl_idx, tl_px = det
    # Prefer ink for first approach; after touch, callers should use P_TL lock.
    vis = frame.copy()
    cv2.drawChessboardCorners(vis, PATTERN, corners, True)
    for i in OUTER_IDX:
        col = (0, 0, 255) if i == tl_idx else (255, 128, 0)
        p = corners.reshape(-1, 2)[i]
        pt = (int(p[0]), int(p[1]))
        cv2.circle(vis, pt, 18, col, 3)
        cv2.putText(
            vis,
            "TL" if i == tl_idx else str(i),
            (pt[0] + 12, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            col,
            3,
        )
    cv2.imwrite(str(OUT_DIR / "approach_tl.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    ptw = PixelToWorld(str(HERE / "camera_config.yaml"))
    ptw.set_robot_pose(pose0)
    table = ptw.pixel_to_table(float(tl_px[0]), float(tl_px[1]), pose0)
    if table is None:
        raise RuntimeError("TL projection failed")
    tip_xy = (float(table[0]), float(table[1]))
    tip_p = load_tip_offset_params()
    solved = tcp_xy_from_tip_config(tip_xy)
    tcp_x, tcp_y = clamp_xy(float(solved["tcp_xy"][0]), float(solved["tcp_xy"][1]))
    # Keep current Z for approach; user can jog down to touch
    target = {
        "x": round(tcp_x, 2),
        "y": round(tcp_y, 2),
        "z": round(float(pose0["z"]), 2),
        "r": float(pose0["r"]),
    }
    print(
        json.dumps(
            {
                "event": "approach_plan",
                "tl_idx": int(tl_idx),
                "tl_px": [round(float(tl_px[0]), 1), round(float(tl_px[1]), 1)],
                "tl_table_xy": [round(tip_xy[0], 3), round(tip_xy[1], 3)],
                "tip_params": {
                    "radius_xy_mm": tip_p["radius_xy_mm"],
                    "delta_deg": tip_p["delta_deg"],
                },
                "tcp_solve": {
                    "tcp_xy": [round(tcp_x, 3), round(tcp_y, 3)],
                    "j1_deg": round(float(solved["j1_deg"]), 4),
                    "err_mm": round(float(solved["err_mm"]), 4),
                },
                "target": target,
                "marked": str(OUT_DIR / "approach_tl.jpg"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    move(bridge, config, target)
    time.sleep(0.4)
    pose1 = get_pose(bridge, config)
    print(
        json.dumps(
            {
                "event": "approached",
                "pose": pose1,
                "note": "Jog tip XY/Z until tip sits on physical TL, then: touch-tl",
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return pose1


def touch_tl(state: dict) -> None:
    bridge, config = _bridge(), _config()
    pose = get_pose(bridge, config)
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
        "tip_params": {
            "radius_xy_mm": params["radius_xy_mm"],
            "delta_deg": params["delta_deg"],
        },
        "tip_xy": [float(tip_xy[0]), float(tip_xy[1])],
    }
    state["P_TL"] = p_tl
    state["status"] = "have_P_TL"
    save_state(state)
    print(
        json.dumps(
            {
                "event": "touch-tl",
                "P_TL": p_tl,
                "tcp": pose,
                "j1_true": angles["j1"],
                "note": "Board must NOT move. Next: raise tip, then look L1 L2 ...",
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def look(state: dict, label: str) -> None:
    if not state.get("P_TL"):
        raise RuntimeError("run touch-tl first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bridge, config = _bridge(), _config()
    pose = get_pose(bridge, config)
    angles = get_angle(bridge, config)
    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))
    K = np.array(cam_cfg["intrinsics"]["camera_matrix"], dtype=np.float64)
    D = np.array(cam_cfg["intrinsics"]["dist_coeffs"], dtype=np.float64)

    frame = capture_frame(cam_cfg)
    raw_path = OUT_DIR / f"look_{label}_raw.jpg"
    cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    corners = detect_chessboard(frame)
    if corners is None:
        raise RuntimeError(f"chessboard not found — saved {raw_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prefer = state["P_TL"][:2] if state.get("P_TL") else None
    raw_tl_idx = physical_tl_open_cv_index(
        gray,
        corners,
        prefer_xy_mm=prefer,
        robot_pose=pose,
        config_path=HERE / "camera_config.yaml",
    )
    corners = reorder_corners_origin_at(corners, raw_tl_idx)

    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
    # TL in camera = tvec (object origin)
    tl_cam = tvec.reshape(3).astype(float)
    u, v = float(corners.reshape(-1, 2)[0, 0]), float(corners.reshape(-1, 2)[0, 1])

    vis = frame.copy()
    cv2.drawChessboardCorners(vis, PATTERN, corners, True)
    cv2.circle(vis, (int(u), int(v)), 20, (0, 0, 255), 3)
    cv2.putText(
        vis,
        f"{label} physTL=({u:.0f},{v:.0f}) rawIdx={raw_tl_idx} J1={angles['j1']:.2f} pnp={rmse:.2f}px",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    marked = OUT_DIR / f"look_{label}_marked.jpg"
    cv2.imwrite(str(marked), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    sample = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "tcp": {k: float(pose[k]) for k in ("x", "y", "z", "r")},
        "j1_deg": float(angles["j1"]),
        "j2_deg": float(angles["j2"]),
        "j3_deg": float(angles["j3"]),
        "j4_deg": float(angles["j4"]),
        "raw_tl_idx": int(raw_tl_idx),
        "tl_px": [u, v],
        "tl_cam_mm": [float(tl_cam[0]), float(tl_cam[1]), float(tl_cam[2])],
        "pnp_rmse_px": rmse,
        "raw": str(raw_path),
        "marked": str(marked),
    }
    state["looks"][label] = sample
    save_state(state)
    print(json.dumps({"event": "look", "label": label, **sample}, indent=2, ensure_ascii=False), flush=True)


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    H = X.T @ Y
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    errs = [float(np.linalg.norm(dst[i] - (R @ src[i] + t))) for i in range(len(src))]
    return R, t, errs


def _tl_cam_candidates_from_raw(raw_path: Path, cam_cfg: dict) -> dict[int, np.ndarray] | None:
    """For one LOOK image, PnP tl_cam under each outer-corner-as-origin hypothesis."""
    frame = cv2.imread(str(raw_path))
    if frame is None:
        return None
    corners = detect_chessboard(frame)
    if corners is None:
        return None
    K = np.array(cam_cfg["intrinsics"]["camera_matrix"], dtype=np.float64)
    D = np.array(cam_cfg["intrinsics"]["dist_coeffs"], dtype=np.float64)
    obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
    out: dict[int, np.ndarray] = {}
    for tl_idx in OUTER_IDX:
        ordered = reorder_corners_origin_at(corners, tl_idx)
        ok, _rvec, tvec = cv2.solvePnP(obj, ordered, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            out[int(tl_idx)] = tvec.reshape(3).astype(np.float64)
    return out or None


def fit(state: dict) -> dict:
    """Fit T_ec in J1 end frame: Rz(-J1)@(P_TL-TCP) = R_ec @ tl_cam + t_ec.

    Ink-based physical-TL ID can flip across viewpoints. Prefer a per-look outer-
    corner hypothesis search that minimizes Kabsch RMS against P_TL (still one
    rigid T_ec). Falls back to stored tl_cam_mm if raw images are missing.
    """
    import itertools

    from eye_in_hand_xyz import rot_z  # noqa: E402

    if not state.get("P_TL"):
        raise RuntimeError("missing P_TL")
    looks = state.get("looks") or {}
    if len(looks) < 3:
        raise RuntimeError("need >=3 looks")

    p_tl = np.array(state["P_TL"][:3], dtype=np.float64)
    cam_cfg = yaml.safe_load((HERE / "camera_config.yaml").read_text(encoding="utf-8"))

    labels: list[str] = []
    tcps: list[np.ndarray] = []
    j1s: list[float] = []
    cand_opts: list[dict[int, np.ndarray]] = []
    for lab, s in sorted(looks.items()):
        tcp = np.array([s["tcp"]["x"], s["tcp"]["y"], s["tcp"]["z"]], dtype=np.float64)
        j1 = float(s.get("j1_deg", math.degrees(math.atan2(tcp[1], tcp[0]))))
        opts = None
        raw = s.get("raw")
        if raw:
            opts = _tl_cam_candidates_from_raw(Path(raw), cam_cfg)
        if not opts:
            # fallback: only the originally stored origin
            stored_idx = int(s.get("raw_tl_idx", 0))
            opts = {stored_idx: np.array(s["tl_cam_mm"], dtype=np.float64)}
        labels.append(lab)
        tcps.append(tcp)
        j1s.append(j1)
        cand_opts.append(opts)

    best = None
    for choice in itertools.product(*[tuple(opts.keys()) for opts in cand_opts]):
        srcs = np.stack([cand_opts[i][choice[i]] for i in range(len(labels))], axis=0)
        dsts = np.stack(
            [rot_z(-j1s[i]) @ (p_tl - tcps[i]) for i in range(len(labels))],
            axis=0,
        )
        R, t, errs = _kabsch(srcs, dsts)
        rms = float(np.sqrt(np.mean(np.square(errs))))
        if best is None or rms < best["rms"]:
            best = {
                "rms": rms,
                "R": R,
                "t": t,
                "errs": errs,
                "choice": {labels[i]: int(choice[i]) for i in range(len(labels))},
                "srcs": srcs,
                "dsts": dsts,
            }

    assert best is not None
    R, t, errs = best["R"], best["t"], best["errs"]
    # Write back corrected tl_cam / raw_tl_idx for the winning hypothesis
    for i, lab in enumerate(labels):
        looks[lab]["raw_tl_idx"] = best["choice"][lab]
        looks[lab]["tl_cam_mm"] = [float(v) for v in best["srcs"][i]]
        looks[lab]["tl_origin_resolved"] = True

    per = {}
    for lab, e, src_pt, tcp, j1, dst in zip(
        labels, errs, best["srcs"], tcps, j1s, best["dsts"]
    ):
        pred = R @ src_pt + t
        err = dst - pred
        per[lab] = {
            "err_mm": round(float(e), 3),
            "err_xyz_mm": [round(float(v), 3) for v in err],
            "raw_tl_idx": best["choice"][lab],
            "j1_deg": round(float(j1), 3),
        }
    fit_out = {
        "n_looks": len(labels),
        "R": R.tolist(),
        "t_mm": [round(float(v), 4) for v in t],
        "rms_mm": round(float(best["rms"]), 3),
        "mean_abs_mm": round(float(np.mean(np.abs(errs))), 3),
        "max_mm": round(float(np.max(errs)), 3),
        "j1_span_deg": round(float(max(j1s) - min(j1s)), 3),
        "tl_idx_choice": best["choice"],
        "per_look": per,
        "model": "T_base_cam = Trans(TCP) @ Rz(J1) @ T_end_to_camera",
        "note": (
            "T_ec maps tl_cam -> Rz(-J1)@(P_TL-TCP); runtime "
            "T_base_cam = Trans(TCP) @ Rz(J1) @ T_ec. "
            "tl_idx_choice = per-look outer-corner origin that minimizes fit RMS "
            "(ink detector alone can flip)."
        ),
    }
    state["looks"] = looks
    state["fit"] = fit_out
    state["status"] = "fitted"
    save_state(state)
    print(json.dumps(fit_out, indent=2, ensure_ascii=False), flush=True)
    return fit_out


def status(state: dict) -> None:
    looks = state.get("looks") or {}
    j1s = [float(v["j1_deg"]) for v in looks.values()]
    span = (max(j1s) - min(j1s)) if j1s else 0.0
    print(
        json.dumps(
            {
                "status": state.get("status"),
                "P_TL": state.get("P_TL"),
                "n_looks": len(looks),
                "look_labels": sorted(looks.keys()),
                "j1_span_deg": round(span, 3),
                "fit": state.get("fit"),
                "next": (
                    "approach-tl -> jog tip onto TL -> touch-tl -> look L1.."
                    if not state.get("P_TL")
                    else (
                        "more looks (J1 span>=25) then fit"
                        if len(looks) < 3 or span < 20
                        else "fit then validate tip-to-projected-TL"
                    )
                ),
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
    if cmd == "approach-tl":
        approach_tl()
        return 0
    if cmd == "touch-tl":
        touch_tl(state)
        return 0
    if cmd == "look":
        if len(argv) < 3:
            print("Usage: look <label>")
            return 1
        look(state, argv[2])
        return 0
    if cmd == "fit":
        fit(state)
        return 0
    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
