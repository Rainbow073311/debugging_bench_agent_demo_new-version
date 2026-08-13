"""Refit extrinsic recal using P_TL-nearest corner on every LOOK, then hover-validate."""
from __future__ import annotations

import json
import math
import shutil
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
from capture_basler_intrinsics import _capture, _open_camera  # noqa: E402
from capture_extrinsics import (  # noqa: E402
    PATTERN,
    _bridge,
    _config,
    clamp_xy,
    detect_chessboard,
    move,
)
from coordinate_transforms import PixelToWorld  # noqa: E402
from eye_in_hand_xyz import rot_z  # noqa: E402
from tip_offset import tcp_xy_from_tip_config  # noqa: E402

CFG = HERE / "camera_config.yaml"
STATE = HERE / "_extrinsic_recal.json"
OUT = HERE / "_extrinsic_recal_session"
SQUARE = 5.0


def kabsch(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    x, y = src - mu_s, dst - mu_d
    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        r = vt.T @ u.T
    t = mu_d - r @ mu_s
    errs = [float(np.linalg.norm(dst[i] - (r @ src[i] + t))) for i in range(len(src))]
    rms = float(np.sqrt(np.mean(np.square(errs))))
    return r, t, rms, errs


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    p_tl = np.array(state["P_TL"][:3], float)
    cam_cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    k = np.array(cam_cfg["intrinsics"]["camera_matrix"], float)
    d = np.array(cam_cfg["intrinsics"]["dist_coeffs"], float)

    srcs, dsts, meta = [], [], []
    for lab, s in sorted((state.get("looks") or {}).items()):
        frame = cv2.imread(s["raw"])
        if frame is None:
            print("skip missing", lab, flush=True)
            continue
        corners = detect_chessboard(frame)
        if corners is None:
            print("skip no board", lab, flush=True)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pose = s["tcp"]
        idx = physical_tl_open_cv_index(
            gray,
            corners,
            prefer_xy_mm=p_tl[:2],
            robot_pose=pose,
            config_path=CFG,
        )
        ordered = reorder_corners_origin_at(corners, idx)
        obj = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
        obj[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2) * SQUARE
        ok, _rvec, tvec = cv2.solvePnP(obj, ordered, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            print("skip pnp", lab, flush=True)
            continue
        tl = tvec.reshape(3).astype(float)
        tcp = np.array([pose["x"], pose["y"], pose["z"]], float)
        j1 = float(s.get("j1_deg", np.degrees(np.arctan2(tcp[1], tcp[0]))))
        srcs.append(tl)
        dsts.append(rot_z(-j1) @ (p_tl - tcp))
        meta.append((lab, int(idx), j1))
        s["raw_tl_idx"] = int(idx)
        s["tl_cam_mm"] = [float(v) for v in tl]
        s["tl_origin_resolved"] = True

    r, t, rms, errs = kabsch(np.stack(srcs), np.stack(dsts))
    j1s = [j1 for _, _, j1 in meta]
    fit = {
        "n_looks": len(meta),
        "R": r.tolist(),
        "t_mm": [round(float(v), 4) for v in t],
        "rms_mm": round(rms, 3),
        "mean_abs_mm": round(float(np.mean(np.abs(errs))), 3),
        "max_mm": round(float(np.max(errs)), 3),
        "j1_span_deg": round(float(max(j1s) - min(j1s)), 3),
        "tl_idx_choice": {lab: idx for lab, idx, _ in meta},
        "per_look": {
            lab: {"err_mm": round(float(e), 3), "raw_tl_idx": idx, "j1_deg": round(j1, 3)}
            for (lab, idx, j1), e in zip(meta, errs)
        },
        "model": "T_base_cam = Trans(TCP) @ Rz(J1) @ T_end_to_camera",
        "note": "J1 end-frame fit; P_TL-nearest outer corner on all looks",
    }
    state["looks"] = state.get("looks") or {}
    state["fit"] = fit
    state["status"] = "fitted"
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(fit, indent=2, ensure_ascii=False), flush=True)

    # temp apply
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = HERE / f"camera_config.yaml.bak_pre_recal_{stamp}"
    shutil.copy2(CFG, bak)
    data = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    old_t = list(data["extrinsics"]["T_end_to_camera"]["t_mm"])
    data["extrinsics"]["T_end_to_camera"]["R"] = fit["R"]
    data["extrinsics"]["T_end_to_camera"]["t_mm"] = fit["t_mm"]
    data["extrinsics"]["j1_model"] = "Trans(TCP) @ Rz(J1) @ T_end_to_camera"
    data["extrinsics"]["j1_source"] = "pose.j1_deg or atan2(TCP_y, TCP_x)"
    data["extrinsics"]["transform_convention"] = (
        "T_end_to_camera maps camera coords into J1-rotating end-effector coords; "
        "T_base_cam = Trans(XYZ) @ Rz(J1) @ T_end_to_camera"
    )
    data["extrinsics"]["date"] = "candidate-j1-recal-" + datetime.now().strftime("%Y%m%d")
    data["extrinsics"]["note_candidate"] = (
        f"J1-model n={fit['n_looks']} rms={fit['rms_mm']}mm j1={fit['j1_span_deg']} TEMP"
    )
    CFG.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(
        json.dumps(
            {"event": "temp_apply", "backup": str(bak), "t_old": old_t, "t_new": fit["t_mm"]},
            ensure_ascii=False,
        ),
        flush=True,
    )

    # hover validate
    bridge, config = _bridge(), _config(12)
    look = {"x": 320.0, "y": 20.0, "z": 50.0, "r": 7.687}
    pose = move(bridge, config, look)
    time.sleep(0.4)
    cam = _open_camera(cam_cfg["calibration"])
    try:
        for _ in range(3):
            _capture(cam)
        time.sleep(0.12)
        frame = _capture(cam)
    finally:
        cam.Close()
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "validate_raw.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    corners = detect_chessboard(frame)
    if corners is None:
        raise RuntimeError("no board on validate")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tl_idx = physical_tl_open_cv_index(
        gray, corners, prefer_xy_mm=p_tl[:2], robot_pose=pose, config_path=CFG
    )
    ptw = PixelToWorld(str(CFG), robot_pose=pose)
    px = corners.reshape(-1, 2)[tl_idx]
    table = ptw.pixel_to_table(float(px[0]), float(px[1]), pose)
    tip_xy = (float(table[0]), float(table[1]))
    err = float(np.linalg.norm(np.array(tip_xy) - p_tl[:2]))
    solved = tcp_xy_from_tip_config(tip_xy)
    tcp_x, tcp_y = clamp_xy(float(solved["tcp_xy"][0]), float(solved["tcp_xy"][1]))
    target = {"x": round(tcp_x, 2), "y": round(tcp_y, 2), "z": -145.0, "r": 7.687}
    vis = frame.copy()
    cv2.drawChessboardCorners(vis, PATTERN, corners, True)
    for i in OUTER_IDX:
        col = (0, 0, 255) if i == tl_idx else (255, 128, 0)
        p = corners.reshape(-1, 2)[i]
        pt = (int(p[0]), int(p[1]))
        cv2.circle(vis, pt, 20, col, 3)
        cv2.putText(
            vis,
            "TL" if i == tl_idx else str(i),
            (pt[0] + 8, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            col,
            2,
        )
    cv2.putText(
        vis,
        f"n={fit['n_looks']} rms={fit['rms_mm']} err_vs_P_TL={err:.2f}mm",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    cv2.imwrite(str(OUT / "validate_marked.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(
        json.dumps(
            {
                "event": "validate_plan",
                "tl_idx": int(tl_idx),
                "tip_xy": [round(tip_xy[0], 3), round(tip_xy[1], 3)],
                "P_TL": [round(float(v), 3) for v in p_tl[:2]],
                "err_mm": round(err, 3),
                "target": target,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    pose1 = move(bridge, config, target)
    print(
        json.dumps(
            {
                "event": "hover",
                "pose": pose1,
                "note": "Check tip on Camera-title LEFT. Reply yes to lock / no to restore.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        raise SystemExit(1)
