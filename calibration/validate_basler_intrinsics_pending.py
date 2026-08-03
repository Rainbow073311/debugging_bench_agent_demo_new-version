"""Batch-check raw Basler calibration captures without deleting originals."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from capture_basler_intrinsics import ACTIVE_SESSION_FILE, _detect, _metrics


def _check(path: Path, pattern: tuple[int, int]) -> dict:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return {"file": path.name, "qualified": False, "reason": "unreadable image"}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners, detector = _detect(gray, pattern)
    if not found or corners is None:
        return {
            "file": path.name,
            "qualified": False,
            "reason": "54 inner corners not detected",
            "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
            "mean_brightness": round(float(gray.mean()), 1),
        }
    values = _metrics(frame, corners, pattern)
    reasons = []
    if not values["full_board_visible"]:
        reasons.append("outer board clipped or insufficient margin")
    if values["sharpness"] < 60.0:
        reasons.append("too blurry")
    if values["saturated_pct"] > 5.0:
        reasons.append("too many saturated pixels")
    return {
        "file": path.name,
        **values,
        "detector": detector,
        "qualified": not reasons,
        "reason": "; ".join(reasons) if reasons else "quality checks passed",
    }


def _angle_distance(first: float, second: float) -> float:
    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


def _feature_distance(first: dict, second: dict) -> float:
    center_a = np.asarray(first["center_norm"], dtype=float)
    center_b = np.asarray(second["center_norm"], dtype=float)
    center = float(np.linalg.norm(center_a - center_b)) / 0.055
    angle = _angle_distance(float(first["angle_deg"]), float(second["angle_deg"])) / 8.0
    bbox_a = first["board_bbox_norm"]
    bbox_b = second["board_bbox_norm"]
    scale_a = math.sqrt(float(bbox_a[2]) * float(bbox_a[3]))
    scale_b = math.sqrt(float(bbox_b[2]) * float(bbox_b[3]))
    scale = abs(math.log(max(scale_a, 1e-6) / max(scale_b, 1e-6))) / 0.07
    shape = abs(float(bbox_a[2]) / float(bbox_a[3]) - float(bbox_b[2]) / float(bbox_b[3])) / 0.08
    return math.sqrt(center * center + angle * angle + scale * scale + shape * shape)


def _select_diverse(records: list[dict], baselines: list[dict]) -> list[str]:
    selected_records = list(baselines)
    selected_names = []
    # Prefer sharp images, but force broad pose coverage through feature distance.
    for record in sorted(records, key=lambda item: float(item.get("sharpness", 0)), reverse=True):
        if not record.get("qualified"):
            continue
        nearest = min(
            (_feature_distance(record, previous) for previous in selected_records),
            default=float("inf"),
        )
        if nearest >= 1.0:
            selected_records.append(record)
            selected_names.append(record["file"])
    return selected_names


def _contact_sheet(pending_dir: Path, records: list[dict], destination: Path) -> None:
    record_by_name = {item["file"]: item for item in records}
    tiles = []
    for path in sorted(pending_dir.glob("pending_*.jpg")):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        tile = cv2.resize(frame, (300, 250), interpolation=cv2.INTER_AREA)
        record = record_by_name[path.name]
        color = (40, 210, 40) if record.get("qualified") else (40, 40, 230)
        cv2.rectangle(tile, (0, 0), (300, 48), (0, 0, 0), -1)
        cv2.putText(tile, path.stem, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        label = "PASS" if record.get("qualified") else "FAIL"
        cv2.putText(tile, label, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)
        tiles.append(tile)
    if not tiles:
        return
    columns = 5
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * columns:
        tiles.append(blank.copy())
    sheet = np.vstack(
        [np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    session_dir = Path(ACTIVE_SESSION_FILE.read_text(encoding="utf-8-sig").strip())
    session_file = session_dir / "session.json"
    session = json.loads(session_file.read_text(encoding="utf-8-sig"))
    pattern = tuple(int(value) for value in session["pattern_inner_corners"])
    pending_dir = session_dir / "pending"
    pending = sorted(pending_dir.glob("pending_*.jpg"))
    if not pending:
        raise RuntimeError(f"No pending images found in {pending_dir}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda path: _check(path, pattern), pending))

    baselines = []
    for image in session.get("images", []):
        checked = _check(session_dir / image["file"], pattern)
        if checked.get("qualified"):
            baselines.append(checked)
    selected = _select_diverse(records, baselines)
    selected_set = set(selected)
    for record in records:
        record["selected_for_calibration"] = record["file"] in selected_set
        if record.get("qualified") and record["file"] not in selected_set:
            record["reason"] = "quality passed but pose is near-duplicate"

    report = {
        "session": str(session_dir),
        "pending_total": len(records),
        "quality_passed": sum(bool(item.get("qualified")) for item in records),
        "selected_diverse": len(selected),
        "existing_accepted": len(session.get("images", [])),
        "selected_files": selected,
        "records": records,
    }
    report_path = session_dir / "pending_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sheet_path = session_dir / "pending_contact_sheet.jpg"
    _contact_sheet(pending_dir, records, sheet_path)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    print(f"contact_sheet={sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
