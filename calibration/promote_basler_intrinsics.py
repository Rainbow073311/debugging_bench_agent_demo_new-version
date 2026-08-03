"""Promote a verified Basler intrinsic candidate while preserving old config."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from capture_basler_intrinsics import ACTIVE_SESSION_FILE


def main() -> int:
    calibration_dir = Path(__file__).resolve().parent
    session_dir = Path(ACTIVE_SESSION_FILE.read_text(encoding="utf-8-sig").strip())
    verification_path = session_dir / "intrinsics_holdout_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8-sig"))
    if not verification.get("passed"):
        raise RuntimeError("Holdout verification did not pass; refusing promotion")

    candidate_path = session_dir / "camera_config_candidate.yaml"
    config = yaml.safe_load(candidate_path.read_text(encoding="utf-8-sig"))
    config["calibration"]["status"] = "active_intrinsics_extrinsics_pending"
    config["calibration"]["promoted_at"] = datetime.now().astimezone().isoformat()
    config["quality"]["holdout_verification"] = {
        "images": verification["holdout_images"],
        "mean_rmse_px": verification["mean_rmse_px"],
        "median_rmse_px": verification["median_rmse_px"],
        "max_rmse_px": verification["max_rmse_px"],
        "passed": True,
    }

    accepted_path = session_dir / "camera_config_accepted.yaml"
    accepted_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    accepted_path.write_text(accepted_text, encoding="utf-8")

    destination = calibration_dir / "camera_config.yaml"
    archive_dir = calibration_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = archive_dir / f"camera_config_pre_basler_{timestamp}.yaml"
    if destination.exists():
        shutil.copy2(destination, backup)

    temporary = calibration_dir / ".camera_config.yaml.tmp"
    temporary.write_text(accepted_text, encoding="utf-8")
    temporary.replace(destination)

    session_file = session_dir / "session.json"
    session = json.loads(session_file.read_text(encoding="utf-8-sig"))
    session["status"] = "intrinsics_calibrated_extrinsics_pending"
    session["intrinsics_result"] = {
        "active_config": str(destination),
        "accepted_config": str(accepted_path),
        "previous_config_backup": str(backup),
        "holdout_verification": str(verification_path),
        "images_used": config["calibration"]["chessboard"]["images_used"],
        "rms_error_px": config["quality"]["rms_error_px"],
    }
    session_file.write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    result = {
        "active_config": str(destination),
        "accepted_config": str(accepted_path),
        "backup": str(backup),
        "status": config["calibration"]["status"],
        "extrinsics_status": config["extrinsics"]["status"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
