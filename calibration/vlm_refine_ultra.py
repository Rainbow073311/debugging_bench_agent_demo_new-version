"""VLM + OpenCV hybrid TP refinement on ultra-close image.

Algorithm:
  1. Project coarse base-point (from close-image VLM) → pixel in ultra-close image
  2. Crop an ROI around that predicted pixel (352×352, same as back_06)
  3. Send ROI crop to VLM → VLM returns normalized TP text center (0-1000)
  4. OpenCV detects silver pads within ROI
  5. Select nearest pad above text center → map back to full-image coords
  6. Produce debug artifacts: ROI crop, VLM text bbox, OpenCV pads, full annotated

Called by server.js finalizeSplitServiceRun (Phase 2):
  python vlm_refine_ultra.py <ultra_img> <coarse_bp> <cam_pose> <tp_id>
    coarse_bp  = "X,Y,Z"   (base point from close-image VLM projection)
    cam_pose   = "X,Y,Z,R" (robot pose when ultra-close image was captured)
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from coordinate_transforms import PixelToWorld

VLM_BASE_URL = os.environ.get(
    "VLM_BASE_URL",
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
VLM_API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3.7-plus")

ROI_HALF = 176   # half-size of ROI crop (352×352, matches back_06)


# ---------------------------------------------------------------------------
# Step 1 — project coarse base point → pixel in ultra-close image
# ---------------------------------------------------------------------------

def project_base_to_pixel(coarse_bp_str: str, cam_pose_str: str
                          ) -> tuple[int, int] | None:
    """Use calibrated extrinsics to project 3D base point to image pixel."""
    parts = [float(v) for v in coarse_bp_str.split(",")]
    if len(parts) != 3:
        return None
    bx, by, bz = parts

    pose_parts = [float(v) for v in cam_pose_str.split(",")]
    if len(pose_parts) < 3:
        return None
    robot_pose = {
        "x": pose_parts[0], "y": pose_parts[1], "z": pose_parts[2],
        "r": pose_parts[3] if len(pose_parts) > 3 else 0.0,
    }

    config_path = str(HERE / "camera_config.yaml")
    mapper = PixelToWorld(config_path, robot_pose=robot_pose)
    pixel = mapper.world_to_pixel(bx, by, bz)
    if pixel is None:
        return None
    return int(round(pixel[0])), int(round(pixel[1]))


# ---------------------------------------------------------------------------
# Step 2 — crop ROI around predicted pixel
# ---------------------------------------------------------------------------

def crop_roi(img: np.ndarray, cx: int, cy: int,
             half: int = ROI_HALF
             ) -> tuple[np.ndarray, int, int]:
    """Crop a region around (cx, cy), clamped to image bounds.
    Returns (roi_array, offset_x, offset_y).
    """
    h, w = img.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return img[y1:y2, x1:x2], x1, y1


def encode_image_bgr(bgr: np.ndarray) -> str:
    """Encode an in-memory BGR image as a base64 data URL."""
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


# ---------------------------------------------------------------------------
# Step 3 — VLM finds TP text center within the ROI crop (normalized 0-1000)
# ---------------------------------------------------------------------------

VLM_PROMPT = (
    '1. Is the text "{tp_id}" present in this image?\n'
    '2. If yes, describe the text of surrounding components near it.\n'
    '3. Return the approximate normalized coordinates of the center of the {tp_id} text, '
    'with coordinate range 0 to 1000.\n'
    '4. Do not guess based on circuit knowledge; only answer based on the image.\n\n'
    'Return ONLY a JSON object:\n'
    '{{"tp9_present": true/false, "surrounding_text": "...", '
    '"norm_center": [x_0_1000, y_0_1000]}}'
)


def ask_vlm_for_text_center_in_roi(roi_bgr: np.ndarray, tp_id: str
                                    ) -> tuple[int, int] | None:
    """Send ROI crop to VLM, get normalized text center, convert to ROI coords.
    Returns (text_cx_roi, text_cy_roi) or None.
    """
    import urllib.request
    import urllib.error

    image_data_url = encode_image_bgr(roi_bgr)
    prompt = VLM_PROMPT.replace("{tp_id}", tp_id)

    body = json.dumps({
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
        "max_tokens": 200,
        "temperature": 0.1,
    }).encode()

    req = urllib.request.Request(
        f"{VLM_BASE_URL}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLM_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"VLM request failed: {e}"}))
        return None

    content = result["choices"][0]["message"]["content"]
    rh, rw = roi_bgr.shape[:2]

    # Parse: {"tp9_present": ..., "surrounding_text": "...", "norm_center": [x, y]}
    m = re.search(r'\{[^{}]*"tp9_present"[^{}]*\}', content, re.DOTALL)
    if not m:
        print(json.dumps({"ok": False, "error": f"VLM unexpected response: {content[:200]}"}))
        return None

    vlm_result = json.loads(m.group())
    tp9_present = vlm_result.get("tp9_present", False)
    norm = vlm_result.get("norm_center", None)

    if not tp9_present or not norm or len(norm) != 2:
        # TP not found in ROI — fall back to ROI center
        print(f"# VLM: TP not found in ROI, using ROI center", file=sys.stderr)
        return rw // 2, rh // 2

    # Convert normalized 0-1000 to ROI pixel coords
    text_cx_roi = int(round(norm[0] * rw / 1000))
    text_cy_roi = int(round(norm[1] * rh / 1000))
    return text_cx_roi, text_cy_roi


# ---------------------------------------------------------------------------
# Step 4 — OpenCV detects silver pads within ROI
# ---------------------------------------------------------------------------

def detect_pads_in_roi(roi_gray: np.ndarray) -> list[dict]:
    """Detect silver pads within a grayscale ROI crop.
    Returns list of {"center": (cx, cy), "radius": r, "method": "circle"|"contour"}
    in ROI-local coordinates.
    """
    candidates = []

    # --- HoughCircles for circular pads ---
    blurred = cv2.GaussianBlur(roi_gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
        param1=50, param2=28, minRadius=6, maxRadius=60,
    )
    if circles is not None:
        for c in circles[0]:
            candidates.append({
                "center": (float(c[0]), float(c[1])),
                "radius": float(c[2]),
                "method": "circle",
            })

    # --- Contour-based for rectangular/oval pads ---
    _, bright = cv2.threshold(roi_gray, 170, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40 or area > 6000:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 8:
            continue
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
        if circularity < 0.25:
            continue
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        if not any(np.hypot(cx - e["center"][0], cy - e["center"][1]) < 12
                   for e in candidates):
            candidates.append({
                "center": (float(cx), float(cy)),
                "radius": float(np.sqrt(area / np.pi)),
                "method": "contour",
            })
    return candidates


def find_best_pad(candidates: list[dict], text_center_roi: tuple[int, int]
                  ) -> tuple[int, int] | None:
    """Pick the nearest silver pad ABOVE the text center (smaller y in image coords).
    Falls back to nearest overall if no pad is above.
    """
    if not candidates:
        return None

    tx_cx, tx_cy = text_center_roi

    # Filter: only pads ABOVE text center (smaller y in image coords)
    above = [(np.hypot(c["center"][0] - tx_cx, c["center"][1] - tx_cy), c)
             for c in candidates if c["center"][1] < tx_cy]

    if above:
        above.sort(key=lambda x: x[0])
        best = above[0][1]
    else:
        # Fallback: nearest overall, excluding text bbox area
        valid = [c for c in candidates if not (
            tx_cx - 35 <= c["center"][0] <= tx_cx + 35 and
            tx_cy - 15 <= c["center"][1] <= tx_cy + 15
        )]
        if not valid:
            valid = candidates
        valid.sort(key=lambda c: np.hypot(c["center"][0] - tx_cx, c["center"][1] - tx_cy))
        best = valid[0]

    return (int(round(best["center"][0])), int(round(best["center"][1])))


# ---------------------------------------------------------------------------
# Debug — save annotated images
# ---------------------------------------------------------------------------

def save_debug_artifacts(
    img: np.ndarray, roi_bgr: np.ndarray,
    text_center_roi: tuple[int, int],
    pad_center_roi: tuple[int, int] | None,
    candidates_roi: list[dict],
    offset_x: int, offset_y: int,
    predicted_px: int, predicted_py: int,
    tp_id: str, out_dir: str,
) -> dict:
    """Save 4 debug artifacts. Returns dict of relative filenames."""
    rh, rw = roi_bgr.shape[:2]
    tx, ty = text_center_roi

    # Derive a text bbox for display (estimate ~60×20 px)
    tbox = [
        max(0, tx - 30), max(0, ty - 10),
        min(rw - 1, tx + 30), min(rh - 1, ty + 10),
    ]

    # 1) ROI crop
    roi_crop_name = f"ultra_roi_{tp_id}.jpg"
    cv2.imwrite(str(Path(out_dir) / roi_crop_name),
                roi_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 2) ROI + VLM text bbox
    roi_vlm = roi_bgr.copy()
    cv2.rectangle(roi_vlm, (tbox[0], tbox[1]), (tbox[2], tbox[3]), (0, 255, 0), 2)
    cv2.putText(roi_vlm, f'{tp_id} text', (tbox[0], max(tbox[1] - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    roi_vlm_name = f"ultra_roi_vlm_text_{tp_id}.jpg"
    cv2.imwrite(str(Path(out_dir) / roi_vlm_name),
                roi_vlm, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 3) ROI + pad candidates + selected
    roi_pads = roi_bgr.copy()
    for c in candidates_roi:
        px_, py_ = int(round(c["center"][0])), int(round(c["center"][1]))
        r_ = int(round(c["radius"]))
        cv2.circle(roi_pads, (px_, py_), r_, (255, 180, 80), 2)
        cv2.circle(roi_pads, (px_, py_), 3, (255, 180, 80), -1)
    cv2.rectangle(roi_pads, (tbox[0], tbox[1]), (tbox[2], tbox[3]), (0, 255, 0), 2)
    if pad_center_roi:
        bpx, bpy = pad_center_roi
        cv2.circle(roi_pads, (bpx, bpy), 12, (0, 0, 255), 3)
        cv2.circle(roi_pads, (bpx, bpy), 4, (0, 0, 255), -1)
        cv2.putText(roi_pads, f'{tp_id} pad', (bpx + 15, bpy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    roi_pads_name = f"ultra_roi_opencv_pads_{tp_id}.jpg"
    cv2.imwrite(str(Path(out_dir) / roi_pads_name),
                roi_pads, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 4) Full ultra-close + all annotations
    full = img.copy()
    # ROI rectangle (yellow)
    cv2.rectangle(full, (offset_x, offset_y),
                  (offset_x + rw, offset_y + rh), (0, 220, 220), 2)
    cv2.putText(full, "ROI", (offset_x + 4, offset_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 2)
    # Predicted pixel (magenta cross)
    cv2.drawMarker(full, (predicted_px, predicted_py), (255, 0, 255),
                   cv2.MARKER_CROSS, 20, 2)
    cv2.putText(full, "coarse pred", (predicted_px + 10, predicted_py - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    # Text bbox (green)
    cv2.rectangle(full,
                  (tbox[0] + offset_x, tbox[1] + offset_y),
                  (tbox[2] + offset_x, tbox[3] + offset_y),
                  (0, 255, 0), 2)
    cv2.putText(full, f'{tp_id} text',
                (tbox[0] + offset_x, max(tbox[1] + offset_y - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Pad candidates (blue)
    for c in candidates_roi:
        cx = int(round(c["center"][0])) + offset_x
        cy = int(round(c["center"][1])) + offset_y
        r = int(round(c["radius"]))
        cv2.circle(full, (cx, cy), r, (255, 180, 80), 2)
    # Selected pad (red)
    if pad_center_roi:
        bpx = pad_center_roi[0] + offset_x
        bpy = pad_center_roi[1] + offset_y
        cv2.circle(full, (bpx, bpy), 14, (0, 0, 255), 3)
        cv2.circle(full, (bpx, bpy), 5, (0, 0, 255), -1)
        cv2.putText(full, f'{tp_id} pad', (bpx + 15, bpy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    full_name = f"ultra_full_refined_{tp_id}.jpg"
    cv2.imwrite(str(Path(out_dir) / full_name),
                full, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {
        "roi_crop": roi_crop_name,
        "roi_vlm_text": roi_vlm_name,
        "roi_opencv_pads": roi_pads_name,
        "full_refined": full_name,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print(json.dumps({
            "ok": False,
            "error": "Usage: vlm_refine_ultra.py <img> <coarse_bp> <cam_pose> [tp_id]",
        }))
        sys.exit(1)

    img_path = sys.argv[1]
    coarse_bp = sys.argv[2]   # "X,Y,Z" from close-image VLM projection
    cam_pose = sys.argv[3]    # "X,Y,Z,R" robot pose at ultra-close capture
    tp_id = sys.argv[4] if len(sys.argv) > 4 else "TP9"

    if not Path(img_path).exists():
        print(json.dumps({"ok": False, "error": f"Image not found: {img_path}"}))
        sys.exit(1)

    if not VLM_API_KEY:
        print(json.dumps({"ok": False, "error": "VLM_API_KEY not set"}))
        sys.exit(1)

    # Load full image
    img = cv2.imread(img_path)
    if img is None:
        print(json.dumps({"ok": False, "error": f"Cannot read image: {img_path}"}))
        sys.exit(1)

    # Step 1 — project coarse base point → predicted pixel in ultra-close image
    predicted = project_base_to_pixel(coarse_bp, cam_pose)
    if predicted is None:
        print(json.dumps({"ok": False, "error": "Cannot project base point to image"}))
        sys.exit(1)
    pred_px, pred_py = predicted

    # Step 2 — crop ROI around predicted pixel
    roi_bgr, off_x, off_y = crop_roi(img, pred_px, pred_py)
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Step 3 — VLM finds text center in ROI (normalized coords → pixel)
    vlm_result = ask_vlm_for_text_center_in_roi(roi_bgr, tp_id)
    if vlm_result is None:
        sys.exit(1)
    text_cx_roi, text_cy_roi = vlm_result

    # Step 4 — OpenCV detects pads, select nearest above text
    candidates_roi = detect_pads_in_roi(roi_gray)
    pad_center_roi = find_best_pad(candidates_roi, (text_cx_roi, text_cy_roi))

    # Map to full-image coordinates
    text_pixel_full = [text_cx_roi + off_x, text_cy_roi + off_y]
    pad_pixel_full = None
    if pad_center_roi:
        pad_pixel_full = [pad_center_roi[0] + off_x, pad_center_roi[1] + off_y]

    # Save debug artifacts
    out_dir = str(Path(img_path).parent)
    artifacts = save_debug_artifacts(
        img, roi_bgr, (text_cx_roi, text_cy_roi),
        pad_center_roi, candidates_roi,
        off_x, off_y, pred_px, pred_py,
        tp_id, out_dir,
    )

    result = {
        "ok": True,
        "tp_id": tp_id,
        "predicted_pixel": [pred_px, pred_py],
        "roi_bbox": [off_x, off_y, off_x + roi_bgr.shape[1], off_y + roi_bgr.shape[0]],
        "text_center_roi": [text_cx_roi, text_cy_roi],
        "text_pixel": text_pixel_full,
        "pad_pixel": pad_pixel_full,
        "pad_candidates": len(candidates_roi),
        "artifacts": artifacts,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
