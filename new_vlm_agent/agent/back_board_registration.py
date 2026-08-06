"""Auditable geometry-first registration for back-side PCBA localisation.

Ordinary solder joints and vias are deliberately treated as *rejected*
candidates.  Board-outline hypotheses establish scale/perspective and explicit
mirror/rotation possibilities; mechanical holes/tooling features only choose
and validate the correct hypothesis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _cv():
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    return cv2, np


def _load(path: str | Path):
    cv2, _ = _cv()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def _write(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2, _ = _cv()
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"cannot write debug image: {path}")


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    tolerance = max(2, int(round(min(ow, oh) * 0.005)))
    return bool(
        ix >= ox - tolerance
        and iy >= oy - tolerance
        and ix + iw <= ox + ow + tolerance
        and iy + ih <= oy + oh + tolerance
    )


def _auxiliary_frame_inner_candidate(
    outer: dict[str, Any],
    candidates: list[dict[str, Any]],
    target_point: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    """Prefer a nested PCB body over an outer panel/auxiliary frame.

    Assembly drawings can contain narrow tooling rails above and below the
    physical PCB.  Both the rail envelope and PCB body are valid rectangles,
    so selecting the largest contour silently includes the rails.  A nested
    candidate is considered the PCB body only when it occupies most of the
    outer area, has a balanced inset on at least one axis (the characteristic
    pair of auxiliary rails), and still contains the marked TP.
    """
    ox, oy, ow, oh = outer["bbox"]
    outer_area = max(float(outer["area_ratio"]), 1e-9)
    matches: list[dict[str, Any]] = []
    for item in candidates:
        if item is outer or not _bbox_contains(outer["bbox"], item["bbox"]):
            continue
        ix, iy, iw, ih = item["bbox"]
        area_fraction = float(item["area_ratio"]) / outer_area
        if not 0.58 <= area_fraction <= 0.93:
            continue
        aspect_ratio = max(iw, ih) / max(1.0, min(iw, ih))
        if aspect_ratio > 4.0:
            continue
        if target_point is not None:
            tx, ty = target_point
            target_margin = max(2, int(round(min(iw, ih) * 0.01)))
            if not (
                ix - target_margin <= tx <= ix + iw + target_margin
                and iy - target_margin <= ty <= iy + ih + target_margin
            ):
                continue

        left, right = ix - ox, (ox + ow) - (ix + iw)
        top, bottom = iy - oy, (oy + oh) - (iy + ih)

        def balanced_pair(first: int, second: int, span: int) -> bool:
            minimum = min(first, second)
            maximum = max(first, second)
            return bool(
                minimum >= span * 0.025
                and maximum <= span * 0.30
                and maximum / max(1.0, minimum) <= 2.5
            )

        if not (balanced_pair(left, right, ow) or balanced_pair(top, bottom, oh)):
            continue
        item["auxiliary_frame_area_fraction"] = area_fraction
        item["auxiliary_frame_insets"] = {
            "left": int(left), "right": int(right),
            "top": int(top), "bottom": int(bottom),
        }
        matches.append(item)
    if not matches:
        return None
    return max(matches, key=lambda item: (item["rectangularity"], item["area_ratio"]))


def _contour_aspect_ratio(contour) -> float:
    cv2, _ = _cv()
    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    short = max(1.0, min(float(width), float(height)))
    return max(float(width), float(height)) / short


def _refine_photo_rect_to_expected_aspect(
    contour,
    support_mask,
    expected_aspect_ratio: float | None,
):
    """Trim connector/screw protrusions using locator aspect and edge support."""
    cv2, np = _cv()
    evidence: dict[str, Any] = {
        "attempted": False,
        "applied": False,
        "expected_aspect_ratio": expected_aspect_ratio,
        "reason": "expected_aspect_ratio_not_provided",
    }
    if expected_aspect_ratio is None:
        return contour, evidence
    try:
        expected = float(expected_aspect_ratio)
    except (TypeError, ValueError):
        return contour, evidence
    if not math.isfinite(expected) or not 1.02 <= expected <= 5.0:
        evidence["reason"] = "expected_aspect_ratio_invalid"
        return contour, evidence

    points = _ordered_box(contour).astype(np.float32)
    top_left, top_right, bottom_right, bottom_left = points
    width = float(np.linalg.norm(top_right - top_left))
    height = float(np.linalg.norm(bottom_left - top_left))
    short = max(1.0, min(width, height))
    current = max(width, height) / short
    evidence.update({
        "attempted": True,
        "current_aspect_ratio": round(current, 6),
        "expected_aspect_ratio": round(expected, 6),
    })
    if abs(current - expected) / expected <= 0.025:
        evidence["reason"] = "aspect_ratio_already_consistent"
        return contour, evidence

    canvas_width = max(8, int(round(width)))
    canvas_height = max(8, int(round(height)))
    destination = np.float32([
        [0, 0],
        [canvas_width - 1, 0],
        [canvas_width - 1, canvas_height - 1],
        [0, canvas_height - 1],
    ])
    matrix = cv2.getPerspectiveTransform(points, destination)
    warped = cv2.warpPerspective(
        support_mask,
        matrix,
        (canvas_width, canvas_height),
        flags=cv2.INTER_NEAREST,
    )
    band = max(5, int(round(min(canvas_width, canvas_height) * 0.04)))
    foreground = warped > 0
    supports = {
        "left": float(foreground[:, :band].mean()),
        "right": float(foreground[:, -band:].mean()),
        "top": float(foreground[:band, :].mean()),
        "bottom": float(foreground[-band:, :].mean()),
    }
    evidence["edge_support"] = {key: round(value, 6) for key, value in supports.items()}

    trim_axis: str
    trim_total: float
    if current > expected:
        trim_axis = "width" if width >= height else "height"
        long_side = max(width, height)
        short_side = min(width, height)
        trim_total = long_side - short_side * expected
    else:
        trim_axis = "width" if width <= height else "height"
        long_side = max(width, height)
        short_side = min(width, height)
        trim_total = short_side - long_side / expected
    axis_size = width if trim_axis == "width" else height
    trim_fraction = trim_total / max(1.0, axis_size)
    evidence.update({
        "trim_axis": trim_axis,
        "trim_total_px": round(trim_total, 3),
        "trim_fraction": round(trim_fraction, 6),
    })
    if trim_total <= 1.0:
        evidence["reason"] = "aspect_adjustment_would_not_shrink"
        return contour, evidence
    if trim_fraction > 0.25:
        evidence["reason"] = "required_trim_exceeds_safety_limit"
        return contour, evidence

    first_name, second_name = (
        ("left", "right") if trim_axis == "width" else ("top", "bottom")
    )
    first_support = supports[first_name]
    second_support = supports[second_name]
    if first_support >= second_support * 1.15 and first_support - second_support >= 0.05:
        first_trim, second_trim = 0.0, trim_total
        trim_side = second_name
    elif second_support >= first_support * 1.15 and second_support - first_support >= 0.05:
        first_trim, second_trim = trim_total, 0.0
        trim_side = first_name
    else:
        first_trim = second_trim = trim_total / 2.0
        trim_side = "symmetric"

    refined = points.copy()
    if trim_axis == "width":
        unit = (top_right - top_left) / max(width, 1.0)
        refined[0] += unit * first_trim
        refined[3] += unit * first_trim
        refined[1] -= unit * second_trim
        refined[2] -= unit * second_trim
    else:
        unit = (bottom_left - top_left) / max(height, 1.0)
        refined[0] += unit * first_trim
        refined[1] += unit * first_trim
        refined[2] -= unit * second_trim
        refined[3] -= unit * second_trim

    # Keep the same integer contour representation returned by findContours;
    # downstream landmark masks use drawContours/fillPoly, which require CV_32S.
    refined_contour = np.rint(refined).reshape((-1, 1, 2)).astype(np.int32)
    evidence.update({
        "applied": True,
        "reason": "trimmed_low_support_protruding_edge",
        "trim_side": trim_side,
        "refined_aspect_ratio": round(_contour_aspect_ratio(refined_contour), 6),
        "original_corners": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        "refined_corners": [[round(float(x), 3), round(float(y), 3)] for x, y in refined],
    })
    return refined_contour, evidence


def _board_contour(
    image,
    *,
    domain: str,
    roi_hint_norm: list[float] | None = None,
    expected_aspect_ratio: float | None = None,
):
    cv2, np = _cv()
    h, w = image.shape[:2]
    roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, w, h
    if isinstance(roi_hint_norm, list) and len(roi_hint_norm) == 4:
        try:
            nx1, ny1, nx2, ny2 = [float(v) for v in roi_hint_norm]
            if not all(math.isfinite(v) for v in (nx1, ny1, nx2, ny2)):
                raise ValueError
            nx1, nx2 = sorted((max(0.0, min(1.0, nx1)), max(0.0, min(1.0, nx2))))
            ny1, ny2 = sorted((max(0.0, min(1.0, ny1)), max(0.0, min(1.0, ny2))))
            if nx2 - nx1 < 0.08 or ny2 - ny1 < 0.08:
                raise ValueError
            # Expand the semantic proposal slightly so OpenCV can recover the
            # physical edges instead of treating the VLM box as exact geometry.
            pad_x = (nx2 - nx1) * 0.08
            pad_y = (ny2 - ny1) * 0.08
            roi_x1 = int(round(max(0.0, nx1 - pad_x) * w))
            roi_y1 = int(round(max(0.0, ny1 - pad_y) * h))
            roi_x2 = int(round(min(1.0, nx2 + pad_x) * w))
            roi_y2 = int(round(min(1.0, ny2 + pad_y) * h))
        except (TypeError, ValueError):
            raise ValueError("roi_hint_norm must be four normalized numbers [x1,y1,x2,y2]")
    work = image[roi_y1:roi_y2, roi_x1:roi_x2]
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    rh, rw = work.shape[:2]
    if domain == "photo":
        # The production fixture uses a red PCB.  Detect the solder-mask body
        # before the generic chromatic fallback so black cables, metal pin
        # headers, gold contacts and other protrusions cannot enlarge the PCB
        # rectangle.
        #
        # Strong morphological OPEN breaks thin connector / pin-header bridges
        # so they don't pull the fitted rectangle outward.  A smaller, single-
        # iteration CLOSE later only fills narrow internal gaps without
        # reconnecting already-severed protrusions.
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        red_substrate = (
            ((hue <= 14) | (hue >= 170))
            & (saturation >= 55)
            & (value >= 25)
        ).astype(np.uint8) * 255
        # ── opening to sever thin fragments ─────────────────────────────
        opening_scale = max(7, int(round(min(rh, rw) * 0.006)))
        if opening_scale % 2 == 0:
            opening_scale += 1
        red_substrate = cv2.morphologyEx(
            red_substrate,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (opening_scale, opening_scale)
            ),
        )

        # Estimate the surrounding surface from the image border.  When that
        # surface is chromatic (for example case-204's green ESD mat), retain
        # only saturated pixels whose hue differs from it.  On a neutral
        # background (case-203), ordinary chromatic segmentation is sufficient.
        border_width = max(3, int(round(min(rh, rw) * 0.04)))
        border = np.concatenate([
            hsv[:border_width].reshape(-1, 3),
            hsv[-border_width:].reshape(-1, 3),
            hsv[:, :border_width].reshape(-1, 3),
            hsv[:, -border_width:].reshape(-1, 3),
        ])
        saturated_border = border[border[:, 1] >= 40]
        hue_samples = saturated_border[:, 0] if saturated_border.size else border[:, 0]
        background_hue = int(np.bincount(hue_samples, minlength=180).argmax())
        background_saturation = float(np.median(border[:, 1]))
        red_area_ratio = float(np.count_nonzero(red_substrate) / max(1, rh * rw))
        if red_area_ratio >= 0.06:
            mask = red_substrate
            photo_method = "red_pcb_substrate"
        elif background_saturation >= 40:
            hue_distance = np.abs(hsv[:, :, 0].astype(np.int16) - background_hue)
            hue_distance = np.minimum(hue_distance, 180 - hue_distance)
            chromatic = (saturation >= 55) & (value >= 25) & (hue_distance >= 20)
            photo_method = "border_hue_contrast"
            mask = chromatic.astype(np.uint8) * 255
        else:
            chromatic = (saturation >= 75) & (value >= 25)
            photo_method = "chromatic_solder_mask"
            blue, green, red = cv2.split(work)
            b16 = blue.astype(np.int16)
            g16 = green.astype(np.int16)
            r16 = red.astype(np.int16)
            green_dominant = (
                (g16 >= 45)
                & (g16 >= r16 + 10)
                & (g16 >= b16 + 10)
            )
            # The green fallback is safe only when the surrounding surface is
            # neutral; otherwise it would reconnect a green mat to the image.
            chromatic |= green_dominant
            mask = chromatic.astype(np.uint8) * 255
        # ── closing to fill internal gaps ────────────────────────────────
        scale = max(9, int(round(min(rh, rw) * 0.012)))
        retrieval_mode = cv2.RETR_EXTERNAL
    else:
        # Locator PDFs are faint gray line art, occasionally with a green TP
        # marker. Use a small closing kernel and RETR_LIST so an internal PCB
        # rectangle remains a candidate instead of being swallowed by the
        # PDF page border/title block.
        saturated = cv2.inRange(hsv, np.array([0, 22, 15]), np.array([180, 255, 247]))
        dark = cv2.threshold(gray, 247, 255, cv2.THRESH_BINARY_INV)[1]
        mask = cv2.bitwise_or(saturated, dark)
        scale = max(3, int(round(min(rh, rw) * 0.002)))
        retrieval_mode = cv2.RETR_LIST
    if scale % 2 == 0:
        scale += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, scale))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2 if domain == "locator" else 1,
    )
    contours, _ = cv2.findContours(mask, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)
    min_area = h * w * 0.05
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        raise ValueError("PCB outline not found")

    candidates: list[dict[str, Any]] = []
    margin = max(3, int(round(min(h, w) * 0.004)))
    min_rectangularity = 0.70 if domain == "photo" else 0.55
    for raw_contour in contours:
        # Photo domain uses the raw contour directly — convex hull would
        # swallow cables, pin headers and other protrusions outside the
        # physical PCB rectangle.
        contour = raw_contour if domain == "photo" else cv2.convexHull(raw_contour)
        contour = contour + np.array([[[roi_x1, roi_y1]]], dtype=contour.dtype)
        rect = cv2.minAreaRect(contour)
        rect_area = max(1.0, float(rect[1][0] * rect[1][1]))
        rectangularity = float(cv2.contourArea(contour) / rect_area)
        x, y, bw, bh = cv2.boundingRect(contour)
        area_ratio = float(cv2.contourArea(contour) / max(1, h * w))
        bbox_area_ratio = float(bw * bh / max(1, h * w))
        border_touch_count = sum((
            x <= margin,
            y <= margin,
            x + bw >= w - margin,
            y + bh >= h - margin,
        ))
        valid = bool(
            0.08 <= area_ratio <= 0.94
            and border_touch_count == 0
            and rectangularity >= min_rectangularity
            and (domain != "locator" or bbox_area_ratio <= 0.80)
        )
        candidates.append({
            "contour": contour,
            "rect": rect,
            "rectangularity": rectangularity,
            "bbox": (x, y, bw, bh),
            "aspect_ratio": max(bw, bh) / max(1.0, min(bw, bh)),
            "area_ratio": area_ratio,
            "bbox_area_ratio": bbox_area_ratio,
            "border_touch_count": border_touch_count,
            "valid": valid,
        })
    valid_candidates = [item for item in candidates if item["valid"]]
    selection_pool = valid_candidates or candidates
    selected = max(selection_pool, key=lambda item: item["area_ratio"])
    selection_method = "largest_valid_area"
    auxiliary_frame_outer = None
    if domain == "locator":
        plausible = [item for item in selection_pool if item["aspect_ratio"] <= 4.0]
        if plausible:
            selected = max(plausible, key=lambda item: item["area_ratio"])
        try:
            target_point = _green_tp(image)
        except ValueError:
            target_point = None
        nested = _auxiliary_frame_inner_candidate(selected, plausible, target_point)
        if nested is not None:
            auxiliary_frame_outer = selected
            selected = nested
            selection_method = "nested_inner_rectangle_over_auxiliary_frame"
    contour = selected["contour"]
    photo_body_refinement = None
    if domain == "photo":
        full_support_mask = np.zeros((h, w), dtype=np.uint8)
        full_support_mask[roi_y1:roi_y2, roi_x1:roi_x2] = mask
        contour, photo_body_refinement = _refine_photo_rect_to_expected_aspect(
            contour,
            full_support_mask,
            expected_aspect_ratio,
        )
    rect = selected["rect"]
    if domain == "photo" and photo_body_refinement and photo_body_refinement["applied"]:
        rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.int32)
    rectangularity = float(selected["rectangularity"])
    # Registration is defined by the physical PCB rectangle, not by shadow or
    # component protrusions in the raw foreground contour.
    filled = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(filled, box, 255)
    x, y, bw, bh = cv2.boundingRect(contour)
    area_ratio = float(selected["area_ratio"])
    valid = bool(selected["valid"])
    metrics = {
        "domain": domain,
        "image_size": [int(w), int(h)],
        "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
        "area_ratio": round(area_ratio, 6),
        "bbox_area_ratio": round(float(selected["bbox_area_ratio"]), 6),
        "rectangularity": round(rectangularity, 6),
        "minimum_rectangularity": min_rectangularity,
        "border_touch_count": int(selected["border_touch_count"]),
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid_candidates),
        "selection_method": selection_method,
        "candidate_summaries": [
            {
                "bbox_xywh": [int(value) for value in item["bbox"]],
                "area_ratio": round(float(item["area_ratio"]), 6),
                "rectangularity": round(float(item["rectangularity"]), 6),
                "aspect_ratio": round(float(item["aspect_ratio"]), 6),
                "valid": bool(item["valid"]),
                "selected": item is selected,
            }
            for item in sorted(candidates, key=lambda candidate: candidate["area_ratio"], reverse=True)
        ],
        "auxiliary_frame_outer_bbox_xywh": (
            [int(value) for value in auxiliary_frame_outer["bbox"]]
            if auxiliary_frame_outer is not None else None
        ),
        "photo_body_refinement": photo_body_refinement,
        "target_marker_inside_selection": (
            bool(
                x <= target_point[0] <= x + bw
                and y <= target_point[1] <= y + bh
            )
            if domain == "locator" and target_point is not None else None
        ),
        "semantic_roi_hint_norm": roi_hint_norm,
        "valid": valid,
        "method": photo_method if domain == "photo" else "internal_faint_line_rectangle",
    }
    return contour, filled, metrics


def _ordered_box(contour):
    cv2, np = _cv()
    points = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    return np.roll(points, -start, axis=0)


def _patch_stats(gray, x: float, y: float, radius: float) -> tuple[float, float, float]:
    cv2, np = _cv()
    h, w = gray.shape[:2]
    r = max(3, int(round(radius)))
    cx, cy = int(round(x)), int(round(y))
    x1, y1, x2, y2 = max(0, cx - 2 * r), max(0, cy - 2 * r), min(w, cx + 2 * r + 1), min(h, cy + 2 * r + 1)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 128.0, 128.0, 0.0
    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    inner = patch[dist <= max(2.0, radius * 0.55)]
    ring = patch[(dist >= radius * 0.82) & (dist <= radius * 1.35)]
    inner_mean = float(np.mean(inner)) if inner.size else 128.0
    ring_mean = float(np.mean(ring)) if ring.size else 128.0
    return inner_mean, ring_mean, abs(inner_mean - ring_mean)


def _candidate_landmarks(image, board_contour, *, domain: str, max_kept: int = 12):
    cv2, np = _cv()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Locator line art is extremely faint after PDF rasterization. Candidate
    # generation should be permissive; semantic acceptance belongs to the VLM.
    edges = cv2.Canny(blur, 8, 55) if domain == "locator" else cv2.Canny(blur, 30, 125)
    board_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(board_mask, [board_contour], -1, 255, thickness=-1)
    edges = cv2.bitwise_and(edges, board_mask)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    _, _, bw, bh = cv2.boundingRect(board_contour)
    board_diag = max(1.0, math.hypot(bw, bh))
    board_corners = _ordered_box(board_contour).astype(np.float32)
    unit_corners = np.float32([[0, 0], [0, 1], [1, 1], [1, 0]])
    to_unit = cv2.getPerspectiveTransform(board_corners, unit_corners)
    raw: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 30 or perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.48:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius_norm = float(radius) / board_diag
        if not (0.002 <= radius_norm <= 0.10):
            continue
        if cv2.pointPolygonTest(board_contour, (float(x), float(y)), False) < 0:
            continue
        uv = cv2.perspectiveTransform(np.float32([[[x, y]]]), to_unit)[0, 0]
        u, v = float(uv[0]), float(uv[1])
        edge_band_norm = min(u, v, 1.0 - u, 1.0 - v)
        inner, ring, contrast = _patch_stats(gray, x, y, radius)
        edge_distance = abs(float(cv2.pointPolygonTest(board_contour, (float(x), float(y)), True))) / board_diag
        size_score = max(0.0, min(1.0, (radius_norm - 0.0045) / 0.018))
        round_score = max(0.0, min(1.0, (circularity - 0.52) / 0.40))
        contrast_score = max(0.0, min(1.0, contrast / 90.0))
        if domain == "photo":
            # A real opening usually exposes a dark, low-detail interior. Bright
            # filled centres are typical solder/copper pads and are penalised.
            opening_score = max(0.0, min(1.0, (145.0 - inner) / 115.0))
        else:
            # Drawings commonly show a white opening bounded by a dark ring.
            opening_score = max(0.0, min(1.0, (inner - ring) / 100.0))
        edge_prior = max(0.0, min(1.0, (0.18 - edge_band_norm) / 0.12))
        score = 0.24 * size_score + 0.16 * round_score + 0.16 * contrast_score + 0.24 * opening_score + 0.20 * edge_prior
        reasons: list[str] = []
        if radius_norm < 0.006:
            reasons.append("too_small_likely_pad_or_via")
        if circularity < 0.62:
            reasons.append("weak_circularity")
        if domain == "photo" and inner > 150:
            reasons.append("bright_filled_center_likely_solder_pad")
        if contrast < 16:
            reasons.append("no_clear_hole_ring")
        if edge_band_norm > 0.18:
            reasons.append("outside_pcb_edge_band")
        accepted = (
            score >= (0.38 if domain == "locator" else 0.42)
            and radius_norm >= 0.006
            and edge_band_norm <= 0.18
        )
        raw.append({
            "x": float(x), "y": float(y), "radius": float(radius),
            "radius_norm": radius_norm, "circularity": circularity,
            "board_uv": [u, v], "edge_band_norm": edge_band_norm,
            "inner_gray": inner, "ring_gray": ring, "ring_contrast": contrast,
            "edge_distance_norm": edge_distance, "mechanical_score": score,
            "accepted": bool(accepted),
            "classification": "pcb_edge_opening_candidate_for_vlm" if accepted else "rejected_pad_via_or_unknown",
            "rejection_reasons": reasons if reasons else ([] if accepted else ["low_mechanical_score"]),
            "proposal_source": "contour",
        })

    if domain == "locator":
        # Assembly drawings often render mounting-hole rings in very light gray.
        # Canny/contours can miss them completely, so add a second proposal path
        # restricted to the PCB corners. This is intentionally only a proposal:
        # the VLM must still reject component circles and rounded board corners.
        scale = min(1.0, 2000.0 / max(gray.shape[:2]))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(small)
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(18, int(round(25 * scale))),
            param1=60,
            param2=24,
            minRadius=max(4, int(round(15 * scale))),
            maxRadius=max(10, int(round(85 * scale))),
        )
        if circles is not None:
            for sx, sy, sr in circles[0]:
                x, y, radius = float(sx / scale), float(sy / scale), float(sr / scale)
                radius_norm = radius / board_diag
                if not (0.006 <= radius_norm <= 0.025):
                    continue
                if cv2.pointPolygonTest(board_contour, (x, y), False) < 0:
                    continue
                uv = cv2.perspectiveTransform(np.float32([[[x, y]]]), to_unit)[0, 0]
                u, v = float(uv[0]), float(uv[1])
                edge_band_norm = min(u, v, 1.0 - u, 1.0 - v)
                corner_distance = min(
                    math.hypot(u, v), math.hypot(u, 1.0 - v),
                    math.hypot(1.0 - u, v), math.hypot(1.0 - u, 1.0 - v),
                )
                # Exclude the outline's own rounded corner (too close to both
                # edges), while retaining the normally inset tooling holes.
                if not (0.025 <= edge_band_norm <= 0.12 and corner_distance <= 0.22):
                    continue
                inner, ring, contrast = _patch_stats(gray, x, y, radius)
                size_fit = max(0.0, 1.0 - abs(radius_norm - 0.012) / 0.014)
                corner_fit = max(0.0, 1.0 - abs(corner_distance - 0.11) / 0.16)
                score = 0.70 + 0.12 * size_fit + 0.10 * corner_fit + 0.08 * min(1.0, contrast / 70.0)
                raw.append({
                    "x": x, "y": y, "radius": radius,
                    "radius_norm": radius_norm, "circularity": 1.0,
                    "board_uv": [u, v], "edge_band_norm": edge_band_norm,
                    "corner_distance_norm": corner_distance,
                    "inner_gray": inner, "ring_gray": ring, "ring_contrast": contrast,
                    "edge_distance_norm": abs(float(cv2.pointPolygonTest(board_contour, (x, y), True))) / board_diag,
                    "mechanical_score": score, "accepted": True,
                    "classification": "pcb_corner_opening_candidate_for_vlm",
                    "rejection_reasons": [], "proposal_source": "faint_ring_hough",
                })
    # Merge duplicate inner/outer contours around the same physical feature.
    raw.sort(key=lambda c: c["mechanical_score"], reverse=True)
    merged: list[dict[str, Any]] = []
    for item in raw:
        if any((item["x"] - old["x"]) ** 2 + (item["y"] - old["y"]) ** 2 < (0.55 * max(item["radius"], old["radius"])) ** 2 for old in merged):
            continue
        merged.append(item)
    accepted = [item for item in merged if item["accepted"]][:max_kept]
    rejected = [item for item in merged if not item["accepted"]][:40]
    accepted_prefix = "P" if domain == "photo" else "L"
    rejected_prefix = "PR" if domain == "photo" else "LR"
    for prefix, items in ((accepted_prefix, accepted), (rejected_prefix, rejected)):
        for index, item in enumerate(items, 1):
            item["id"] = f"{prefix}{index}"
    return accepted, rejected


def _outline_iou(locator_mask, board_mask, homography) -> float:
    cv2, np = _cv()
    h, w = board_mask.shape[:2]
    warped = cv2.warpPerspective(locator_mask, homography, (w, h), flags=cv2.INTER_NEAREST)
    intersection = int(np.count_nonzero((warped > 0) & (board_mask > 0)))
    union = int(np.count_nonzero((warped > 0) | (board_mask > 0)))
    return float(intersection / union) if union else 0.0


def _match_landmarks(locator, board, homography, board_diag: float):
    cv2, np = _cv()
    if not locator or not board:
        return [], board_diag
    src = np.float32([[[item["x"], item["y"]] for item in locator]])
    projected = cv2.perspectiveTransform(src, homography)[0]
    threshold = max(12.0, board_diag * 0.035)
    candidates: list[tuple[float, int, int, float]] = []
    for li, point in enumerate(projected):
        for bi, target in enumerate(board):
            distance = float(np.linalg.norm(point - np.float32([target["x"], target["y"]])))
            radius_penalty = abs(math.log(max(locator[li]["radius_norm"], 1e-5) / max(target["radius_norm"], 1e-5)))
            cost = distance + threshold * 0.35 * min(radius_penalty, 2.0)
            if distance <= threshold:
                candidates.append((cost, li, bi, distance))
    pairs: list[dict[str, Any]] = []
    used_l: set[int] = set()
    used_b: set[int] = set()
    for _, li, bi, distance in sorted(candidates):
        if li in used_l or bi in used_b:
            continue
        used_l.add(li)
        used_b.add(bi)
        pairs.append({
            "locator_id": locator[li]["id"], "board_id": board[bi]["id"],
            "locator_px": [locator[li]["x"], locator[li]["y"]],
            "board_px": [board[bi]["x"], board[bi]["y"]],
            "projected_px": [float(projected[li][0]), float(projected[li][1])],
            "error_px": distance,
        })
    return pairs, threshold


def _hypotheses(locator_corners, board_corners, locator_mask, board_mask, locator_landmarks, board_landmarks):
    cv2, np = _cv()
    _, _, bw, bh = cv2.boundingRect(np.int32(board_corners))
    diag = max(1.0, math.hypot(bw, bh))
    results: list[dict[str, Any]] = []
    for mirrored in (False, True):
        base = board_corners.copy()
        if mirrored:
            base = base[[0, 3, 2, 1]]
        for rotation in range(4):
            destination = np.roll(base, -rotation, axis=0).astype(np.float32)
            matrix = cv2.getPerspectiveTransform(locator_corners.astype(np.float32), destination)
            pairs, threshold = _match_landmarks(locator_landmarks, board_landmarks, matrix, diag)
            errors = [pair["error_px"] for pair in pairs]
            mean_error = float(sum(errors) / len(errors)) if errors else threshold * 2.0
            iou = _outline_iou(locator_mask, board_mask, matrix)
            score = 2.4 * len(pairs) + 4.0 * iou - min(mean_error / threshold, 2.0)
            results.append({
                "rotation_quadrants": rotation, "mirrored": mirrored,
                "outline_iou": iou, "pair_count": len(pairs),
                "mean_error_px": mean_error, "matching_threshold_px": threshold,
                "score": score, "pairs": pairs, "matrix": matrix,
            })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _reviewed_hypotheses(locator_corners, board_corners, locator_mask, board_mask, locator_candidates, board_candidates, review_matches):
    """Evaluate orientation hypotheses against exact correspondences chosen by the VLM."""
    cv2, np = _cv()
    locator_by_id = {item["id"]: item for item in locator_candidates}
    board_by_id = {item["id"]: item for item in board_candidates}
    resolved: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for match in review_matches:
        locator_item = locator_by_id.get(str(match.get("locator_id", "")))
        board_item = board_by_id.get(str(match.get("board_id", "")))
        if locator_item is None or board_item is None:
            raise ValueError(
                f"VLM review references unknown candidate pair: {match.get('locator_id')} -> {match.get('board_id')}"
            )
        landmark_type = str(match.get("landmark_type", "")).strip().lower()
        if landmark_type not in {"mounting_hole", "tooling_hole", "non_plated_hole", "fiducial", "board_cutout"}:
            raise ValueError(f"unsupported VLM landmark type: {landmark_type or 'missing'}")
        if float(match.get("confidence", 0.0)) < 0.75:
            raise ValueError(f"VLM landmark pair confidence below 0.75: {locator_item['id']} -> {board_item['id']}")
        resolved.append((locator_item, board_item, match))
    if len(resolved) < 2:
        raise ValueError("VLM review must provide at least two reliable landmark correspondences")
    _, _, bw, bh = cv2.boundingRect(np.int32(board_corners))
    diag = max(1.0, math.hypot(bw, bh))
    threshold = max(12.0, diag * 0.035)
    results: list[dict[str, Any]] = []
    for mirrored in (False, True):
        base = board_corners.copy()
        if mirrored:
            base = base[[0, 3, 2, 1]]
        for rotation in range(4):
            destination = np.roll(base, -rotation, axis=0).astype(np.float32)
            matrix = cv2.getPerspectiveTransform(locator_corners.astype(np.float32), destination)
            pairs: list[dict[str, Any]] = []
            inlier_count = 0
            weighted_error = 0.0
            total_weight = 0.0
            for locator_item, board_item, semantic in resolved:
                source = np.float32([[[locator_item["x"], locator_item["y"]]]])
                projected = cv2.perspectiveTransform(source, matrix)[0, 0]
                distance = float(np.linalg.norm(projected - np.float32([board_item["x"], board_item["y"]])))
                confidence = float(semantic.get("confidence", 0.0))
                if distance <= threshold:
                    inlier_count += 1
                weighted_error += min(distance, threshold * 4.0) * confidence
                total_weight += confidence
                pairs.append({
                    "locator_id": locator_item["id"], "board_id": board_item["id"],
                    "landmark_type": semantic.get("landmark_type"),
                    "vlm_confidence": confidence, "vlm_evidence": semantic.get("evidence", ""),
                    "locator_px": [locator_item["x"], locator_item["y"]],
                    "board_px": [board_item["x"], board_item["y"]],
                    "projected_px": [float(projected[0]), float(projected[1])],
                    "error_px": distance, "inlier": bool(distance <= threshold),
                })
            mean_error = weighted_error / max(total_weight, 1e-6)
            iou = _outline_iou(locator_mask, board_mask, matrix)
            score = 3.2 * inlier_count + 4.0 * iou - min(mean_error / threshold, 3.0)
            results.append({
                "rotation_quadrants": rotation, "mirrored": mirrored,
                "outline_iou": iou, "pair_count": inlier_count,
                "reviewed_pair_count": len(resolved),
                "mean_error_px": mean_error, "matching_threshold_px": threshold,
                "score": score, "pairs": pairs, "matrix": matrix,
            })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def _refine_homography_with_review(locator_corners, board_mask, locator_mask, reviewed_best):
    """Fit corners + VLM-reviewed edge holes, accepting only a measurable improvement."""
    cv2, np = _cv()
    base_matrix = reviewed_best["matrix"]
    base_destination = cv2.perspectiveTransform(
        locator_corners.reshape(1, -1, 2).astype(np.float32), base_matrix
    )[0]
    inlier_pairs = [
        pair for pair in reviewed_best.get("pairs", [])
        if pair.get("inlier") and float(pair.get("vlm_confidence", 0.0)) >= 0.75
    ]
    validation: dict[str, Any] = {
        "attempted": True,
        "reviewed_pair_count": len(reviewed_best.get("pairs", [])),
        "eligible_pair_count": len(inlier_pairs),
        "accepted": False,
        "fallback_reason": None,
    }
    if len(inlier_pairs) < 2:
        validation["fallback_reason"] = "fewer_than_two_high_confidence_edge_hole_pairs"
        return base_matrix, validation

    src = [list(map(float, point)) for point in locator_corners]
    dst = [list(map(float, point)) for point in base_destination]
    for pair in inlier_pairs:
        src.append(list(map(float, pair["locator_px"])))
        dst.append(list(map(float, pair["board_px"])))
    src_np = np.float32(src)
    dst_np = np.float32(dst)
    refined, ransac_mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 8.0)
    if refined is None:
        validation["fallback_reason"] = "homography_fit_failed"
        return base_matrix, validation

    locator_holes = np.float32([[pair["locator_px"] for pair in inlier_pairs]])
    board_holes = np.float32([pair["board_px"] for pair in inlier_pairs])
    base_holes = cv2.perspectiveTransform(locator_holes, base_matrix)[0]
    refined_holes = cv2.perspectiveTransform(locator_holes, refined)[0]
    base_errors = np.linalg.norm(base_holes - board_holes, axis=1)
    refined_errors = np.linalg.norm(refined_holes - board_holes, axis=1)
    refined_corners = cv2.perspectiveTransform(
        locator_corners.reshape(1, -1, 2).astype(np.float32), refined
    )[0]
    corner_errors = np.linalg.norm(refined_corners - base_destination, axis=1)
    h, w = board_mask.shape[:2]
    board_diag = math.hypot(w, h)
    base_mean = float(np.mean(base_errors))
    refined_mean = float(np.mean(refined_errors))
    corner_rmse = float(np.sqrt(np.mean(corner_errors ** 2)))
    base_iou = _outline_iou(locator_mask, board_mask, base_matrix)
    refined_iou = _outline_iou(locator_mask, board_mask, refined)
    improvement = base_mean - refined_mean
    accepted = bool(
        refined_mean <= max(10.0, board_diag * 0.012)
        and improvement >= max(1.5, base_mean * 0.08)
        and corner_rmse <= max(12.0, board_diag * 0.012)
        and refined_iou >= base_iou - 0.02
    )
    validation.update({
        "base_hole_errors_px": [round(float(v), 3) for v in base_errors],
        "refined_hole_errors_px": [round(float(v), 3) for v in refined_errors],
        "base_mean_hole_error_px": round(base_mean, 3),
        "refined_mean_hole_error_px": round(refined_mean, 3),
        "hole_error_improvement_px": round(improvement, 3),
        "corner_rmse_px": round(corner_rmse, 3),
        "base_outline_iou": round(base_iou, 4),
        "refined_outline_iou": round(refined_iou, 4),
        "ransac_inliers": int(np.count_nonzero(ransac_mask)) if ransac_mask is not None else 0,
        "accepted": accepted,
        "fallback_reason": None if accepted else "refinement_did_not_pass_strict_improvement_gate",
    })
    return (refined if accepted else base_matrix), validation


def _green_tp(image):
    cv2, np = _cv()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 90, 90]), np.array([90, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("green TP marker not found on locator image")
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        raise ValueError("green TP marker has invalid centroid")
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def _draw_outline(image, contour, corners, label: str):
    cv2, np = _cv()
    out = image.copy()
    # Thin orange = raw segmentation evidence, never used directly as the
    # mapping boundary. Thick green = the strict rectangle actually used.
    cv2.drawContours(out, [contour], -1, (0, 165, 255), 1)
    rectangle = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [rectangle], True, (0, 255, 0), 4, cv2.LINE_AA)
    for index, (x, y) in enumerate(corners):
        cv2.circle(out, (int(x), int(y)), 10, (0, 0, 255), -1)
        cv2.putText(out, f"C{index}", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(out, label, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 2)
    cv2.putText(out, "GREEN=mapping rectangle  ORANGE=raw segmentation", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 120, 0), 2)
    return out


def _draw_candidates(image, accepted, rejected, label: str):
    cv2, _ = _cv()
    out = image.copy()
    for item in rejected:
        x, y, r = int(item["x"]), int(item["y"]), max(3, int(item["radius"]))
        cv2.circle(out, (x, y), r, (120, 120, 120), 1)
        cv2.putText(out, item["id"], (x + r, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)
    for item in accepted:
        x, y, r = int(item["x"]), int(item["y"]), max(4, int(item["radius"]))
        cv2.circle(out, (x, y), r, (0, 220, 0), 3)
        cv2.putText(out, f"{item['id']} {item['mechanical_score']:.2f}", (x + r, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
    cv2.putText(out, label, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 2)
    return out


def _side_by_side(left, right):
    cv2, np = _cv()
    height = max(left.shape[0], right.shape[0])
    def resize(image):
        scale = height / image.shape[0]
        return cv2.resize(image, (int(round(image.shape[1] * scale)), height))
    return np.hstack([resize(left), resize(right)])


def _candidate_contact_sheet(locator, board, locator_candidates, board_candidates):
    """High-resolution full-board comparison plus enlarged edge-hole crops for VLM review."""
    cv2, np = _cv()
    top = _side_by_side(
        _draw_candidates(locator, locator_candidates, [], "LOCATOR edge-hole candidates"),
        _draw_candidates(board, board_candidates, [], "PHOTO edge-hole candidates"),
    )
    tile_w, tile_h = 220, 170
    items = [("LOCATOR", locator, item) for item in locator_candidates[:8]]
    items += [("PHOTO", board, item) for item in board_candidates[:8]]
    cols = 8
    rows = max(1, math.ceil(len(items) / cols))
    strip = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for index, (domain, image, item) in enumerate(items):
        row, col = divmod(index, cols)
        x, y = int(round(item["x"])), int(round(item["y"]))
        radius = max(35, int(round(item["radius"] * 3.2)))
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(image.shape[1], x + radius), min(image.shape[0], y + radius)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (tile_w, tile_h - 28), interpolation=cv2.INTER_AREA)
        ox, oy = col * tile_w, row * tile_h
        strip[oy + 28:oy + tile_h, ox:ox + tile_w] = crop
        cv2.putText(strip, f"{domain} {item['id']}", (ox + 6, oy + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 70, 180), 2)
    if strip.shape[1] != top.shape[1]:
        strip = cv2.resize(strip, (top.shape[1], max(1, int(strip.shape[0] * top.shape[1] / strip.shape[1]))))
    return np.vstack([top, strip])


def _comparison(locator_image, board_image, locator_candidates, board_candidates, pairs):
    cv2, _ = _cv()
    left = _draw_candidates(locator_image, locator_candidates, [], "LOCATOR accepted mechanical landmarks")
    right = _draw_candidates(board_image, board_candidates, [], "PHOTO accepted mechanical landmarks")
    canvas = _side_by_side(left, right)
    right_offset = int(round(left.shape[1] * (canvas.shape[0] / left.shape[0])))
    left_scale = canvas.shape[0] / locator_image.shape[0]
    right_scale = canvas.shape[0] / board_image.shape[0]
    for index, pair in enumerate(pairs, 1):
        lx, ly = pair["locator_px"]
        bx, by = pair["board_px"]
        p1 = (int(lx * left_scale), int(ly * left_scale))
        p2 = (right_offset + int(bx * right_scale), int(by * right_scale))
        color = ((37 * index) % 255, (97 * index) % 255, (173 * index) % 255)
        cv2.line(canvas, p1, p2, color, 2)
        cv2.putText(canvas, f"P{index}", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(canvas, f"P{index}", p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


def _local_tp_refinement(board, projected_x: float, projected_y: float):
    """Use solder pads only as a tightly gated *local* verification step."""
    cv2, np = _cv()
    h, w = board.shape[:2]
    diagonal = math.hypot(w, h)
    roi_radius = max(70, int(round(diagonal * 0.055)))
    x1, y1 = max(0, int(projected_x) - roi_radius), max(0, int(projected_y) - roi_radius)
    x2, y2 = min(w, int(projected_x) + roi_radius), min(h, int(projected_y) + roi_radius)
    roi = board[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 18 or perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.50:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if not (3.0 <= radius <= diagonal * 0.022):
            continue
        gx, gy = float(x1 + cx), float(y1 + cy)
        distance = math.hypot(gx - projected_x, gy - projected_y)
        candidates.append({
            "x": gx, "y": gy, "radius": float(radius),
            "circularity": circularity, "distance_to_projection_px": distance,
        })
    candidates.sort(key=lambda item: (item["distance_to_projection_px"], -item["circularity"]))
    # Merge duplicate inner/outer contours.
    unique: list[dict[str, Any]] = []
    for item in candidates:
        if any((item["x"] - old["x"]) ** 2 + (item["y"] - old["y"]) ** 2 < (0.65 * max(item["radius"], old["radius"])) ** 2 for old in unique):
            continue
        unique.append(item)
    unique = unique[:24]
    selected = unique[0] if unique else None
    snap_limit = max(8.0, diagonal * 0.010)
    distinct = bool(selected) and (len(unique) == 1 or unique[1]["distance_to_projection_px"] >= max(12.0, selected["distance_to_projection_px"] * 1.55))
    snapped = bool(selected and selected["distance_to_projection_px"] <= snap_limit and distinct)
    final_x = float(selected["x"]) if snapped else float(projected_x)
    final_y = float(selected["y"]) if snapped else float(projected_y)
    overlay = board.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)
    for index, item in enumerate(unique, 1):
        color = (0, 220, 0) if snapped and item is selected else (255, 180, 0)
        cv2.circle(overlay, (int(item["x"]), int(item["y"])), max(4, int(item["radius"])), color, 2)
        cv2.putText(overlay, f"C{index}", (int(item["x"]) + 5, int(item["y"])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.drawMarker(overlay, (int(projected_x), int(projected_y)), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.drawMarker(overlay, (int(final_x), int(final_y)), (0, 180, 0) if snapped else (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 3)
    evidence = {
        "projection_px": [round(projected_x, 3), round(projected_y, 3)],
        "roi_bbox": [x1, y1, x2, y2],
        "snap_limit_px": round(snap_limit, 3),
        "candidate_count": len(unique),
        "candidates": unique,
        "selected_candidate": selected,
        "selected_is_distinct": distinct,
        "snapped": snapped,
        "final_px": [round(final_x, 3), round(final_y, 3)],
        "policy": "Solder pads are forbidden as global anchors; nearest distinct pad may refine only inside the projected TP ROI.",
    }
    return final_x, final_y, evidence, overlay


def prepare_landmark_review(locator_path: str | Path, board_path: str | Path, debug_dir: str | Path) -> dict[str, Any]:
    """Generate PCB-edge hole candidates for optional, fail-safe VLM semantic review."""
    locator = _load(locator_path)
    board = _load(board_path)
    locator_contour, _, locator_outline = _board_contour(locator, domain="locator")
    board_contour, _, board_outline = _board_contour(
        board,
        domain="photo",
        expected_aspect_ratio=_contour_aspect_ratio(locator_contour),
    )
    locator_corners = _ordered_box(locator_contour)
    board_corners = _ordered_box(board_contour)
    loc_ok, loc_rejected = _candidate_landmarks(locator, locator_contour, domain="locator")
    brd_ok, brd_rejected = _candidate_landmarks(board, board_contour, domain="photo")
    debug = Path(debug_dir)
    debug.mkdir(parents=True, exist_ok=True)
    _write(debug / "back_01_locator_outline.png", _draw_outline(locator, locator_contour, locator_corners, "LOCATOR PCB outline"))
    _write(debug / "back_01_photo_outline.png", _draw_outline(board, board_contour, board_corners, "PHOTO PCB outline"))
    _json(debug / "back_01_outline_metrics.json", {
        "locator_corners": locator_corners.tolist(), "photo_corners": board_corners.tolist(),
        "locator_quality": locator_outline, "photo_quality": board_outline,
    })
    if not locator_outline["valid"] or not board_outline["valid"]:
        raise ValueError(
            "PCB outline quality gate failed; inspect back_01_*_outline.png and back_01_outline_metrics.json"
        )
    locator_sheet = _draw_candidates(locator, loc_ok, loc_rejected, "LOCATOR PCB-edge hole candidates")
    board_sheet = _draw_candidates(board, brd_ok, brd_rejected, "PHOTO PCB-edge hole candidates")
    _write(debug / "back_02_locator_edge_hole_candidates.png", locator_sheet)
    _write(debug / "back_02_photo_edge_hole_candidates.png", board_sheet)
    pair_sheet = _candidate_contact_sheet(locator, board, loc_ok, brd_ok)
    _write(debug / "back_02_vlm_edge_hole_candidate_sheet.png", pair_sheet)
    payload = {
        "instruction": (
            "VLM must match only true PCB-edge openings: mounting holes, tooling holes, non-plated holes, "
            "or distinctive board cutouts. Ordinary solder pads, vias, TP pads and component circles are forbidden. "
            "Use relative position along the four PCB edges and nearby outline geometry; require confidence >= 0.75."
        ),
        "locator": {"accepted_by_cv": loc_ok, "rejected_by_cv": loc_rejected},
        "photo": {"accepted_by_cv": brd_ok, "rejected_by_cv": brd_rejected},
    }
    _json(debug / "back_02_edge_hole_candidates.json", payload)
    return {
        "locator_candidate_count": len(loc_ok) + len(loc_rejected),
        "photo_candidate_count": len(brd_ok) + len(brd_rejected),
        "candidate_sheet": str(debug / "back_02_vlm_edge_hole_candidate_sheet.png"),
        "candidate_json": str(debug / "back_02_edge_hole_candidates.json"),
    }


def register(
    locator_path: str | Path,
    board_path: str | Path,
    debug_dir: str | Path | None = None,
    review_path: str | Path | None = None,
    locator_roi_hint_norm: list[float] | None = None,
    board_roi_hint_norm: list[float] | None = None,
) -> dict[str, Any]:
    cv2, np = _cv()
    locator = _load(locator_path)
    board = _load(board_path)
    locator_contour, locator_mask, locator_outline = _board_contour(
        locator, domain="locator", roi_hint_norm=locator_roi_hint_norm
    )
    board_contour, board_mask, board_outline = _board_contour(
        board,
        domain="photo",
        roi_hint_norm=board_roi_hint_norm,
        expected_aspect_ratio=_contour_aspect_ratio(locator_contour),
    )
    if not locator_outline["valid"] or not board_outline["valid"]:
        raise ValueError("PCB outline quality gate failed before registration")
    locator_corners = _ordered_box(locator_contour)
    board_corners = _ordered_box(board_contour)
    review_file = Path(review_path) if review_path is not None else None
    review: dict[str, Any] = {}
    review_matches: list[dict[str, Any]] = []
    loc_ok: list[dict[str, Any]] = []
    loc_rejected: list[dict[str, Any]] = []
    brd_ok: list[dict[str, Any]] = []
    brd_rejected: list[dict[str, Any]] = []
    review_error: str | None = None
    if review_file is not None and review_file.is_file():
        try:
            loc_ok, loc_rejected = _candidate_landmarks(locator, locator_contour, domain="locator")
            brd_ok, brd_rejected = _candidate_landmarks(board, board_contour, domain="photo")
            review = json.loads(review_file.read_text(encoding="utf-8"))
            if review.get("review_source") == "vlm_visual_semantic_review" and isinstance(review.get("matches"), list):
                review_matches = review["matches"]
        except Exception as exc:  # VLM review is advisory; rectangle path must remain available.
            review_error = str(exc)

    h, w = board.shape[:2]
    rectangle_matrix = cv2.getPerspectiveTransform(
        locator_corners.astype(np.float32), board_corners.astype(np.float32)
    )
    threshold = max(12.0, math.hypot(w, h) * 0.035)
    best = {
        "rotation_quadrants": 0,
        "mirrored": False,
        "outline_iou": _outline_iou(locator_mask, board_mask, rectangle_matrix),
        "pair_count": 0,
        "reviewed_pair_count": len(review_matches),
        "mean_error_px": 0.0,
        "matching_threshold_px": threshold,
        "score": 0.0,
        "pairs": [],
        "matrix": rectangle_matrix,
    }
    matrix = rectangle_matrix
    score_margin = 1.0
    orientation_source = "fixed_fixture_tl_to_tl_no_rotation_no_mirror"
    refinement_validation: dict[str, Any] = {
        "attempted": bool(review_file is not None and review_file.is_file()),
        "accepted": False,
        "orientation_locked": True,
        "rotation_quadrants": 0,
        "mirrored": False,
        "hole_validation_passed": False,
        "fallback_reason": review_error or "no_valid_vlm_edge_hole_review",
    }
    if not review_error and review_matches:
        locator_by_id = {
            item["id"]: item for item in (loc_ok + loc_rejected)
        }
        board_by_id = {
            item["id"]: item for item in (brd_ok + brd_rejected)
        }
        validated_pairs: list[dict[str, Any]] = []
        pair_errors: list[float] = []
        for reviewed in review_matches:
            locator_item = locator_by_id.get(str(reviewed.get("locator_id", "")))
            board_item = board_by_id.get(str(reviewed.get("board_id", "")))
            if locator_item is None or board_item is None:
                continue
            source_point = np.float32([[[locator_item["x"], locator_item["y"]]]])
            projected_point = cv2.perspectiveTransform(source_point, rectangle_matrix)[0, 0]
            distance = float(np.linalg.norm(
                projected_point - np.float32([board_item["x"], board_item["y"]])
            ))
            pair_errors.append(distance)
            if distance <= threshold:
                validated_pairs.append({
                    "locator_id": locator_item["id"],
                    "board_id": board_item["id"],
                    "locator_px": [locator_item["x"], locator_item["y"]],
                    "board_px": [board_item["x"], board_item["y"]],
                    "projected_px": [float(projected_point[0]), float(projected_point[1])],
                    "error_px": distance,
                })
        best["pairs"] = validated_pairs
        best["pair_count"] = len(validated_pairs)
        best["mean_error_px"] = (
            float(sum(pair["error_px"] for pair in validated_pairs) / len(validated_pairs))
            if validated_pairs else 0.0
        )
        hole_validation_passed = len(validated_pairs) >= 2
        refinement_validation = {
            "attempted": True,
            "accepted": False,
            "orientation_locked": True,
            "rotation_quadrants": 0,
            "mirrored": False,
            "hole_validation_passed": hole_validation_passed,
            "reviewed_pair_count": len(review_matches),
            "inlier_pair_count": len(validated_pairs),
            "reviewed_errors_px": [round(value, 3) for value in pair_errors],
            "fallback_reason": None if hole_validation_passed else "fixed_orientation_hole_validation_insufficient",
        }
    tx, ty = _green_tp(locator)
    projected = cv2.perspectiveTransform(np.float32([[[tx, ty]]]), matrix)[0, 0]
    projected_x = float(np.clip(projected[0], 0, w - 1))
    projected_y = float(np.clip(projected[1], 0, h - 1))
    px, py, local_evidence, local_overlay = _local_tp_refinement(board, projected_x, projected_y)
    errors = [float(pair["error_px"]) for pair in best["pairs"]]
    p95 = float(np.percentile(errors, 95)) if errors else 0.0
    threshold = float(best["matching_threshold_px"])
    confidence = max(0.0, min(0.95,
        0.18 + 0.10 * best["pair_count"] + 0.28 * best["outline_iou"]
        + min(score_margin, 2.0) * 0.08 - min(p95 / max(threshold, 1.0), 2.0) * 0.16
    ))
    result: dict[str, Any] = {
        "source": "back_board_outline_mechanical_landmarks_homography",
        "mapping_method": "back_board_outline_holes",
        "board_roi_target_px_approx": [round(px, 3), round(py, 3)],
        "board_roi_target_px_before_local_refine": [round(projected_x, 3), round(projected_y, 3)],
        "tp_locator_center": [round(tx, 3), round(ty, 3)],
        "homography_3x3": [[round(float(value), 9) for value in row] for row in matrix.tolist()],
        "orientation": {"rotation_quadrants": best["rotation_quadrants"], "mirrored": best["mirrored"]},
        "orientation_source": orientation_source,
        "registration_selection": "fixed_outline_holes_validated" if refinement_validation.get("hole_validation_passed") else "fixed_outline",
        "edge_hole_refinement_validation": refinement_validation,
        "outline_iou": round(float(best["outline_iou"]), 4),
        "inlier_hole_count": int(best["pair_count"]),
        "mean_hole_error_px": round(float(best["mean_error_px"]), 3),
        "p95_landmark_error_px": round(p95, 3),
        "landmark_acceptance_threshold_px": round(threshold, 3),
        "orientation_score_margin": round(score_margin, 4),
        "confidence": round(float(confidence), 3),
        "landmark_pairs": best["pairs"],
        "locator_accepted_count": len(loc_ok), "photo_accepted_count": len(brd_ok),
        "locator_rejected_count": len(loc_rejected), "photo_rejected_count": len(brd_rejected),
        "image_size": [int(w), int(h)],
        "local_tp_verification": local_evidence,
        "vlm_landmark_review": {
            "review_path": str(review_file) if review_file is not None and review_file.is_file() else None,
            "overall_evidence": review.get("overall_evidence", ""),
            "reviewed_match_count": len(review_matches),
            "optional": True,
            "review_error": review_error,
        },
    }

    if debug_dir is not None:
        debug = Path(debug_dir)
        debug.mkdir(parents=True, exist_ok=True)
        locator_outline_image = _draw_outline(locator, locator_contour, locator_corners, "LOCATOR PCB outline")
        board_outline_image = _draw_outline(board, board_contour, board_corners, "PHOTO PCB outline")
        _write(debug / "back_01_locator_outline.png", locator_outline_image)
        _write(debug / "back_01_photo_outline.png", board_outline_image)
        _json(debug / "back_01_outline_metrics.json", {
            "locator_corners": locator_corners.tolist(), "photo_corners": board_corners.tolist(),
            "locator_quality": locator_outline, "photo_quality": board_outline,
            "selected_outline_iou": best["outline_iou"],
        })
        if review_matches:
            locator_candidates = _draw_candidates(locator, loc_ok, loc_rejected, "LOCATOR landmark classification")
            board_candidates = _draw_candidates(board, brd_ok, brd_rejected, "PHOTO landmark classification")
            _write(debug / "back_04_vlm_hole_pair_comparison.png", _comparison(locator, board, loc_ok, brd_ok, best["pairs"]))
            _json(debug / "back_04_vlm_hole_refinement_validation.json", refinement_validation)
            _write(debug / "back_04_locator_hole_review.png", locator_candidates)
            _write(debug / "back_04_photo_hole_review.png", board_candidates)
        warped = cv2.warpPerspective(locator, matrix, (w, h))
        reprojection = cv2.addWeighted(board, 0.65, warped, 0.35, 0)
        for pair in best["pairs"]:
            bx, by = map(int, pair["board_px"])
            px2, py2 = map(int, pair["projected_px"])
            cv2.line(reprojection, (bx, by), (px2, py2), (0, 0, 255), 2)
        _write(debug / "back_04_reprojection_overlay.png", reprojection)
        _json(debug / "back_04_registration_validation.json", {
            "note": "Landmarks choose/validate an outline-derived transform; errors are independent of the four outline corners.",
            "errors_px": errors, "mean_error_px": best["mean_error_px"], "p95_error_px": p95,
            "threshold_px": threshold, "passed": bool(p95 <= threshold and score_margin >= 0.12),
            "registration_selection": result["registration_selection"],
            "edge_hole_refinement_validation": refinement_validation,
        })
        projection = board.copy()
        cv2.drawMarker(projection, (int(round(px)), int(round(py))), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.circle(projection, (int(round(px)), int(round(py))), max(30, int(0.025 * math.hypot(w, h))), (0, 165, 255), 2)
        cv2.putText(projection, "Projected TP - requires local pad verification", (max(5, int(px) + 20), max(28, int(py) - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        _write(debug / "back_05_tp_projection.png", projection)
        _write(debug / "back_06_tp_local_candidates.png", local_overlay)
        _json(debug / "back_06_tp_local_candidates.json", local_evidence)
        _json(debug / "back_registration_summary.json", result)
    return result


def draw_overlay(board_path: str | Path, result: dict[str, Any], output_path: str | Path) -> None:
    cv2, _ = _cv()
    image = _load(board_path)
    for pair in result.get("landmark_pairs", []):
        x, y = pair["board_px"]
        cv2.circle(image, (int(round(x)), int(round(y))), 12, (255, 180, 0), 2)
    target = result["board_roi_target_px_approx"]
    x, y = int(round(target[0])), int(round(target[1]))
    cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 36, 3)
    cv2.putText(image, "TP (back registration)", (max(0, x + 18), max(24, y - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    _write(Path(output_path), image)
