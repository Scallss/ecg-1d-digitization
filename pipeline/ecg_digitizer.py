from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PARAMS: dict[str, Any] = {
    "preprocess_mode": "current_then_hybrid",  # "current" or "current_then_hybrid"
    "mask_method": "red_excess",
    "lab_a_thresh": 120,
    "use_otsu_for_red_excess": True,
    "red_excess_manual_thr": 12,
    "binarize": "otsu",
    "fixed_gray_thr": 100,
    "gray_thr_bright": 70,
    "dark_thresh": 121,
    "gray_thresh": 151,
    "spread_thresh": 38,
    "open_k": 1,
    "baseline_smooth_ksize": 61,
    "baseline_num_peaks": 4,
    "baseline_min_sep_ratio": 0.18,  # matches detect_baselines' previously-hardcoded 0.18; kept as the default so wiring this param up doesn't change existing behavior
    "sep_vclose_h": 65,
    "sep_vopen_ratio": 0.18,
    "sep_min_height_ratio": 0.10,
    "sep_max_width_px": 50,
    "sep_min_aspect": 4.0,
    "sep_cluster_tol_px": 24,
    "sep_min_cluster_members": 2,
    "sep_border_exclude_ratio": 0.03,
    "num_cols": 4,
    "crop_halfspan_ratio": 0.70,
    "rhythm_crop_extra_ratio": 0.08,
    "lr_pad_ratio": 0.015,
    "left_margin_ratio": 0.04,
    "right_margin_ratio": 0.02,
    "right_boundary_mode": "margin",  # "margin" (default, unchanged) or "extrapolate" (mirrors the left-edge logic: projects one more column-width past the last detected separator, instead of a fixed width-fraction cutoff)
    "dilate_k": 3,
    "top_cc_enable": True,
    "top_cc_y_thresh_px": 45,
    "top_cc_y_anchor_offset_px": 3,
    "top_cc_min_width_px": 25,
    "top_cc_min_coverage_ratio": 0.25,
    "rhythm_bottom_cc_enable": True,
    "rhythm_bottom_cc_y_thresh_px": 15,
    "iterative_recovery_enable": True,
    "iterative_recovery_initial_halfspan_ratio": 0.70,
    "iterative_recovery_max_iters": 10,
    "iterative_recovery_target_span_ratio": 0.95,
    "iterative_recovery_step_ratio": 0.06,
    "iterative_recovery_soft_step_ratio": 0.03,
    "interp_dt_ms": 2.0,
    "time_seconds_total": 10.0,
    "voltage_time_ratio_mV_per_ms": 0.0025,
    "overlap_ratio_threshold": 0.03,
    "rhythm_overlap_ratio_threshold": 0.008,
    "overlap_vertical_close_h": 7,
    "overlap_column_count_threshold": 3,
}

LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TOP_ROW_LEADS = {"I", "aVR", "V1", "V4"}
TOP_EDGE_PX = 2
BOTTOM_EDGE_PX = 2


@dataclass
class LeadResult:
    lead: str
    crop_binary: np.ndarray
    crop_box: tuple[int, int, int, int]
    connected_mask: np.ndarray
    processed_trace_view_bgr: np.ndarray
    y_trace: np.ndarray | None
    t_ms: np.ndarray | None
    v_mV: np.ndarray | None
    interpolation_mask: np.ndarray | None = None
    overlap_result: dict[str, Any] | None = None
    legacy_overlap_result: dict[str, Any] | None = None
    overlap_mask: np.ndarray | None = None
    overlap_processed_trace_view_bgr: np.ndarray | None = None


@dataclass
class DigitizationResult:
    image_path: Path
    raw_image_bgr: np.ndarray
    whitened_image_bgr: np.ndarray
    ink_mask: np.ndarray
    baselines: list[int]
    separators: list[float]
    bounds: list[int]
    rhythm_strip: np.ndarray | None
    time_res_ms_per_px: float | None
    volt_res_mV_per_px: float | None
    lead_results: dict[str, LeadResult]
    overlap_detection_mode: str = "vertical_close_column_count"
    overlap_vertical_close_h: int | None = None
    overlap_column_count_threshold: int | None = None
    rhythm_overlap_result: dict[str, Any] | None = None
    legacy_rhythm_overlap_result: dict[str, Any] | None = None
    rhythm_interpolation_mask: np.ndarray | None = None
    rhythm_overlap_mask: np.ndarray | None = None
    rhythm_overlap_processed_trace_view_bgr: np.ndarray | None = None
    rhythm_crop_box: tuple[int, int, int, int] | None = None


@dataclass
class LeadRecoveryMetrics:
    top_edge_mask_fraction: float
    bottom_edge_mask_fraction: float
    top_edge_trace_fraction: float
    bottom_edge_trace_fraction: float
    raw_top_ink_fraction: float
    raw_bottom_ink_fraction: float
    edge_flag: bool
    x_span_ratio: float
    x_coverage_ratio: float
    max_empty_gap_ratio: float


def digitize_ecg_image(image_path: str | Path, params: dict[str, Any] | None = None) -> DigitizationResult:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    p = {**DEFAULT_PARAMS, **(params or {})}
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = img.shape[:2]
    frame_top_y = detect_red_frame_top(img)
    whitened, ink = preprocess_for_digitization(img, p)
    baselines = detect_baselines(ink, height, width, p)
    if len(baselines) < 3:
        raise ValueError(f"Expected at least 3 baselines, found {len(baselines)}")

    separators = detect_column_separators(ink, baselines, height, width, p)
    bounds, rhythm, lead_results, rhythm_crop_box = extract_lead_results_with_recovery(
        img,
        ink,
        baselines,
        separators,
        height,
        width,
        p,
        frame_top_y=frame_top_y,
    )

    time_res_ms = None
    volt_res_mV = None
    ref_w = rhythm.shape[1] if rhythm is not None else max(result.crop_binary.shape[1] for result in lead_results.values())
    if ref_w > 0:
        time_res_ms = (float(p["time_seconds_total"]) * 1000.0) / ref_w
        volt_res_mV = time_res_ms * float(p["voltage_time_ratio_mV_per_ms"])

    lead_results = attach_calibrated_signals(lead_results, time_res_ms, volt_res_mV, float(p["interp_dt_ms"]))
    (
        lead_results,
        rhythm_overlap_result,
        legacy_rhythm_overlap_result,
        rhythm_interpolation_mask,
        rhythm_overlap_mask,
        rhythm_overlap_trace_view_bgr,
    ) = attach_overlap_annotations(lead_results, rhythm, p, frame_top_y=frame_top_y)

    return DigitizationResult(
        image_path=image_path,
        raw_image_bgr=img,
        whitened_image_bgr=whitened,
        ink_mask=ink,
        baselines=baselines,
        separators=separators,
        bounds=bounds,
        rhythm_strip=rhythm,
        time_res_ms_per_px=time_res_ms,
        volt_res_mV_per_px=volt_res_mV,
        lead_results=lead_results,
        overlap_vertical_close_h=int(p["overlap_vertical_close_h"]),
        overlap_column_count_threshold=int(p["overlap_column_count_threshold"]),
        rhythm_overlap_result=rhythm_overlap_result,
        legacy_rhythm_overlap_result=legacy_rhythm_overlap_result,
        rhythm_interpolation_mask=rhythm_interpolation_mask,
        rhythm_overlap_mask=rhythm_overlap_mask,
        rhythm_overlap_processed_trace_view_bgr=rhythm_overlap_trace_view_bgr,
        rhythm_crop_box=rhythm_crop_box,
    )


def extract_lead_results_with_recovery(
    raw_image_bgr: np.ndarray,
    ink: np.ndarray,
    baselines: list[int],
    separators: list[float],
    height: int,
    width: int,
    params: dict[str, Any],
    frame_top_y: int | None = None,
) -> tuple[list[int], np.ndarray | None, dict[str, LeadResult], tuple[int, int, int, int] | None]:
    initial_halfspan_ratio = float(params.get("iterative_recovery_initial_halfspan_ratio", params["crop_halfspan_ratio"]))
    crop_params = {**params, "crop_halfspan_ratio": initial_halfspan_ratio}
    bounds, rhythm, lead_crops = crop_leads(ink, baselines, separators, height, width, crop_params)

    lead_results: dict[str, LeadResult] = {}
    metrics: dict[str, LeadRecoveryMetrics] = {}
    for lead, (crop, box) in lead_crops.items():
        lead_result, metric = build_lead_result_from_crop(lead, crop, box, params, frame_top_y=frame_top_y)
        lead_results[lead] = lead_result
        metrics[lead] = metric

    baselines_3 = baselines[:3]
    vd = float(np.mean(np.diff(baselines_3))) if len(baselines_3) >= 2 else height / 4
    base_halfspan = int(initial_halfspan_ratio * vd)
    rhythm_base_halfspan = base_halfspan + int(float(params.get("rhythm_crop_extra_ratio", 0.08)) * vd)
    lr_pad = int(float(params["lr_pad_ratio"]) * width)
    rhythm_left = int(bounds[0])
    rhythm_right = int(bounds[-1])
    rhythm_x0 = int(max(0, rhythm_left + lr_pad))
    rhythm_x1 = int(min(width - 1, rhythm_right - lr_pad))

    # Box for `rhythm` as currently held (matches crop_leads()'s own formula, which used
    # the same initial_halfspan_ratio + rhythm_crop_extra_ratio combination) -- tracked
    # explicitly so callers can get the box that actually matches the returned array,
    # rather than re-deriving it (and risking drift, e.g. from the iterative-recovery
    # growth below not being replicated externally).
    rhythm_box: tuple[int, int, int, int] | None = None
    if rhythm is not None and len(baselines) >= 4:
        _baseline0 = baselines[3]
        _y0 = int(max(0, _baseline0 - rhythm_base_halfspan))
        _y1 = int(min(height - 1, _baseline0 + rhythm_base_halfspan))
        rhythm_box = (rhythm_x0, _y0, rhythm_x1, _y1)

    if not bool(params.get("iterative_recovery_enable", True)):
        return bounds, rhythm, lead_results, rhythm_box

    step = max(5, int(float(params["iterative_recovery_step_ratio"]) * vd))
    soft_step = max(3, int(float(params["iterative_recovery_soft_step_ratio"]) * vd))
    max_iters = int(params["iterative_recovery_max_iters"])
    target_span_ratio = float(params["iterative_recovery_target_span_ratio"])

    row_lookup = {lead_name: row for row, row_leads in enumerate(LEAD_LAYOUT) for lead_name in row_leads}
    col_lookup = {lead_name: col for row_leads in LEAD_LAYOUT for col, lead_name in enumerate(row_leads)}

    for lead in detect_clipped_leads(metrics):
        current_result = lead_results[lead]
        current_metric = metrics[lead]
        row_idx = row_lookup[lead]
        col_idx = col_lookup[lead]
        baseline = baselines_3[row_idx]

        grow_up = 0
        grow_down = 0
        current_box = current_result.crop_box
        x0 = int(max(0, bounds[col_idx] + lr_pad))
        x1 = int(min(width - 1, bounds[col_idx + 1] - lr_pad))

        for _ in range(max_iters):
            if current_metric.x_span_ratio >= target_span_ratio:
                break

            top_heavy = current_metric.top_edge_mask_fraction > 0.01 or current_metric.top_edge_trace_fraction > 0.01
            bottom_heavy = current_metric.bottom_edge_mask_fraction > 0.01 or current_metric.bottom_edge_trace_fraction > 0.01

            if top_heavy and not bottom_heavy:
                next_up = grow_up + step
                next_down = grow_down + soft_step
            elif bottom_heavy and not top_heavy:
                next_up = grow_up + soft_step
                next_down = grow_down + step
            else:
                next_up = grow_up + step
                next_down = grow_down + step

            if next_up == grow_up and next_down == grow_down:
                break

            y0 = max(0, int(baseline - base_halfspan - next_up))
            y1 = min(height - 1, int(baseline + base_halfspan + next_down))
            if (x0, y0, x1, y1) == current_box:
                break

            crop = ink[y0:y1, x0:x1].copy()
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                break

            current_result, current_metric = build_lead_result_from_crop(
                lead,
                crop,
                (x0, y0, x1, y1),
                params,
                frame_top_y=frame_top_y,
            )
            current_box = current_result.crop_box
            grow_up = next_up
            grow_down = next_down

        lead_results[lead] = current_result

    if rhythm is not None and len(baselines) >= 4:
        baseline = baselines[3]
        y0 = int(max(0, baseline - rhythm_base_halfspan))
        y1 = int(min(height - 1, baseline + rhythm_base_halfspan))
        rhythm_result, rhythm_metric = build_lead_result_from_crop(
            None,
            rhythm,
            (rhythm_x0, y0, rhythm_x1, y1),
            params,
            frame_top_y=frame_top_y,
        )

        if rhythm_metric.edge_flag:
            grow_up = 0
            grow_down = 0
            current_result = rhythm_result
            current_metric = rhythm_metric
            current_box = current_result.crop_box

            for _ in range(max_iters):
                if current_metric.x_span_ratio >= target_span_ratio:
                    break

                top_heavy = current_metric.top_edge_mask_fraction > 0.01 or current_metric.top_edge_trace_fraction > 0.01
                bottom_heavy = current_metric.bottom_edge_mask_fraction > 0.01 or current_metric.bottom_edge_trace_fraction > 0.01

                if top_heavy and not bottom_heavy:
                    next_up = grow_up + step
                    next_down = grow_down + soft_step
                elif bottom_heavy and not top_heavy:
                    next_up = grow_up + soft_step
                    next_down = grow_down + step
                else:
                    next_up = grow_up + step
                    next_down = grow_down + step

                if next_up == grow_up and next_down == grow_down:
                    break

                y0 = max(0, int(baseline - rhythm_base_halfspan - next_up))
                y1 = min(height - 1, int(baseline + rhythm_base_halfspan + next_down))
                if (rhythm_x0, y0, rhythm_x1, y1) == current_box:
                    break

                crop = ink[y0:y1, rhythm_x0:rhythm_x1].copy()
                if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                    break

                current_result, current_metric = build_lead_result_from_crop(
                    None,
                    crop,
                    (rhythm_x0, y0, rhythm_x1, y1),
                    params,
                    frame_top_y=frame_top_y,
                )
                current_box = current_result.crop_box
                grow_up = next_up
                grow_down = next_down

            rhythm = current_result.crop_binary
            rhythm_box = current_box

    return bounds, rhythm, lead_results, rhythm_box


def build_lead_result_from_crop(
    lead: str | None,
    crop: np.ndarray,
    box: tuple[int, int, int, int],
    params: dict[str, Any],
    frame_top_y: int | None = None,
) -> tuple[LeadResult, LeadRecoveryMetrics]:
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        mask = np.zeros_like(crop)
        y_trace = None
        interpolation_mask = np.zeros(crop.shape[1], dtype=np.uint8)
    else:
        y_trace, debug = extract_trace(
            crop,
            dilate_k=int(params["dilate_k"]),
            lead=lead,
            params=params,
            crop_y0=box[1],
            frame_top_y=frame_top_y,
        )
        mask = debug["mask"]
        interpolation_mask = debug.get("interpolation_mask")
        if interpolation_mask is None:
            interpolation_mask = np.zeros(crop.shape[1], dtype=np.uint8)

    lead_result = LeadResult(
        lead=lead,
        crop_binary=crop,
        crop_box=box,
        connected_mask=mask,
        processed_trace_view_bgr=build_processed_trace_view(mask, y_trace),
        y_trace=y_trace,
        t_ms=None,
        v_mV=None,
        interpolation_mask=interpolation_mask,
    )
    metrics = compute_recovery_metrics(mask, y_trace, crop)
    return lead_result, metrics


def attach_calibrated_signals(
    lead_results: dict[str, LeadResult],
    time_res_ms: float | None,
    volt_res_mV: float | None,
    interp_dt_ms: float,
) -> dict[str, LeadResult]:
    calibrated: dict[str, LeadResult] = {}
    for lead, lead_result in lead_results.items():
        y_trace = lead_result.y_trace
        t_ms = None
        v_mV = None
        if y_trace is not None and time_res_ms is not None and volt_res_mV is not None:
            valid = ~np.isnan(y_trace)
            if valid.sum() >= 20:
                y0 = np.nanmedian(y_trace[valid])
                amp_px = y0 - y_trace
                amp_mV = amp_px * volt_res_mV
                t_px = np.arange(len(amp_mV), dtype=np.float32) * time_res_ms
                t_ms = np.arange(0, t_px[-1] + 1e-6, interp_dt_ms, dtype=np.float32)
                v_mV = np.interp(t_ms, t_px[valid], amp_mV[valid], left=np.nan, right=np.nan)

        calibrated[lead] = LeadResult(
            lead=lead_result.lead,
            crop_binary=lead_result.crop_binary,
            crop_box=lead_result.crop_box,
            connected_mask=lead_result.connected_mask,
            processed_trace_view_bgr=lead_result.processed_trace_view_bgr,
            y_trace=lead_result.y_trace,
            t_ms=t_ms,
            v_mV=v_mV,
            interpolation_mask=lead_result.interpolation_mask,
            overlap_result=lead_result.overlap_result,
            legacy_overlap_result=lead_result.legacy_overlap_result,
            overlap_mask=lead_result.overlap_mask,
            overlap_processed_trace_view_bgr=lead_result.overlap_processed_trace_view_bgr,
        )
    return calibrated


def compute_recovery_metrics(
    connected_mask: np.ndarray,
    y_trace: np.ndarray | None,
    crop_binary: np.ndarray,
) -> LeadRecoveryMetrics:
    mask = connected_mask > 0
    h, w = mask.shape
    area = int(mask.sum())
    top_edge_mask_fraction = float(mask[:TOP_EDGE_PX, :].sum() / area) if area else 0.0
    bottom_edge_mask_fraction = float(mask[-BOTTOM_EDGE_PX:, :].sum() / area) if area else 0.0

    top_edge_trace_fraction = 0.0
    bottom_edge_trace_fraction = 0.0
    if y_trace is not None:
        valid = ~np.isnan(y_trace)
        if valid.any():
            top_edge_trace_fraction = float((y_trace[valid] <= 1.5).mean())
            bottom_edge_trace_fraction = float((y_trace[valid] >= (h - 2.5)).mean())

    crop = (crop_binary > 0).astype(np.uint8) * 255
    raw_area = int((crop > 0).sum())
    raw_top_ink_fraction = float((crop[:TOP_EDGE_PX, :] > 0).sum() / raw_area) if raw_area else 0.0
    raw_bottom_ink_fraction = float((crop[-BOTTOM_EDGE_PX:, :] > 0).sum() / raw_area) if raw_area else 0.0

    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        component_count = 0
        largest_component_fraction = 0.0
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        dilated = cv2.dilate(crop, kernel, iterations=1)
        num, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        component_count = max(0, num - 1)
        if component_count > 0:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_component_fraction = float(np.max(areas) / max(1, np.sum(areas)))
        else:
            largest_component_fraction = 0.0

    cols_with_mask = np.where(mask.any(axis=0))[0]
    if cols_with_mask.size:
        x_span_ratio = float((cols_with_mask[-1] - cols_with_mask[0] + 1) / max(1, w))
        x_coverage_ratio = float(cols_with_mask.size / max(1, w))
        gaps = np.diff(cols_with_mask) - 1
        max_empty_gap_ratio = float(max(0, gaps.max(initial=0)) / max(1, w))
    else:
        x_span_ratio = 0.0
        x_coverage_ratio = 0.0
        max_empty_gap_ratio = 1.0

    horizontal_flag = x_span_ratio < 0.90 or x_coverage_ratio < 0.85 or max_empty_gap_ratio > 0.05
    edge_flag = (
        top_edge_mask_fraction > 0.01
        or bottom_edge_mask_fraction > 0.01
        or top_edge_trace_fraction > 0.01
        or bottom_edge_trace_fraction > 0.01
        or raw_top_ink_fraction > 0.01
        or raw_bottom_ink_fraction > 0.01
        or ((raw_top_ink_fraction > 0.003 or raw_bottom_ink_fraction > 0.003) and component_count >= 3 and largest_component_fraction < 0.80)
        or horizontal_flag
    )

    return LeadRecoveryMetrics(
        top_edge_mask_fraction=top_edge_mask_fraction,
        bottom_edge_mask_fraction=bottom_edge_mask_fraction,
        top_edge_trace_fraction=top_edge_trace_fraction,
        bottom_edge_trace_fraction=bottom_edge_trace_fraction,
        raw_top_ink_fraction=raw_top_ink_fraction,
        raw_bottom_ink_fraction=raw_bottom_ink_fraction,
        edge_flag=edge_flag,
        x_span_ratio=x_span_ratio,
        x_coverage_ratio=x_coverage_ratio,
        max_empty_gap_ratio=max_empty_gap_ratio,
    )


def detect_clipped_leads(metrics: dict[str, LeadRecoveryMetrics]) -> list[str]:
    return [lead for lead, metric in metrics.items() if metric.edge_flag]


def detect_border_red_lines(mask_red: np.ndarray, height: int, width: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_red, connectivity=8)
    border_mask = np.zeros_like(mask_red)

    max_thickness = max(3, int(0.012 * min(height, width)))
    min_hline_len = int(0.35 * width)
    min_vline_len = int(0.35 * height)
    border_band = max(2, int(0.01 * min(height, width)))

    for component in range(1, num):
        x, y, w, h, _ = stats[component]
        touches_border = (
            x <= border_band
            or y <= border_band
            or (x + w) >= (width - border_band)
            or (y + h) >= (height - border_band)
        )
        if not touches_border:
            continue

        thin_horizontal = h <= max_thickness and w >= min_hline_len
        thin_vertical = w <= max_thickness and h >= min_vline_len
        if thin_horizontal or thin_vertical:
            border_mask[labels == component] = 255

    return border_mask


def whiten_red_grid(img_bgr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    _, a_chan, _ = cv2.split(lab)
    b_chan, g_chan, r_chan = cv2.split(img_bgr)
    red_excess = r_chan.astype(np.int16) - np.maximum(g_chan, b_chan).astype(np.int16)
    red_excess_clip = np.clip(red_excess, 0, 255).astype(np.uint8)

    if params["use_otsu_for_red_excess"]:
        threshold, _ = cv2.threshold(red_excess_clip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        red_excess_thr = int(threshold)
    else:
        red_excess_thr = int(params["red_excess_manual_thr"])

    mask_lab = (a_chan >= int(params["lab_a_thresh"])).astype(np.uint8) * 255
    mask_re = (red_excess_clip >= red_excess_thr).astype(np.uint8) * 255

    if params["mask_method"] == "lab_a":
        red_mask = mask_lab
    elif params["mask_method"] == "red_excess":
        red_mask = mask_re
    else:
        red_mask = cv2.bitwise_or(mask_lab, mask_re)

    red_mask_raw = red_mask.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[red_mask_raw > 0]
    if vals.size > 0:
        thr = int(np.clip(np.percentile(vals, 0.25), 60, 130))
    else:
        thr = int(params["gray_thr_bright"])

    bright = (gray >= thr).astype(np.uint8) * 255
    border_red = detect_border_red_lines(red_mask_raw, img_bgr.shape[0], img_bgr.shape[1])
    red_mask = cv2.bitwise_or(cv2.bitwise_and(red_mask_raw, bright), border_red)

    whitened = img_bgr.copy()
    whitened[red_mask > 0] = (255, 255, 255)
    return whitened


def build_ink_mask(img_bgr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if params["binarize"] == "otsu":
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return ink
    return (gray <= int(params["fixed_gray_thr"])).astype(np.uint8) * 255


def build_dark_hybrid_ink_mask(img_bgr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    max_channel = img_bgr.max(axis=2)
    min_channel = img_bgr.min(axis=2)
    spread = max_channel - min_channel

    strong_black = max_channel <= int(params["dark_thresh"])
    neutral_dark = (gray <= int(params["gray_thresh"])) & (spread <= int(params["spread_thresh"]))
    ink = (strong_black | neutral_dark).astype(np.uint8) * 255

    open_k = int(params["open_k"])
    if open_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)
    return ink


def preprocess_for_digitization(img_bgr: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mode = str(params.get("preprocess_mode", "current"))
    whitened = whiten_red_grid(img_bgr, params)
    if mode == "current":
        return whitened, build_ink_mask(whitened, params)
    if mode == "current_then_hybrid":
        ink = build_dark_hybrid_ink_mask(whitened, params)
        filtered = np.full_like(img_bgr, 255)
        filtered[ink > 0] = (0, 0, 0)
        return filtered, ink
    raise ValueError(f"Unknown preprocess_mode: {mode}")


def detect_baselines(ink: np.ndarray, height: int, width: int, params: dict[str, Any]) -> list[int]:
    x_left = int(0.12 * width)
    x_right = int(0.98 * width)
    ink_roi = ink[:, x_left:x_right]

    vertical_kernel_h = max(25, int(0.20 * height))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, vertical_kernel_h))
    vertical_only = cv2.morphologyEx(ink_roi, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    ink_no_vertical = cv2.bitwise_and(ink_roi, cv2.bitwise_not(vertical_only))

    band_h = max(9, int(0.03 * height))
    band_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, band_h))
    ink_band = cv2.dilate(ink_no_vertical, band_kernel, iterations=1)

    projection = (ink_band > 0).sum(axis=1).astype(np.float32)
    ksize = int(params["baseline_smooth_ksize"])
    if ksize % 2 == 0:
        ksize += 1
    smooth = cv2.GaussianBlur(projection.reshape(-1, 1), (1, ksize), 0).ravel()

    y_min = int(0.05 * height)
    y_max = int(0.95 * height)
    smooth[:y_min] = -1e9
    smooth[y_max:] = -1e9
    min_sep = int(float(params.get("baseline_min_sep_ratio", 0.18)) * height)
    return pick_peaks_local(smooth, top_k=int(params["baseline_num_peaks"]), min_sep=min_sep)


def pick_peaks_local(values: np.ndarray, top_k: int, min_sep: int) -> list[int]:
    arr = np.asarray(values)
    candidates = np.where((arr[1:-1] >= arr[:-2]) & (arr[1:-1] >= arr[2:]))[0] + 1
    candidates = candidates[np.argsort(arr[candidates])[::-1]]

    peaks: list[int] = []
    for idx in candidates:
        if all(abs(int(idx) - peak) > min_sep for peak in peaks):
            peaks.append(int(idx))
        if len(peaks) >= top_k:
            break
    return sorted(peaks)


def detect_column_separators(
    ink: np.ndarray,
    baselines: list[int],
    height: int,
    width: int,
    params: dict[str, Any],
) -> list[float]:
    baselines_3 = baselines[:3]
    if len(baselines_3) < 3:
        raise ValueError("Need three baselines to detect column separators")

    vd = float(np.mean(np.diff(baselines_3))) if len(baselines_3) >= 2 else height / 4
    halfspan = int(float(params["crop_halfspan_ratio"]) * vd)

    per_row_x: list[list[float]] = []
    for baseline in baselines_3:
        y0 = max(0, int(baseline - halfspan))
        y1 = min(height, int(baseline + halfspan))
        row = ink[y0:y1, :]
        row_v = detect_separators_in_row(row, y1 - y0, params)

        num, labels, stats, centers = cv2.connectedComponentsWithStats(row_v, connectivity=8)
        baseline_local = int(baseline - y0)
        xs: list[float] = []
        for component in range(1, num):
            _, y, w, h, _ = stats[component]
            cx, _ = centers[component]
            if cx < 0.10 * width or cx > 0.97 * width:
                continue
            if h < 0.20 * (y1 - y0):
                continue
            if w > 60:
                continue
            if (h / max(w, 1)) < 2.5:
                continue
            if not np.any(labels[baseline_local, :] == component):
                continue
            xs.append(float(cx))
        per_row_x.append(xs)

    all_x: list[float] = []
    row_id: list[int] = []
    for idx, xs in enumerate(per_row_x):
        for x in xs:
            all_x.append(x)
            row_id.append(idx)

    if not all_x:
        return fallback_separators(width, params)

    all_x_arr = np.array(all_x, dtype=np.float32)
    row_id_arr = np.array(row_id, dtype=np.int32)

    clusters: list[dict[str, Any]] = []
    for order_idx in np.argsort(all_x_arr):
        x_val = float(all_x_arr[order_idx])
        row_val = int(row_id_arr[order_idx])
        if not clusters:
            clusters.append({"xs": [x_val], "rows": {row_val}})
            continue

        cluster_mean = float(np.mean(clusters[-1]["xs"]))
        if abs(x_val - cluster_mean) <= int(params["sep_cluster_tol_px"]):
            clusters[-1]["xs"].append(x_val)
            clusters[-1]["rows"].add(row_val)
        else:
            clusters.append({"xs": [x_val], "rows": {row_val}})

    scored = [
        {
            "x": float(np.mean(cluster["xs"])),
            "n": len(cluster["xs"]),
            "row_count": len(cluster["rows"]),
        }
        for cluster in clusters
    ]
    scored.sort(key=lambda item: (-item["row_count"], -item["n"], item["x"]))
    needed = int(params["num_cols"]) - 1
    separators = sorted(
        [item["x"] for item in scored if item["row_count"] >= int(params["sep_min_cluster_members"])][:needed]
    )

    if len(separators) < needed:
        return fallback_separators(width, params)
    return separators


def detect_separators_in_row(row_img: np.ndarray, row_h: int, params: dict[str, Any]) -> np.ndarray:
    close_h = int(params["sep_vclose_h"])
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(5, close_h)))
    row_closed = cv2.morphologyEx(row_img, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    open_h = max(10, int(0.35 * row_h))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, open_h))
    return cv2.morphologyEx(row_closed, cv2.MORPH_OPEN, open_kernel, iterations=1)


def fallback_separators(width: int, params: dict[str, Any]) -> list[float]:
    left = int(float(params["left_margin_ratio"]) * width)
    right = int(width - float(params["right_margin_ratio"]) * width)
    return list(np.linspace(left, right, int(params["num_cols"]) + 1)[1:-1])


def crop_leads(
    ink: np.ndarray,
    baselines: list[int],
    separators: list[float],
    height: int,
    width: int,
    params: dict[str, Any],
) -> tuple[list[int], np.ndarray | None, dict[str, tuple[np.ndarray, tuple[int, int, int, int]]]]:
    baselines_3 = baselines[:3]
    vd = float(np.mean(np.diff(baselines_3))) if len(baselines_3) >= 2 else height / 4
    halfspan = int(float(params["crop_halfspan_ratio"]) * vd)
    rhythm_extra = int(float(params.get("rhythm_crop_extra_ratio", 0.08)) * vd)
    lr_pad = int(float(params["lr_pad_ratio"]) * width)

    left = int(max(0, separators[0] - (separators[1] - separators[0])))
    right_margin = int(width - float(params["right_margin_ratio"]) * width)
    right = right_margin
    if str(params.get("right_boundary_mode", "margin")) == "extrapolate" and len(separators) >= 2:
        right_extrapolated = int(separators[-1] + (separators[-1] - separators[-2]))
        right = min(right_margin, max(right_extrapolated, int(separators[-1])))
    bounds = sorted([left] + [int(x) for x in separators] + [right])

    lead_crops: dict[str, tuple[np.ndarray, tuple[int, int, int, int]]] = {}
    for row_idx in range(3):
        baseline = baselines_3[row_idx]
        y0 = int(max(0, baseline - halfspan))
        y1 = int(min(height - 1, baseline + halfspan))
        for col_idx in range(4):
            x0 = int(max(0, bounds[col_idx] + lr_pad))
            x1 = int(min(width - 1, bounds[col_idx + 1] - lr_pad))
            lead = LEAD_LAYOUT[row_idx][col_idx]
            lead_crops[lead] = (ink[y0:y1, x0:x1].copy(), (x0, y0, x1, y1))

    rhythm = None
    if len(baselines) >= 4:
        baseline = baselines[3]
        y0 = int(max(0, baseline - halfspan - rhythm_extra))
        y1 = int(min(height - 1, baseline + halfspan + rhythm_extra))
        rhythm = ink[y0:y1, left + lr_pad : right - lr_pad].copy()

    return bounds, rhythm, lead_crops


def build_trace_from_mask(mask: np.ndarray) -> np.ndarray | None:
    if mask.size == 0:
        return None

    _, width = mask.shape
    y_med = np.full(width, np.nan, dtype=np.float32)
    for x_coord in range(width):
        ys = np.where(mask[:, x_coord] > 0)[0]
        if ys.size:
            y_med[x_coord] = np.median(ys)

    if np.isnan(y_med).all():
        return None
    return y_med


def _find_edge_index(proj: np.ndarray, search_len: int, from_start: bool) -> int | None:
    if proj.size == 0:
        return None
    if from_start:
        region = proj[:search_len]
        offset = 0
    else:
        region = proj[-search_len:]
        offset = proj.size - search_len
    if region.size == 0:
        return None
    peak = float(region.max(initial=0.0))
    if peak <= 0.0:
        return None
    thresh = max(0.12, 0.50 * peak)
    idxs = np.where(region >= thresh)[0]
    if idxs.size == 0:
        return None
    return int(offset + (idxs[0] if from_start else idxs[-1]))


def detect_red_frame_top(image_bgr: np.ndarray) -> int | None:
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return None

    b, g, r = cv2.split(image_bgr)
    red_score = r.astype(np.int16) - np.maximum(g, b).astype(np.int16)
    red_mask = ((red_score >= 14) & (r >= 55)).astype(np.uint8) * 255
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(21, int(0.08 * w)), 1))
    mask_h = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, horiz_kernel)
    row_proj = (mask_h > 0).mean(axis=1).astype(np.float32)
    return _find_edge_index(row_proj, max(10, int(0.35 * h)), from_start=True)


def interpolate_short_gaps(trace: np.ndarray, max_gap: int = 24) -> np.ndarray:
    out = trace.copy()
    valid = ~np.isnan(out)
    n = len(out)
    x_coord = 0
    while x_coord < n:
        if valid[x_coord]:
            x_coord += 1
            continue

        start = x_coord
        while x_coord < n and not valid[x_coord]:
            x_coord += 1
        end = x_coord

        gap = end - start
        left = start - 1
        right = end
        if gap <= max_gap and left >= 0 and right < n and valid[left] and valid[right]:
            out[start:end] = np.interp(
                np.arange(start, end, dtype=np.float32),
                np.array([left, right], dtype=np.float32),
                np.array([out[left], out[right]], dtype=np.float32),
            )
    return out


def interpolate_short_gaps_with_mask(trace: np.ndarray | None, max_gap: int = 24) -> tuple[np.ndarray | None, np.ndarray | None]:
    if trace is None:
        return None, None

    out = trace.copy()
    filled = np.zeros(len(out), dtype=np.uint8)
    valid = ~np.isnan(out)
    n = len(out)
    x_coord = 0
    while x_coord < n:
        if valid[x_coord]:
            x_coord += 1
            continue

        start = x_coord
        while x_coord < n and not valid[x_coord]:
            x_coord += 1
        end = x_coord

        gap = end - start
        left = start - 1
        right = end
        if gap <= max_gap and left >= 0 and right < n and valid[left] and valid[right]:
            out[start:end] = np.interp(
                np.arange(start, end, dtype=np.float32),
                np.array([left, right], dtype=np.float32),
                np.array([out[left], out[right]], dtype=np.float32),
            )
            filled[start:end] = 1
    return out, filled


def interpolate_top_occluded_nan_gaps(trace: np.ndarray | None, y_thresh: int, crop_height: int, max_gap: int) -> np.ndarray | None:
    if trace is None or crop_height <= 0 or y_thresh <= 0:
        return trace

    out = trace.copy()
    band_margin = max(10.0, 0.08 * crop_height)
    valid = ~np.isnan(out)
    n = len(out)
    x_coord = 0
    while x_coord < n:
        if valid[x_coord]:
            x_coord += 1
            continue

        start = x_coord
        while x_coord < n and not valid[x_coord]:
            x_coord += 1
        end = x_coord

        gap = end - start
        left = start - 1
        right = end
        if gap > max_gap or left < 0 or right >= n or not valid[left] or not valid[right]:
            continue

        endpoints_near_top = (float(out[left]) <= y_thresh + band_margin) or (float(out[right]) <= y_thresh + band_margin)
        if not endpoints_near_top:
            continue

        out[start:end] = np.interp(
            np.arange(start, end, dtype=np.float32),
            np.array([left, right], dtype=np.float32),
            np.array([out[left], out[right]], dtype=np.float32),
        )
    return out


def interpolate_top_occluded_nan_gaps_with_mask(
    trace: np.ndarray | None,
    y_thresh: int,
    crop_height: int,
    max_gap: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if trace is None or crop_height <= 0 or y_thresh <= 0:
        return trace, None if trace is None else np.zeros(len(trace), dtype=np.uint8)

    out = trace.copy()
    filled = np.zeros(len(out), dtype=np.uint8)
    band_margin = max(10.0, 0.08 * crop_height)
    valid = ~np.isnan(out)
    n = len(out)
    x_coord = 0
    while x_coord < n:
        if valid[x_coord]:
            x_coord += 1
            continue

        start = x_coord
        while x_coord < n and not valid[x_coord]:
            x_coord += 1
        end = x_coord

        gap = end - start
        left = start - 1
        right = end
        if gap > max_gap or left < 0 or right >= n or not valid[left] or not valid[right]:
            continue

        endpoints_near_top = (float(out[left]) <= y_thresh + band_margin) or (float(out[right]) <= y_thresh + band_margin)
        if not endpoints_near_top:
            continue

        out[start:end] = np.interp(
            np.arange(start, end, dtype=np.float32),
            np.array([left, right], dtype=np.float32),
            np.array([out[left], out[right]], dtype=np.float32),
        )
        filled[start:end] = 1
    return out, filled


def interpolate_bottom_occluded_nan_gaps(trace: np.ndarray | None, crop_height: int, max_gap: int) -> np.ndarray | None:
    if trace is None or crop_height <= 0:
        return trace

    out = trace.copy()
    band_margin = max(10.0, 0.08 * crop_height)
    valid = ~np.isnan(out)
    n = len(out)
    x_coord = 0
    while x_coord < n:
        if valid[x_coord]:
            x_coord += 1
            continue

        start = x_coord
        while x_coord < n and not valid[x_coord]:
            x_coord += 1
        end = x_coord

        gap = end - start
        left = start - 1
        right = end
        if gap > max_gap or left < 0 or right >= n or not valid[left] or not valid[right]:
            continue

        bottom_band_y = (crop_height - 1) - band_margin
        endpoints_near_bottom = (float(out[left]) >= bottom_band_y) or (float(out[right]) >= bottom_band_y)
        if not endpoints_near_bottom:
            continue

        out[start:end] = np.interp(
            np.arange(start, end, dtype=np.float32),
            np.array([left, right], dtype=np.float32),
            np.array([out[left], out[right]], dtype=np.float32),
        )
    return out


def interpolate_bottom_occluded_nan_gaps_with_mask(
    trace: np.ndarray | None,
    crop_height: int,
    max_gap: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if trace is None or crop_height <= 0:
        return trace, None if trace is None else np.zeros(len(trace), dtype=np.uint8)

    out = trace.copy()
    filled = np.zeros(len(out), dtype=np.uint8)
    band_margin = max(10.0, 0.08 * crop_height)
    valid = ~np.isnan(out)
    n = len(out)
    x_coord = 0
    while x_coord < n:
        if valid[x_coord]:
            x_coord += 1
            continue

        start = x_coord
        while x_coord < n and not valid[x_coord]:
            x_coord += 1
        end = x_coord

        gap = end - start
        left = start - 1
        right = end
        if gap > max_gap or left < 0 or right >= n or not valid[left] or not valid[right]:
            continue

        bottom_band_y = (crop_height - 1) - band_margin
        endpoints_near_bottom = (float(out[left]) >= bottom_band_y) or (float(out[right]) >= bottom_band_y)
        if not endpoints_near_bottom:
            continue

        out[start:end] = np.interp(
            np.arange(start, end, dtype=np.float32),
            np.array([left, right], dtype=np.float32),
            np.array([out[left], out[right]], dtype=np.float32),
        )
        filled[start:end] = 1
    return out, filled


def count_vertical_clusters(
    ys: np.ndarray,
    gap_threshold: int = 3,
    min_cluster_pixels: int = 2,
) -> int:
    if len(ys) == 0:
        return 0

    clusters: list[list[int]] = []
    current = [int(ys[0])]
    for raw_y in ys[1:]:
        y_val = int(raw_y)
        if y_val - current[-1] > gap_threshold:
            clusters.append(current)
            current = [y_val]
        else:
            current.append(y_val)
    clusters.append(current)
    return sum(1 for cluster in clusters if len(cluster) >= min_cluster_pixels)


def detect_overlap_in_mask(
    selected_mask: np.ndarray,
    gap_threshold: int = 3,
    min_cluster_pixels: int = 2,
    overlap_ratio_threshold: float = 0.03,
    anomaly_column_threshold: int | None = None,
) -> dict[str, Any]:
    mask = selected_mask > 0
    _, width = mask.shape
    cluster_counts = np.zeros(width, dtype=np.int32)
    for x_coord in range(width):
        ys = np.where(mask[:, x_coord])[0]
        if ys.size > 0:
            cluster_counts[x_coord] = count_vertical_clusters(
                ys,
                gap_threshold=gap_threshold,
                min_cluster_pixels=min_cluster_pixels,
            )

    valid_cols = int(np.sum(np.any(mask, axis=0)))
    overlap_cols = int(np.sum(cluster_counts >= 2))
    overlap_ratio = overlap_cols / max(1, valid_cols)

    if anomaly_column_threshold is None:
        is_overlapping = overlap_ratio >= float(overlap_ratio_threshold)
        decision_mode = "ratio"
        decision_threshold: int | float = float(overlap_ratio_threshold)
    else:
        is_overlapping = overlap_cols > int(anomaly_column_threshold)
        decision_mode = "column_count"
        decision_threshold = int(anomaly_column_threshold)

    return {
        "is_overlapping": bool(is_overlapping),
        "overlap_ratio": float(overlap_ratio),
        "overlap_columns": overlap_cols,
        "total_columns": valid_cols,
        "cluster_counts": cluster_counts,
        "decision_mode": decision_mode,
        "decision_threshold": decision_threshold,
    }


def vertical_close_crop(crop_bin: np.ndarray, close_h: int) -> np.ndarray:
    crop = (crop_bin > 0).astype(np.uint8) * 255
    close_h = max(1, int(close_h))
    if close_h <= 1:
        return crop.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_h))
    return cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel, iterations=1)


def detect_overlap_with_vertical_close(
    crop_bin: np.ndarray,
    lead: str | None = None,
    params: dict[str, Any] | None = None,
    crop_y0: int = 0,
    frame_top_y: int | None = None,
    overlap_ratio_threshold: float = 0.03,
    anomaly_column_threshold: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_params = DEFAULT_PARAMS if params is None else params
    close_h = int(active_params.get("overlap_vertical_close_h", 7))
    if anomaly_column_threshold is None:
        anomaly_column_threshold = int(active_params.get("overlap_column_count_threshold", 3))

    if crop_bin.size == 0:
        # A degenerate (zero-size) crop -- e.g. an extremely tight row spacing collapsed
        # this lead's box to zero height/width. Nothing to analyze; cv2.morphologyEx
        # would otherwise crash on empty input downstream in vertical_close_crop.
        info = {
            "is_overlapping": False,
            "overlap_ratio": 0.0,
            "overlap_columns": 0,
            "total_columns": 0,
            "cluster_counts": np.zeros(crop_bin.shape[1] if crop_bin.ndim >= 2 else 0, dtype=np.int32),
            "decision_mode": "column_count" if anomaly_column_threshold is not None else "ratio",
            "decision_threshold": anomaly_column_threshold if anomaly_column_threshold is not None else overlap_ratio_threshold,
        }
        debug = {"close_h": close_h, "crop": crop_bin, "mask": crop_bin, "trace": None, "extract_debug": {}}
        return info, debug

    closed_crop = vertical_close_crop(crop_bin, close_h)
    closed_trace, closed_debug = extract_trace(
        closed_crop,
        dilate_k=int(active_params["dilate_k"]),
        lead=lead,
        params=active_params,
        crop_y0=crop_y0,
        frame_top_y=frame_top_y,
    )
    closed_mask = closed_debug["mask"]
    info = detect_overlap_in_mask(
        closed_mask,
        gap_threshold=3,
        min_cluster_pixels=2,
        overlap_ratio_threshold=overlap_ratio_threshold,
        anomaly_column_threshold=anomaly_column_threshold,
    )
    debug = {
        "close_h": close_h,
        "crop": closed_crop,
        "mask": closed_mask,
        "trace": closed_trace,
        "extract_debug": closed_debug,
    }
    return info, debug


def extract_trace(
    crop_bin: np.ndarray,
    dilate_k: int = 9,
    lead: str | None = None,
    params: dict[str, Any] | None = None,
    crop_y0: int = 0,
    frame_top_y: int | None = None,
) -> tuple[np.ndarray | None, dict[str, np.ndarray | bool | int | float | str]]:
    crop = (crop_bin > 0).astype(np.uint8) * 255

    kernel_width = max(3, int(dilate_k))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    dilated = cv2.dilate(crop, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if num <= 1:
        return None, {
            "crop": crop,
            "dil": dilated,
            "mask": np.zeros_like(crop),
            "interpolation_mask": np.zeros(crop.shape[1], dtype=np.uint8),
            "selected_method": "none",
            "fallback_used": False,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    standard_mask = (labels == best).astype(np.uint8) * 255
    standard_trace = build_trace_from_mask(standard_mask)
    standard_debug = {
        "crop": crop,
        "dil": dilated,
        "mask": standard_mask,
        "interpolation_mask": np.zeros(crop.shape[1], dtype=np.uint8),
        "selected_method": "largest_cc",
        "fallback_used": False,
    }

    if params and bool(params.get("rhythm_bottom_cc_enable", True)) and lead is None:
        y_thresh = int(
            np.clip(
                crop.shape[0] - 1 - int(params.get("rhythm_bottom_cc_y_thresh_px", params.get("top_cc_y_thresh_px", 45))),
                0,
                max(0, crop.shape[0] - 1),
            )
        )
        min_width = int(params.get("top_cc_min_width_px", 25))
        min_coverage_ratio = float(params.get("top_cc_min_coverage_ratio", 0.25))

        keep_mask = np.zeros_like(crop)
        kept = 0
        for component in range(1, num):
            comp_mask = labels == component
            ys, xs = np.where(comp_mask)
            if xs.size == 0:
                continue

            width = int(np.unique(xs).size)
            max_y = int(ys.max())
            if max_y >= y_thresh and width >= min_width:
                keep_mask[comp_mask] = 255
                kept += 1

        if kept == 0:
            standard_debug["fallback_used"] = True
            standard_debug["y_thresh"] = y_thresh
            return standard_trace, standard_debug

        selected_cols = np.any(keep_mask > 0, axis=0)
        coverage_ratio = float(selected_cols.mean()) if selected_cols.size else 0.0
        if coverage_ratio < min_coverage_ratio:
            standard_debug["fallback_used"] = True
            standard_debug["y_thresh"] = y_thresh
            standard_debug["coverage_ratio"] = coverage_ratio
            return standard_trace, standard_debug

        bottom_trace = build_trace_from_mask(keep_mask)
        if bottom_trace is None:
            standard_debug["fallback_used"] = True
            standard_debug["y_thresh"] = y_thresh
            standard_debug["coverage_ratio"] = coverage_ratio
            return standard_trace, standard_debug

        _, width = crop.shape
        bottom_trace, bottom_interp_mask = interpolate_bottom_occluded_nan_gaps_with_mask(
            bottom_trace,
            crop_height=crop.shape[0],
            max_gap=max(10, int(0.06 * width)),
        )
        if bottom_trace is None:
            standard_debug["fallback_used"] = True
            standard_debug["y_thresh"] = y_thresh
            standard_debug["coverage_ratio"] = coverage_ratio
            return standard_trace, standard_debug
        bottom_trace, short_gap_interp_mask = interpolate_short_gaps_with_mask(
            bottom_trace,
            max_gap=max(8, int(0.03 * width)),
        )
        interpolation_mask = (bottom_interp_mask | short_gap_interp_mask).astype(np.uint8)

        return bottom_trace, {
            "crop": crop,
            "dil": dilated,
            "mask": keep_mask,
            "interpolation_mask": interpolation_mask,
            "selected_method": "bottom_threshold",
            "fallback_used": False,
            "y_thresh": y_thresh,
            "coverage_ratio": coverage_ratio,
        }

    if not params or not bool(params.get("top_cc_enable", True)) or lead not in TOP_ROW_LEADS:
        return standard_trace, standard_debug

    y_anchor_local = None if frame_top_y is None else int(frame_top_y - crop_y0)
    if y_anchor_local is not None:
        y_thresh = int(
            np.clip(
                y_anchor_local + int(params.get("top_cc_y_anchor_offset_px", 3)),
                1,
                max(1, crop.shape[0] - 1),
            )
        )
    else:
        y_thresh = int(params.get("top_cc_y_thresh_px", 45))
    min_width = int(params.get("top_cc_min_width_px", 25))
    min_coverage_ratio = float(params.get("top_cc_min_coverage_ratio", 0.25))

    keep_mask = np.zeros_like(crop)
    kept = 0
    for component in range(1, num):
        comp_mask = labels == component
        ys, xs = np.where(comp_mask)
        if xs.size == 0:
            continue

        width = int(np.unique(xs).size)
        min_y = int(ys.min())
        if min_y <= y_thresh and width >= min_width:
            keep_mask[comp_mask] = 255
            kept += 1

    if kept == 0:
        standard_debug["fallback_used"] = True
        return standard_trace, standard_debug

    selected_cols = np.any(keep_mask > 0, axis=0)
    coverage_ratio = float(selected_cols.mean()) if selected_cols.size else 0.0
    if coverage_ratio < min_coverage_ratio:
        standard_debug["fallback_used"] = True
        return standard_trace, standard_debug

    top_trace = build_trace_from_mask(keep_mask)
    if top_trace is None:
        standard_debug["fallback_used"] = True
        return standard_trace, standard_debug

    _, width = crop.shape
    top_trace, top_interp_mask = interpolate_top_occluded_nan_gaps_with_mask(
        top_trace,
        y_thresh=y_thresh,
        crop_height=crop.shape[0],
        max_gap=max(10, int(0.06 * width)),
    )
    if top_trace is None:
        standard_debug["fallback_used"] = True
        return standard_trace, standard_debug
    top_trace, short_gap_interp_mask = interpolate_short_gaps_with_mask(
        top_trace,
        max_gap=max(8, int(0.03 * width)),
    )
    interpolation_mask = (top_interp_mask | short_gap_interp_mask).astype(np.uint8)

    return top_trace, {
        "crop": crop,
        "dil": dilated,
        "mask": keep_mask,
        "interpolation_mask": interpolation_mask,
        "selected_method": "top_threshold",
        "fallback_used": False,
        "y_thresh": y_thresh,
        "coverage_ratio": coverage_ratio,
    }


def attach_overlap_annotations(
    lead_results: dict[str, LeadResult],
    rhythm_strip: np.ndarray | None,
    params: dict[str, Any],
    frame_top_y: int | None = None,
) -> tuple[
    dict[str, LeadResult],
    dict[str, Any] | None,
    dict[str, Any] | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    annotated: dict[str, LeadResult] = {}
    for lead, lead_result in lead_results.items():
        legacy_overlap_result = detect_overlap_in_mask(
            lead_result.connected_mask,
            overlap_ratio_threshold=float(params["overlap_ratio_threshold"]),
        )
        overlap_result, overlap_debug = detect_overlap_with_vertical_close(
            lead_result.crop_binary,
            lead=lead,
            params=params,
            crop_y0=int(lead_result.crop_box[1]),
            frame_top_y=frame_top_y,
            overlap_ratio_threshold=float(params["overlap_ratio_threshold"]),
            anomaly_column_threshold=int(params["overlap_column_count_threshold"]),
        )
        overlap_mask = overlap_debug["mask"].astype(np.uint8)
        overlap_trace_view = build_processed_trace_view(overlap_mask, overlap_debug["trace"])

        annotated[lead] = LeadResult(
            lead=lead_result.lead,
            crop_binary=lead_result.crop_binary,
            crop_box=lead_result.crop_box,
            connected_mask=lead_result.connected_mask,
            processed_trace_view_bgr=lead_result.processed_trace_view_bgr,
            y_trace=lead_result.y_trace,
            t_ms=lead_result.t_ms,
            v_mV=lead_result.v_mV,
            interpolation_mask=lead_result.interpolation_mask,
            overlap_result=overlap_result,
            legacy_overlap_result=legacy_overlap_result,
            overlap_mask=overlap_mask,
            overlap_processed_trace_view_bgr=overlap_trace_view,
        )

    legacy_rhythm_overlap_result = None
    rhythm_overlap_result = None
    rhythm_interpolation_mask = None
    rhythm_overlap_mask = None
    rhythm_overlap_trace_view_bgr = None
    if rhythm_strip is not None:
        rhythm_trace, rhythm_debug = extract_trace(
            rhythm_strip,
            dilate_k=int(params["dilate_k"]),
            lead=None,
            params=params,
        )
        rhythm_mask = rhythm_debug["mask"]
        rhythm_interpolation_mask = rhythm_debug.get("interpolation_mask")
        if rhythm_mask.size > 0:
            legacy_rhythm_overlap_result = detect_overlap_in_mask(
                rhythm_mask,
                overlap_ratio_threshold=float(params["rhythm_overlap_ratio_threshold"]),
            )
        rhythm_overlap_result, rhythm_overlap_debug = detect_overlap_with_vertical_close(
            rhythm_strip,
            lead=None,
            params=params,
            frame_top_y=frame_top_y,
            overlap_ratio_threshold=float(params["rhythm_overlap_ratio_threshold"]),
            anomaly_column_threshold=int(params["overlap_column_count_threshold"]),
        )
        rhythm_overlap_mask = rhythm_overlap_debug["mask"].astype(np.uint8)
        rhythm_overlap_trace_view_bgr = build_processed_trace_view(
            rhythm_overlap_mask,
            rhythm_overlap_debug["trace"],
        )

    return (
        annotated,
        rhythm_overlap_result,
        legacy_rhythm_overlap_result,
        rhythm_interpolation_mask,
        rhythm_overlap_mask,
        rhythm_overlap_trace_view_bgr,
    )


def build_processed_trace_view(connected_mask: np.ndarray, y_trace: np.ndarray | None) -> np.ndarray:
    if connected_mask.size == 0:
        # A degenerate (zero-size) crop -- e.g. an extremely tight row spacing collapsed
        # this lead's box to zero height/width. Nothing to render; cv2.cvtColor would
        # otherwise crash on empty input.
        h, w = connected_mask.shape[:2] if connected_mask.ndim >= 2 else (0, 0)
        return np.zeros((h, w, 3), dtype=np.uint8)
    overlay = cv2.cvtColor((connected_mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    if y_trace is None:
        return overlay

    for x_coord, y_coord in enumerate(y_trace):
        if np.isnan(y_coord):
            continue
        cv2.circle(overlay, (x_coord, int(y_coord)), 1, (0, 255, 0), -1)
    return overlay


def build_highlighted_trace_view(
    base_bgr: np.ndarray,
    y_trace: np.ndarray | None,
    active_cols: np.ndarray | None,
    *,
    line_color: tuple[int, int, int],
    point_color: tuple[int, int, int] | None = None,
    alpha: float = 0.42,
) -> np.ndarray:
    if active_cols is None:
        return base_bgr

    cols = np.flatnonzero(np.asarray(active_cols) > 0)
    if cols.size == 0:
        return base_bgr

    overlay = base_bgr.copy()
    height = overlay.shape[0]
    band_color = tuple(int(channel) for channel in line_color)
    for x_coord in cols:
        cv2.line(overlay, (int(x_coord), 0), (int(x_coord), height - 1), band_color, 2)

    highlighted = cv2.addWeighted(overlay, alpha, base_bgr, 1.0 - alpha, 0.0)
    marker_color = point_color if point_color is not None else line_color
    if y_trace is not None:
        for x_coord in cols:
            if 0 <= x_coord < len(y_trace) and not np.isnan(y_trace[x_coord]):
                cv2.circle(highlighted, (int(x_coord), int(y_trace[x_coord])), 3, marker_color, -1)
    return highlighted


def build_overlap_trace_view(
    connected_mask: np.ndarray,
    y_trace: np.ndarray | None,
    cluster_counts: np.ndarray | None,
) -> np.ndarray:
    base = build_processed_trace_view(connected_mask, y_trace)
    active_cols = None if cluster_counts is None else (np.asarray(cluster_counts) >= 2).astype(np.uint8)
    return build_highlighted_trace_view(
        base,
        y_trace,
        active_cols,
        line_color=(32, 64, 255),
        point_color=(32, 64, 255),
        alpha=0.48,
    )


def build_interpolation_trace_view(
    connected_mask: np.ndarray,
    y_trace: np.ndarray | None,
    interpolation_mask: np.ndarray | None,
) -> np.ndarray:
    base = build_processed_trace_view(connected_mask, y_trace)
    return build_highlighted_trace_view(
        base,
        y_trace,
        interpolation_mask,
        line_color=(0, 170, 255),
        point_color=(0, 170, 255),
        alpha=0.42,
    )


def _build_figure_axes(
    title: str,
    *,
    width: float = 13.5,
    height: float = 10.0,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, height_ratios=[1.0, 1.0, 1.0, 1.15])
    axes: dict[str, plt.Axes] = {}
    for idx, lead in enumerate(LEAD_ORDER):
        row, col = divmod(idx, 4)
        axes[lead] = fig.add_subplot(grid[row, col])
    axes["__rhythm__"] = fig.add_subplot(grid[3, :])
    fig.suptitle(title)
    return fig, axes


def _get_rhythm_trace_debug(result: DigitizationResult) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if result.rhythm_strip is None:
        return None, None
    rhythm_trace, rhythm_debug = extract_trace(
        result.rhythm_strip,
        dilate_k=int(DEFAULT_PARAMS["dilate_k"]),
        lead=None,
        params=DEFAULT_PARAMS,
    )
    return rhythm_trace, rhythm_debug


def build_trace_figure(result: DigitizationResult) -> plt.Figure:
    fig, axes = _build_figure_axes(f"{result.image_path.name} y-trace overlays", width=13.0, height=9.8)
    for lead in LEAD_ORDER:
        ax = axes[lead]
        lead_result = result.lead_results.get(lead)
        ax.set_title(lead)
        ax.axis("off")
        if lead_result is not None:
            ax.imshow(bgr_to_rgb(lead_result.processed_trace_view_bgr), aspect="auto")
        else:
            ax.text(0.5, 0.5, "No trace", ha="center", va="center", transform=ax.transAxes)

    rhythm_ax = axes["__rhythm__"]
    rhythm_ax.set_title("II (Rhythm Strip)")
    rhythm_ax.axis("off")
    rhythm_trace, rhythm_debug = _get_rhythm_trace_debug(result)
    if rhythm_debug is not None:
        rhythm_ax.imshow(bgr_to_rgb(build_processed_trace_view(rhythm_debug["mask"], rhythm_trace)), aspect="auto")
    else:
        rhythm_ax.text(0.5, 0.5, "No rhythm strip", ha="center", va="center", transform=rhythm_ax.transAxes)
    return fig


def build_crop_figure(result: DigitizationResult) -> plt.Figure:
    fig, axes = _build_figure_axes(f"{result.image_path.name} lead crops", width=13.0, height=9.4)
    for lead in LEAD_ORDER:
        ax = axes[lead]
        lead_result = result.lead_results.get(lead)
        ax.set_title(lead)
        ax.axis("off")
        if lead_result is not None:
            ax.imshow(lead_result.crop_binary, cmap="gray", vmin=0, vmax=255)
        else:
            ax.text(0.5, 0.5, "No crop", ha="center", va="center", transform=ax.transAxes)

    rhythm_ax = axes["__rhythm__"]
    rhythm_ax.set_title("II (Rhythm Strip Crop)")
    rhythm_ax.axis("off")
    if result.rhythm_strip is not None:
        rhythm_ax.imshow(result.rhythm_strip, cmap="gray", vmin=0, vmax=255, aspect="auto")
    else:
        rhythm_ax.text(0.5, 0.5, "No rhythm strip", ha="center", va="center", transform=rhythm_ax.transAxes)
    return fig


def build_overlap_figure(result: DigitizationResult) -> plt.Figure:
    fig, axes = _build_figure_axes(f"{result.image_path.name} overlap views", width=13.0, height=9.8)
    for lead in LEAD_ORDER:
        ax = axes[lead]
        lead_result = result.lead_results.get(lead)
        ax.axis("off")
        if lead_result is None:
            ax.set_title(lead)
            ax.text(0.5, 0.5, "No lead result", ha="center", va="center", transform=ax.transAxes)
            continue

        overlap_info = lead_result.overlap_result or {}
        overlap_cols = int(overlap_info.get("overlap_columns", 0))
        status = "OVERLAP" if overlap_info.get("is_overlapping", False) else "clean"
        ax.set_title(f"{lead}: {status} ({overlap_cols} cols)")
        image = build_overlap_trace_view(
            lead_result.overlap_mask if lead_result.overlap_mask is not None else lead_result.connected_mask,
            lead_result.y_trace,
            None if lead_result.overlap_result is None else lead_result.overlap_result.get("cluster_counts"),
        )
        ax.imshow(bgr_to_rgb(image), aspect="auto")

    rhythm_ax = axes["__rhythm__"]
    rhythm_ax.axis("off")
    rhythm_info = result.rhythm_overlap_result or {}
    rhythm_cols = int(rhythm_info.get("overlap_columns", 0))
    rhythm_status = "OVERLAP" if rhythm_info.get("is_overlapping", False) else "clean"
    rhythm_ax.set_title(f"II (Rhythm Strip): {rhythm_status} ({rhythm_cols} cols)")
    rhythm_trace, rhythm_debug = _get_rhythm_trace_debug(result)
    if rhythm_debug is not None:
        rhythm_ax.imshow(
            bgr_to_rgb(
                build_overlap_trace_view(
                    result.rhythm_overlap_mask if result.rhythm_overlap_mask is not None else rhythm_debug["mask"],
                    rhythm_trace,
                    None if result.rhythm_overlap_result is None else result.rhythm_overlap_result.get("cluster_counts"),
                )
            ),
            aspect="auto",
        )
    else:
        rhythm_ax.text(0.5, 0.5, "No rhythm strip", ha="center", va="center", transform=rhythm_ax.transAxes)
    return fig


def build_interpolation_figure(result: DigitizationResult) -> plt.Figure:
    fig, axes = _build_figure_axes(f"{result.image_path.name} interpolation views", width=13.0, height=9.8)
    for lead in LEAD_ORDER:
        ax = axes[lead]
        lead_result = result.lead_results.get(lead)
        ax.axis("off")
        if lead_result is None:
            ax.set_title(lead)
            ax.text(0.5, 0.5, "No lead result", ha="center", va="center", transform=ax.transAxes)
            continue

        interp_cols = 0 if lead_result.interpolation_mask is None else int(np.sum(lead_result.interpolation_mask > 0))
        ax.set_title(f"{lead}: interpolated {interp_cols} cols")
        image = build_interpolation_trace_view(
            lead_result.connected_mask,
            lead_result.y_trace,
            lead_result.interpolation_mask,
        )
        ax.imshow(bgr_to_rgb(image), aspect="auto")

    rhythm_ax = axes["__rhythm__"]
    rhythm_ax.axis("off")
    rhythm_interp_cols = 0 if result.rhythm_interpolation_mask is None else int(np.sum(result.rhythm_interpolation_mask > 0))
    rhythm_ax.set_title(f"II (Rhythm Strip): interpolated {rhythm_interp_cols} cols")
    rhythm_trace, rhythm_debug = _get_rhythm_trace_debug(result)
    if rhythm_debug is not None:
        rhythm_ax.imshow(
            bgr_to_rgb(
                build_interpolation_trace_view(
                    rhythm_debug["mask"],
                    rhythm_trace,
                    result.rhythm_interpolation_mask,
                )
            ),
            aspect="auto",
        )
    else:
        rhythm_ax.text(0.5, 0.5, "No rhythm strip", ha="center", va="center", transform=rhythm_ax.transAxes)
    return fig


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
