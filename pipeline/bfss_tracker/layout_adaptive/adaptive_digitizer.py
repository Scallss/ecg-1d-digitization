"""Digitization of a sheet whose layout is not 3x4 + rhythm strip (layout_method="adaptive"): one crop per slot of the planned layout, trace from the beam tracker.

Differences to the fixed 3x4 path (ecg_digitizer.digitize_ecg_image), all deliberate:
* crops come from the plan (row baselines, column edges), not from LEAD_LAYOUT; no iterative crop recovery, no top-of-page occlusion handling (the frame-top logic is 3x4 specific)
* time scale from the paper: 25 mm/s and 10 mm/mV with the sheet's px/mm (the fixed path divides 10 s by the rhythm strip width, which does not exist in most layouts)
* the trace is always the overlap tracker's (trace_method is ignored)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class SlotResult:
    name: str
    sign: int
    kind: str
    row: int
    col: int
    ncols: int
    shifted: bool
    crop_box: tuple
    crop_binary: np.ndarray
    y_trace: np.ndarray | None
    interpolation_mask: np.ndarray
    overlap_result: dict | None = None


@dataclass
class AdaptiveResult:
    image_path: Any
    raw_image_bgr: np.ndarray
    ink_mask: np.ndarray
    plan: dict
    slots: list
    time_res_ms_per_px: float
    volt_res_mV_per_px: float
    paper_x0: int


def digitize_adaptive(img, ink, plan, params, ed, image_path=None):
    H, W = ink.shape
    ppm = plan["px_per_mm"]
    rows = plan["rows"]
    pitch = float(np.median(np.diff(rows))) if len(rows) > 1 else H / 2
    hs = int(float(params.get("adaptive_crop_halfspan_ratio") or params["crop_halfspan_ratio"]) * pitch)
    hs_r = hs + int(float(params.get("rhythm_crop_extra_ratio", 0.08)) * pitch)
    lr_pad = int(float(params["lr_pad_ratio"]) * W)
    slots = []
    for s in plan["slots"]:
        b = int(s["baseline"]); h = hs_r if s["kind"] == "rhythm" else hs
        y0, y1 = int(max(0, b - h)), int(min(H - 1, b + h))
        x0, x1 = int(max(0, s["x_edges"][0] + lr_pad)), int(min(W - 1, s["x_edges"][1] - lr_pad))
        crop = ink[y0:y1, x0:x1].copy()
        if crop.size == 0:
            y, im = None, np.zeros(max(1, x1 - x0), dtype=np.uint8)
        else:
            y, im = ed.track_overlap_trace(crop)
        ov = None
        if crop.size:
            try:
                ov, _ = ed.detect_overlap_with_vertical_close(crop, lead=None, params=params, crop_y0=y0, frame_top_y=None,
                                                              overlap_ratio_threshold=float(params["overlap_ratio_threshold"]),
                                                              anomaly_column_threshold=int(params["overlap_column_count_threshold"]))
            except Exception:
                ov = None
        slots.append(SlotResult(s["name"], s["sign"], s["kind"], s["row"], s["col"], s["ncols"], s["shifted"], (x0, y0, x1, y1), crop, y, im, ov))
    ppm_t = plan.get("px_per_mm_fine", ppm)     # refined grid period for the scales; the coarse one drives detection
    time_res = 1000.0 / (25.0 * ppm_t)         # ms per px at 25 mm/s
    volt_res = 0.1 / ppm_t                      # mV per px at 10 mm/mV
    return AdaptiveResult(image_path, img, ink, plan, slots, time_res, volt_res, lr_pad)


def build_adaptive_export(res: AdaptiveResult, params: dict, lead_order: list):
    """Canonical 10 s x 12 export (uV, dt_ms grid) like extract_all_npz.build_canonical_export. One signal per lead: a rhythm-strip slot wins over a cell of the same lead.
    A slot printed with a minus sign (Cabrera -aVR) is stored under its lead name with the sign undone. Time: crop x position minus the first crop's x (same convention as the fixed path)."""
    dt = float(params["interp_dt_ms"]); n = int(float(params["time_seconds_total"]) * 1000.0 / dt)
    t_can = np.arange(n, dtype=np.float32) * dt
    chosen: dict[str, tuple] = {}
    for sl in res.slots:
        if sl.y_trace is None: continue
        valid = ~np.isnan(sl.y_trace)
        if valid.sum() < 20: continue
        rank = (1 if sl.kind == "rhythm" else 0, int(valid.sum()))
        if sl.name not in chosen or rank > chosen[sl.name][0]: chosen[sl.name] = (rank, sl)
    sig, ov_cols, ip_cols = {}, {}, {}

    def paint(active, t_centers):
        m = np.zeros(n, dtype=np.uint8)
        if active is None: return m
        half = 0.5 * res.time_res_ms_per_px
        for x in np.where(active > 0)[0]:
            i0 = max(0, int(np.floor(max(0.0, t_centers[x] - half) / dt))); i1 = min(n, int(np.ceil(min(float(t_can[-1] + dt), t_centers[x] + half) / dt)))
            m[i0:i1] = 1
        return m

    for name, (_, sl) in chosen.items():
        y = sl.y_trace; valid = ~np.isnan(y)
        amp = (np.nanmedian(y[valid]) - y) * res.volt_res_mV_per_px * sl.sign
        t0px = (sl.crop_box[0] - res.paper_x0) if sl.shifted else 0
        t_loc = (t0px + np.arange(len(amp), dtype=np.float32)) * res.time_res_ms_per_px
        v = np.interp(t_can, t_loc[valid], amp[valid], left=np.nan, right=np.nan) * 1000.0
        sig[name] = v
        cc = None if sl.overlap_result is None else sl.overlap_result.get("cluster_counts")
        isov = bool(sl.overlap_result is not None and sl.overlap_result.get("is_overlapping", False))
        ov = paint((cc >= 2).astype(np.uint8) if (cc is not None and isov) else None, t_loc)
        ip = paint(sl.interpolation_mask, t_loc)
        ok = (~np.isnan(v)).astype(np.uint8)
        ov_cols[name] = ov & ok; ip_cols[name] = ip & ok
    nan = np.full(n, np.nan, dtype=np.float32); z = np.zeros(n, dtype=np.uint8)
    matrix = np.column_stack([sig.get(l, nan) for l in lead_order])
    overlap = np.column_stack([ov_cols.get(l, z) for l in lead_order]).astype(np.uint8)
    interp = np.column_stack([ip_cols.get(l, z) for l in lead_order]).astype(np.uint8)
    meta = {"layout_method": "adaptive", "layout": res.plan["layout"], "order_status": res.plan["order_status"], "time_mode": "assumed_shifted",
            "px_per_mm": res.plan["px_per_mm"], "starts_source": res.plan.get("starts_source"), "column_status": res.plan.get("column_status"),
            "time_scale": "paper 25 mm/s, 10 mm/mV from the sheet's px/mm", "paper_x0_px": res.paper_x0,
            "slots": [{"name": s["name"], "sign": s["sign"], "kind": s["kind"], "row": s["row"], "col": s["col"], "ncols": s["ncols"], "x_edges": s["x_edges"], "baseline": s["baseline"]} for s in res.plan["slots"]],
            "leads_missing": [l for l in lead_order if l not in sig], "overlap_mask_note": "overlap flags per slot as in the fixed path (lead=None)"}
    return matrix, overlap, interp, t_can, meta
