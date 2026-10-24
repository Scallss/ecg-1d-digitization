"""Column boundaries: printed ticks first, lead labels as the fallback. Own code.

    out = column_starts(img_bgr)  -> {"source": "ticks"|"labels"|"none", "starts_px": [x of every column start but the first], "ticks": [...], "layout": catalogue name or None, ...}

Ticks (printed dashes at the column starts) are detected by cross-row stability: the same short vertical ink at the same x in most rows, with the same amount of ink in each,
which QRS strokes (different in every row) do not give. A tick set is trusted only when all start positions of an equal 2-, 3- or 4-column grid are among the detections (largest grid first).
When it is not (no ticks printed, only some of them found), the labels give the layout (via layout_infer.infer2). fallback="label_edge": each start is the left edge of that column's labels minus margin_px pixel columns, so the neighbouring crop ends just before the label; a column without a found label uses the equal grid up to the end of the trace area (content_end). fallback="equal_grid": equal grid only. fallback="grid_capped" (default, decision 2026-10-12): the equal grid, but a start is never placed right of the label edge (so a label is never cut).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import layout_detect as D, label_detect as L, layout_infer as I
import ecg_digitizer as PIPE      # pipeline/bfss_tracker (put on sys.path by layout_detect): the updated stability_repair separator detector (2026-10-13)


def detect_ticks(ink: np.ndarray, rows, ppm: float, iou_min: float = 0.75):
    """Candidates: columns with a similar amount of ink in at least half of the rows. Confirmed when the vertical ink pattern around each baseline (the printed dashes) is common
    to those rows (the ink shared by all of them is at least iou_min of the smallest row's ink); QRS strokes differ from row to row."""
    H, W = ink.shape; R = len(rows)
    if R < 2: return []
    pitch = float(np.median(np.diff(rows))); half = int(0.45 * pitch); span = int(7.0 * ppm)
    counts = np.stack([ink[max(0, int(b) - half): int(b) + half + 1, :].sum(0) for b in rows]).astype(np.float32)
    lo, hi = int(0.05 * W), int(0.95 * W); hits = []
    for x in range(lo, hi):
        c = counts[:, x]; ok = c >= 3.0 * ppm
        if ok.sum() < max(2, 0.5 * R): continue
        v = c[ok]; med = np.median(v)
        if (v.max() - v.min()) > 0.6 * med + 2: continue
        profs = []
        for r in np.where(ok)[0]:
            b = int(rows[r]); seg = ink[max(0, b - span): b + span + 1, max(0, x - 1): x + 2].max(1)
            pr = np.zeros(2 * span + 1, bool); pr[: len(seg)] = seg > 0; profs.append(pr)
        common = np.logical_and.reduce(profs)                       # the printed dashes are in every row; the trace crossing the tick adds ink to a row but does not remove the dashes
        up = common[: span - int(1.5 * ppm)].sum(); dn = common[span + int(1.5 * ppm):].sum()      # dashes above and below the baseline (a label sits above only)
        if common.sum() >= iou_min * min(pr.sum() for pr in profs) and up >= 2.0 * ppm and dn >= 2.0 * ppm: hits.append(x)
    groups = []
    for x in hits:
        if groups and x - groups[-1][-1] <= 3: groups[-1].append(x)
        else: groups.append([x])
    return [float(np.mean(g)) for g in groups]


def content_end(ink: np.ndarray, rows, ppm: float):
    """x where the trace area ends: the leftmost stable vertical structure in the right-hand 7% of the page (the end-of-last-column tick on our scans, otherwise the left edge of the
    calibration pulse), found like the ticks: ink of similar amount in at least half of the rows."""
    H, W = ink.shape; R = len(rows)
    if R < 2: return None
    pitch = float(np.median(np.diff(rows))); half = int(0.45 * pitch)
    counts = np.stack([ink[max(0, int(b) - half): int(b) + half + 1, :].sum(0) for b in rows]).astype(np.float32)
    for x in range(int(0.93 * W), int(0.995 * W)):
        c = counts[:, x]; ok = c >= 3.0 * ppm
        if ok.sum() >= max(2, 0.5 * R):
            v = c[ok]
            if (v.max() - v.min()) <= 0.6 * np.median(v) + 2: return float(x)
    return None


def match_grid(ticks, W, ppm, cal_mm=6.0, tol=0.03):
    """Largest column grid (2-4 columns of the trace area W - calibration pulse) whose start positions are all among the detected ticks; extra detections are ignored.
    Returns the matched tick x list or None."""
    content = W - cal_mm * ppm
    for ncols in (4, 3, 2):
        exp = [(k + 1) * content / ncols for k in range(ncols - 1)]
        found = [min(ticks, key=lambda t: abs(t - e)) if ticks else None for e in exp]
        if all(f is not None and abs(f - e) <= tol * content for f, e in zip(found, exp)): return [float(f) for f in found]
    return None


def pipeline_separators(img_bgr, rows, ncols, ink_w):
    """The pipeline's separator detector (separator_method="stability_repair", preset black165_stable_sep_repair_tracker) with its status; needs >= 3 rows and 2-4 columns.
    Returns (separators, status) with status in ok / repaired / anchored / irregular_kept / fallback, or (None, 'unavailable')."""
    if len(rows) < 3 or not 2 <= ncols <= 4: return None, "unavailable"
    params = PIPE.get_params("black165_stable_sep_repair_tracker"); params["num_cols"] = ncols
    H, W = img_bgr.shape[:2]
    try: seps, status = PIPE.detect_column_separators_stability_repair_with_status(img_bgr, [int(r) for r in rows[:3]] + [int(r) for r in rows[3:]], H, W, params)
    except Exception: return None, "unavailable"
    return [float(x) for x in seps], status


def column_starts(img_bgr: np.ndarray, use_ticks: bool = True, fallback: str = "grid_capped", margin_px: int = 3, separator: str = "pipeline", verify: bool = True):
    ink = D.ink_mask(img_bgr); ppm = L.estimate_px_per_mm(img_bgr); H, W = ink.shape
    rows, _ = D.detect_rows(ink)
    lab = I.infer2(img_bgr)
    ncols_lab = max(len(r) for r in I.CAT[lab["candidate"]].pages[0]) if lab.get("candidate") else 0
    if separator == "pipeline" and use_ticks:
        seps, status = pipeline_separators(img_bgr, rows, ncols_lab, W)
        # trusted only when the detector itself says its result is regular (ok) or was repaired to a regular set, the set is an equal grid over the trace area, and (verify=True) each separator is also a printed tick by the dash-pattern test (above and below the baseline in the same way in every row): the pipeline detector alone accepted stable false columns on 2 of 72 synthetic sheets without printed ticks
        if seps is not None and status in ("ok", "repaired") and match_grid(seps, W, ppm, tol=0.03) is not None and len(seps) == ncols_lab - 1 \
                and (not verify or all(min((abs(t - x) for t in detect_ticks(ink, rows, ppm)), default=1e9) <= 1.5 * ppm for x in seps)):
            return {"source": "pipeline_" + status, "starts_px": seps, "ticks": seps, "layout": lab.get("candidate"), "rows": rows, "status": status}
        ticks = []; sep_status = status
    else:
        ticks = detect_ticks(ink, rows, ppm) if use_ticks else []; sep_status = None
    grid = match_grid(ticks, W, ppm) if (use_ticks and separator != "pipeline") else None
    if grid:
        return {"source": "ticks", "starts_px": grid, "ticks": ticks, "layout": lab.get("candidate"), "rows": rows}
    starts = []
    if lab.get("candidate"):
        ncols = max(len(r) for r in I.CAT[lab["candidate"]].pages[0])
        end = content_end(ink, rows, ppm)
        end = end if end is not None else W - 6.0 * ppm
        grid = [k * end / ncols for k in range(1, ncols)]                           # equal grid inside the trace area
        x0s = (lab.get("label_x0_px") or [])[1:]
        if fallback == "label_edge":        # the boundary is placed a few pixel columns before the first label glyph of the next column, so the neighbouring crop ends just before the label
            starts = [(x0s[k] - margin_px) if k < len(x0s) and x0s[k] is not None else grid[k] for k in range(ncols - 1)]
        elif fallback == "grid_capped":     # equal grid, but never to the right of the label edge: the neighbouring crop still ends before the label, and no further right than needed
            starts = [min(grid[k], x0s[k] - margin_px) if k < len(x0s) and x0s[k] is not None else grid[k] for k in range(ncols - 1)]
        else:
            starts = grid
    return {"source": "labels" if starts else "none", "starts_px": starts, "ticks": ticks, "layout": lab.get("candidate"), "rows": rows, "status": sep_status}
