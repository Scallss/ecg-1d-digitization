"""Generic layout detection for single-page 12-lead sheets (classical, no learning). Own code.

    info = detect(img_bgr)  ->  {"rows": [baseline_y...], "ticks": [[x...] per row], "ncols": [n per row], "signature": "4,4,4,1", "arrangement": "3x4+1", "confidence": ...}

Step 1  ink mask from the pipeline's black filter (experiments use pipeline/bfss_tracker/ecg_digitizer.preprocess_simple_black).
Step 2  rows: baseline-emphasised horizontal projection, peaks; spacing regularised (rows are about equally spaced).
Step 3  columns: printed column ticks = the same short vertical ink strokes at the same x in every tick row (cross-row stability), centred on the baseline.
Step 4  arrangement from the per-row column counts.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2, numpy as np
from scipy.signal import find_peaks

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]  # this folder + pipeline/bfss_tracker
import ecg_digitizer as E  # noqa: E402


def ink_mask(img_bgr: np.ndarray) -> np.ndarray:
    return E.preprocess_simple_black(img_bgr, {"simple_black_thresh": 165})[1] > 0


def remove_small_components(ink8: np.ndarray, min_width_px: float) -> np.ndarray:
    """Drop 8-connected ink components narrower than min_width_px (text, printed ticks, calibration pulses); trace rows are far wider."""
    joined = cv2.dilate(ink8, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))      # join the fragments of a thin anti-aliased line
    n, lab, st, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_WIDTH] >= min_width_px
    return ink8 * keep[lab].astype(np.uint8)


def detect_rows(ink: np.ndarray, min_rows: int = 3, max_rows: int = 14):
    """Baselines of the trace rows. Remove tall vertical strokes (QRS, ticks) so the projection is dominated by the flat parts, then peak-pick."""
    h, w = ink.shape
    ink8 = remove_small_components(ink.astype(np.uint8) * 255, 0.06 * w)    # labels, ticks, calibration pulses
    roi = ink8[:, int(0.05 * w): int(0.93 * w)]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(0.04 * h))))
    vert = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k)
    flat = cv2.bitwise_and(roi, cv2.bitwise_not(vert))
    proj = (flat > 0).sum(1).astype(np.float32)
    sm = cv2.GaussianBlur(proj.reshape(-1, 1), (1, 0), sigmaX=0.002 * h + 1, sigmaY=max(1.0, 0.004 * h)).ravel()
    return comb_rows(sm, h, min_rows, max_rows), sm


def comb_rows(sm: np.ndarray, h: int, min_rows: int, max_rows: int):
    """Rows are about equally spaced: fit a comb (R teeth, pitch, phase) to the smoothed projection. Each tooth scores by how far it stands above the valley next to it; teeth in valleys and extra teeth cost. Returns the baseline y of each tooth (refined to the local maximum)."""
    pk, _ = find_peaks(sm, distance=max(3, int(0.02 * h)), prominence=0.05 * sm.max())
    heights = np.sort(sm[pk])[::-1]; ref = float(np.median(heights[: max(3, len(heights) // 2)])) or 1.0
    best, best_score = None, -1e9
    for R in range(min_rows, max_rows + 1):
        for pitch in np.linspace(0.8 * h / R, 1.1 * h / R, 25):
            for y0 in np.arange(0.02 * h, pitch + 0.02 * h, max(1.0, 0.004 * h)):
                ys = y0 + pitch * np.arange(R)
                if ys[-1] > h - 0.02 * h: break
                half = max(2, int(0.2 * pitch)); q = int(0.5 * pitch)
                tooth = np.array([sm[max(0, int(y) - half): int(y) + half + 1].max() for y in ys])
                valley = np.array([min(sm[max(0, int(y) - q): int(y) + 1].min(), sm[int(y): min(h, int(y) + q + 1)].min()) for y in ys])
                c = (tooth - valley) / ref                     # how far each tooth stands above the lowest point on either side within half a pitch
                score = np.clip(c, 0, 1.0).sum() - 1.0 * (c < 0.25).sum() - 0.3 * R       # teeth in valleys and extra teeth cost; chosen on 24 PTB-XL records x 22 layouts (comb_score_lab.py)
                if score > best_score: best_score, best = score, ys
    ys = best; half = max(2, int(0.2 * (ys[1] - ys[0])))
    return [int(max(0, int(y) - half) + np.argmax(sm[max(0, int(y) - half): int(y) + half + 1])) for y in ys]

