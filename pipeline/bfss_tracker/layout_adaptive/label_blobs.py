"""Font-agnostic detection of printed lead labels as text blobs (classical, no learning). Own code.

    words = find_label_words(ink, px_per_mm)   [{"x0","y0","x1","y1","n_glyphs"}]

A glyph is a connected ink component of glyph size (height 1.5-5.5 mm, width 0.6-4 mm); glyphs on one text line with small gaps form a word. Traces, ticks (too thin), the calibration
pulse (too tall) and grid are rejected by size. Labels that touch a trace merge into the trace component and are missed (reported as recall loss).
"""
from __future__ import annotations
import cv2, numpy as np


def find_label_words(ink: np.ndarray, px_per_mm: float, h_mm=(1.5, 5.5), w_mm=(0.6, 4.0), gap_mm=1.8, line_mm=1.3, max_glyphs=5, side_mm=8.0):
    ink8 = ink.astype(np.uint8)
    H, W = ink8.shape
    joined = cv2.dilate(ink8, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))                      # bridge the 1-8 px breaks of the thin anti-aliased trace
    nj, lj, sj, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    is_trace = np.zeros(nj, bool); is_trace[1:] = sj[1:, cv2.CC_STAT_WIDTH] >= 0.06 * W                # long components = trace rows
    trace_mask = is_trace[lj]
    vert = cv2.morphologyEx(ink8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(3.0 * px_per_mm)))))
    flat8 = ink8 & (1 - vert)                       # ink without tall vertical strokes (QRS, ticks): what is left of a trace beside a label is its flat/sloped parts
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink8, connectivity=8)
    g = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h_mm[0] * px_per_mm <= h <= h_mm[1] * px_per_mm and w_mm[0] * px_per_mm <= w <= w_mm[1] * px_per_mm: g.append((x, y, x + w, y + h))
    g.sort(key=lambda b: (b[0], b[1]))
    used = [False] * len(g); words = []
    for i, b in enumerate(g):
        if used[i]: continue
        used[i] = True; cur = [b]; grew = True
        while grew and len(cur) < max_glyphs:
            grew = False
            for j, c in enumerate(g):
                if used[j]: continue
                x1 = max(q[2] for q in cur); yc = np.mean([(q[1] + q[3]) / 2 for q in cur])
                if 0 <= c[0] - x1 <= gap_mm * px_per_mm and abs((c[1] + c[3]) / 2 - yc) <= line_mm * px_per_mm:
                    cur.append(c); used[j] = True; grew = True; break
        wd = {"x0": min(q[0] for q in cur), "y0": min(q[1] for q in cur), "x1": max(q[2] for q in cur), "y1": max(q[3] for q in cur), "n_glyphs": len(cur)}
        if _on_trace(flat8, wd, side_mm * px_per_mm): continue
        words.append(wd)
    return words


def _on_trace(ink8, wd, side_px, min_px=6):
    """A text label has free space to at least one side in its own band; a bump of the trace (P, T wave, noise) has trace ink on both sides."""
    H, W = ink8.shape; y0, y1 = max(0, wd["y0"]), min(H, wd["y1"] + 1); s = int(side_px)
    left = int(ink8[y0:y1, max(0, wd["x0"] - s): max(0, wd["x0"] - 1)].sum()); right = int(ink8[y0:y1, min(W, wd["x1"] + 2): min(W, wd["x1"] + s)].sum())
    return left >= min_px and right >= min_px
