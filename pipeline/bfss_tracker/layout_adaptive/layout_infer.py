"""Layout inference from row baselines + label words (no printed separators needed). Own code.

    out = infer(img_bgr)   {"rows": [...], "col_starts_px": [...], "ncols_per_row": [...], "candidate": layout name, "family": "3x4+1", "boundaries_px": [[x per column] per row], "n_words_kept": int}

1. rows from layout_detect.detect_rows; px/mm from the grid.
2. label words (label_blobs) assigned to rows by y; words with the most common offset to their baseline are kept (real labels sit at the same height above every baseline, false hits scatter).
3. column starts = clusters of word x0 that occur in at least half of the rows that have any word beyond the first column; per row, the columns present.
4. signature (columns per row) -> catalogue layouts; glyph counts of the words (I=1, II=2, III=3, aVR=3, -aVR=4, V1=2) choose standard vs Cabrera among layouts with the same signature.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import layout_detect as D, label_detect as L, label_blobs as B
from layouts import catalogue

CAT = catalogue()
def signature(lay): return tuple(len(r) for r in lay.pages[0])
SIGS = {}
for n, lay in CAT.items(): SIGS.setdefault(signature(lay), []).append(n)


def infer(img_bgr: np.ndarray):
    ink = D.ink_mask(img_bgr); ppm = L.estimate_px_per_mm(img_bgr); H, W = ink.shape
    rows, _ = D.detect_rows(ink); pitch = float(np.median(np.diff(rows))) if len(rows) > 1 else H
    words = B.find_label_words(ink, ppm)
    cell = []                                         # (row index, dy, word)
    for w in words:
        yc = (w["y0"] + w["y1"]) / 2; r = int(np.argmin([abs(yc - b) for b in rows])); dy = yc - rows[r]
        if -0.65 * pitch <= dy <= 0.5 * pitch: cell.append((r, dy, w))
    if not cell: return {"rows": rows, "ncols_per_row": [1] * len(rows), "candidate": None, "family": None, "col_starts_px": [], "boundaries_px": [], "n_words_kept": 0}
    dys = np.array([c[1] for c in cell]); bins = np.arange(dys.min() - 1.5 * ppm, dys.max() + 1.5 * ppm, ppm)       # 1 mm bins
    hist = np.array([((dys >= b) & (dys < b + 3 * ppm)).sum() for b in bins]); mode = bins[int(np.argmax(hist))] + 1.5 * ppm
    kept = [(r, w) for r, dy, w in cell if abs(dy - mode) <= 1.5 * ppm]
    # column starts: cluster word x0 across rows
    xs = sorted(w["x0"] for _, w in kept); clusters = []
    for x in xs:
        if clusters and x - clusters[-1][-1] <= 3.0 * ppm: clusters[-1].append(x)
        else: clusters.append([x])
    rows_with = lambda cl: {r for r, w in kept if min(cl) - 1 <= w["x0"] <= max(cl) + 1}
    n_rows_multi = len({r for r, w in kept if w["x0"] < 0.1 * W})          # rows that have a first-column label: a later column must be present in at least half of them
    starts = []
    for cl in clusters:
        rs = rows_with(cl)
        if np.mean(cl) < 0.1 * W: starts.append(float(np.mean(cl)))                          # first column: any row
        elif len(rs) >= max(2, 0.5 * n_rows_multi): starts.append(float(np.mean(cl)))
    ncols = [sum(1 for c in starts if any(r == i and abs(w["x0"] - c) <= 3.0 * ppm for r, w in kept)) or 1 for i in range(len(rows))]
    sig = tuple(ncols); cands = SIGS.get(sig, [])
    cand = None
    if cands:
        def glyph_score(name):
            lay = CAT[name]; sc = 0
            for ri, row in enumerate(lay.pages[0]):
                for seg in row:
                    c = starts[seg.col] if seg.col < len(starts) else None
                    for r, w in kept:
                        if r == ri and c is not None and abs(w["x0"] - c) <= 3.0 * ppm: sc += (w["n_glyphs"] == len(seg.lead))
            return sc
        cand = max(cands, key=glyph_score)
    fam = None if cand is None else f"{CAT[cand].arrangement}+{CAT[cand].rhythm}"
    bounds = [[c - 1.0 * ppm for c in starts] for _ in rows]                                    # a label starts about 1-2 mm right of the segment start (first column 1 mm, others 2.2 mm)
    return {"rows": rows, "col_starts_px": starts, "ncols_per_row": ncols, "candidate": cand, "family": fam, "boundaries_px": bounds, "n_words_kept": len(kept), "px_per_mm": ppm}


# ---------------------------------------------------------------- model-based inference
def expected_points(lay, content_mm):
    """(row, x_mm of the segment start, label) for every segment of a catalogue layout; the label sits ~1-2 mm right of the start (absorbed by the fitted shift)."""
    pts = []
    for ri, row in enumerate(lay.pages[0]):
        for seg in row: pts.append((ri, seg.col * content_mm / seg.ncols, seg.lead))
    return pts


def infer2(img_bgr: np.ndarray, cal_mm: float = 6.0, only_names=None, w_unmatched: float = 0.5, glyph_penalty: float = 0.35):
    """Layout by matching predicted label positions to the detected words (rows fixed by the comb), with a global shift/scale fit and a cost for missing labels."""
    ink = D.ink_mask(img_bgr); ppm = L.estimate_px_per_mm(img_bgr); H, W = ink.shape
    rows, _ = D.detect_rows(ink); R = len(rows); pitch = float(np.median(np.diff(rows))) if R > 1 else H
    words = B.find_label_words(ink, ppm)
    cell = []
    for w in words:
        yc = (w["y0"] + w["y1"]) / 2; r = int(np.argmin([abs(yc - b) for b in rows])); dy = yc - rows[r]
        if -0.65 * pitch <= dy <= 0.5 * pitch: cell.append((r, dy, w))
    if not cell: return {"rows": rows, "candidate": None, "family": None, "cost": None, "n_words": 0}
    dys = np.array([c[1] for c in cell]); bins = np.arange(dys.min() - 1.5 * ppm, dys.max() + 1.5 * ppm, ppm)
    hist = np.array([((dys >= b) & (dys < b + 3 * ppm)).sum() for b in bins]); mode = bins[int(np.argmax(hist))] + 1.5 * ppm
    kept = [(r, w) for r, dy, w in cell if abs(dy - mode) <= 1.5 * ppm]
    content_mm = W / ppm - cal_mm
    best = None
    for name, lay in CAT.items():
        if only_names and name not in only_names: continue
        if len(lay.pages[0]) != R: continue
        exp = expected_points(lay, content_mm)
        for s in np.linspace(0.92, 1.08, 17):
            for b in np.linspace(-1.0, 4.0, 21):                         # label x0 minus segment start, mm (first column ~1, others ~2.2)
                cost = 0.0; matched = set()
                for ri, xm, lead in exp:
                    xp = (xm * s + b) * ppm; cand = [(abs(w["x0"] - xp), j, w) for j, (r, w) in enumerate(kept) if r == ri]
                    if not cand: cost += 1.0; continue
                    d, j, w = min(cand, key=lambda c: c[0]); d_mm = d / ppm
                    if d_mm > 3.0: cost += 1.0; continue
                    matched.add(j); cost += d_mm / 6.0 + (0.0 if w["n_glyphs"] == len(lead) else glyph_penalty)
                cost += w_unmatched * (len(kept) - len(matched)) / max(1, len(exp))
                if best is None or cost < best[0]: best = (cost, name, s, b)
    if best is None: return {"rows": rows, "candidate": None, "family": None, "cost": None, "n_words": len(kept)}
    cost, name, s, b = best; lay = CAT[name]
    starts_mm = sorted({seg.col * content_mm / seg.ncols for seg in lay.pages[0][0]})
    model_starts = [(x * s) * ppm for x in starts_mm]                       # geometry of the layout model (assumes the renderer's calibration-pulse width)
    # starts from the detected labels themselves: median x0 of the words matched to each column minus the typical label offset right of the segment start (2.2 mm)
    label_starts = []; label_x0 = []
    for k, ms in enumerate(model_starts):
        xs = [w["x0"] for r, w in kept if abs(w["x0"] - (ms + b * ppm)) <= 3.0 * ppm]
        label_x0.append(float(np.median(xs)) if xs else None)                  # left edge of the labels of this column (median over rows), None if no label was found
        label_starts.append(float(np.median(xs)) - (2.2 if k > 0 else 1.0) * ppm if xs else ms)
    # assignment of detected words to the label slots of the chosen layout: slot (row, col, lead) -> nearest word of that row within 3 mm of the predicted x, else None
    assign = []
    for ri, xm, lead in expected_points(lay, content_mm):
        xp = (xm * s + b) * ppm; cand = [(abs(w["x0"] - xp), w) for r, w in kept if r == ri and abs(w["x0"] - xp) <= 3.0 * ppm]
        col = int(round(xm / (content_mm / max(1, max(len(r) for r in lay.pages[0])))))
        assign.append({"row": ri, "x_mm": xm, "lead": lead, "word": min(cand, key=lambda c: c[0])[1] if cand else None,
                       "slot_xy": (xp + 0.0, rows[ri] + mode)})          # predicted left edge and vertical centre of this slot's label, whether or not a word was found there
    return {"rows": rows, "candidate": name, "family": f"{lay.arrangement}+{lay.rhythm}", "cost": cost, "scale": s, "shift_mm": b, "n_words": len(kept),
            "col_starts_px": label_starts, "label_x0_px": label_x0, "model_starts_px": model_starts, "px_per_mm": ppm, "assign": assign, "ink": ink}
