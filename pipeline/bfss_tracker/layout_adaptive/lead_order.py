"""Lead order of a 3x4 + rhythm sheet: accept on the labels, else check the signal relations, else flag (scheme chosen by the user 2026-10-12). Own code.

    out = decide(img_bgr, bank)   {"status": "labels" | "relations" | "unverified", "order": [12 lead names in row-major slot order] or None, "detail": {...}}

1. Labels: layout slots by geometry (layout_infer.infer2, glyph penalty 0, 3x4 + 1 only), each slot's word read by template matching (label_reader, templates from real scans), one-to-one
   assignment. Accepted only when all 12 slots have a word and every assigned name has similarity >= TAU_LABEL.
2. Otherwise: signals of the first two columns (rows 0-2) are extracted with the default overlap tracker from the detected rows and the column starts (ticks first, labels as fallback).
   Standard order if II = I + III (col 1) and aVR + aVL + aVF = 0 (col 2); Cabrera order if II = I + III-type relation (col 1: aVL + (-aVR) = I) and aVF = (II + III) / 2 (col 2); residuals < T_REL.
3. Otherwise the order is unverified (None).
Scope: 3x4 + rhythm sheets (the layout of our scans). Thresholds were fixed on earlier experiments (ledger L11: 0.7, L9: 0.55) and not tuned here.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import layout_infer as I, label_reader as R, label_blobs as B, ticks_fallback as T
from layouts import STANDARD, CABRERA, catalogue
from overlap_tracker import trace as _trace
def trace(ink): return _trace(ink > 0)  # frozen tracker copy of pipeline/bfss_tracker (same call as track_overlap_trace)

TAU_LABEL, T_REL = 0.7, 0.55
STD_SLOTS = [s.lead for row in catalogue()["3x4+1_std"].pages[0][:3] for s in row]        # row-major order of the 12 main slots
CAB_SLOTS = [s.lead for row in catalogue()["3x4+1_cab"].pages[0][:3] for s in row]
MAIN = STANDARD[:]


def _assign_words(words, slots, ppm):
    out = []
    for s in slots:
        x, y = s["slot_xy"]; c = [(abs(w["x0"] - x) + abs((w["y0"] + w["y1"]) / 2 - y), w) for w in words if abs(w["x0"] - x) <= 3 * ppm and abs((w["y0"] + w["y1"]) / 2 - y) <= 3 * ppm]
        out.append(min(c, key=lambda t: t[0])[1] if c else None)
    return out


def has_leading_dash(ink, w, ppm):
    """A minus sign printed just left of a label word (-aVR on Cabrera sheets): a thin, flat ink component 0.8-3 mm wide and at most 0.7 mm tall, within 2.5 mm left of the word at mid-height."""
    import cv2
    ym = (w["y0"] + w["y1"]) // 2; ya, yb = int(ym - 1.4 * ppm), int(ym + 1.4 * ppm); xa, xb = int(w["x0"] - 3.2 * ppm), int(w["x0"] - 0.1 * ppm)
    if xa < 0 or ya < 0: return False
    reg = ink[ya:yb, xa:xb].astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(reg, connectivity=8)
    return any(0.8 * ppm <= st[i, cv2.CC_STAT_WIDTH] <= 3.0 * ppm and st[i, cv2.CC_STAT_HEIGHT] <= 0.7 * ppm and st[i, cv2.CC_STAT_WIDTH] >= 2.5 * st[i, cv2.CC_STAT_HEIGHT] for i in range(1, n))


def read_labels(ink, ppm, slots, bank):
    words = _assign_words(B.find_label_words(ink, ppm), slots, ppm)
    sim = np.full((12, 12), -5.0)
    for i, w in enumerate(words):
        if w is None: continue
        sc = R.scores(ink, w, ppm, bank, None)
        for j, n in enumerate(MAIN):
            if n in sc: sim[i, j] = sc[n]
    cols = R.assign_unique(sim); picked = [sim[i, c] for i, c in enumerate(cols)]
    found = sum(w is not None for w in words)
    names = [MAIN[c] for c in cols]
    for i, w in enumerate(words):                                    # '-aVR': the same word with a minus sign in front (sign of the signal is inverted)
        if w is not None and names[i] == "aVR" and has_leading_dash(ink, w, ppm): names[i] = "-aVR"
    return names, found, float(min(picked))


def column_signals(ink, rows, starts, end):
    H = ink.shape[0]; pitch = float(np.median(np.diff(rows))); half = int(0.5 * pitch)
    edges = [0.0] + list(starts) + [end]; sig = {}
    for c in (0, 1):
        xa, xb = int(edges[c]), int(edges[c + 1]); rs = []
        for r in range(3):
            b = int(rows[r]); y0, y1 = max(0, b - half), min(H, b + half + 1)
            y = np.asarray(trace(ink[y0:y1, xa:xb]), float)
            if np.all(np.isnan(y)): y = np.zeros(xb - xa)
            idx = np.arange(len(y)); ok = ~np.isnan(y); y = np.interp(idx, idx[ok], y[ok]) if ok.any() else y
            rs.append(-(y + y0 - b))                                   # up is positive, relative to the detected baseline
        n = min(len(v) for v in rs); sig[c] = [v[:n] - np.median(v[:n]) for v in rs]
    return sig


def relations(sig):
    (a1, a2, a3), (b1, b2, b3) = sig[0], sig[1]
    e1 = np.linalg.norm(a2 - a1 - a3) / (np.linalg.norm(a2) + 1e-9)
    es = np.linalg.norm(b1 + b2 + b3) / (np.linalg.norm(b1) + np.linalg.norm(b2) + np.linalg.norm(b3) + 1e-9)
    ec = np.linalg.norm(b2 - (b1 + b3) / 2) / (np.linalg.norm(b2) + 1e-9)
    return e1, es, ec


def decide(img_bgr, bank):
    o = I.infer2(img_bgr, only_names=["3x4+1_std"], glyph_penalty=0.0)
    if o.get("candidate") is None or len(o["rows"]) != 4: return {"status": "unverified", "order": None, "detail": {"reason": "not a 3x4 + rhythm layout"}}
    ink, ppm, rows = o["ink"], o["px_per_mm"], o["rows"]; slots = [a for a in o["assign"] if a["row"] < 3]
    names, found, weakest = read_labels(ink, ppm, slots, bank)
    detail = {"found": found, "weakest_similarity": weakest}
    if found == 12 and weakest >= TAU_LABEL: return {"status": "labels", "order": names, "detail": detail}
    cs = T.column_starts(img_bgr); end = T.content_end(ink, rows, ppm) or ink.shape[1] - 6 * ppm
    starts = cs["starts_px"] if len(cs["starts_px"]) == 3 else [k * end / 4 for k in (1, 2, 3)]
    e1, es, ec = relations(column_signals(ink, rows, starts, end)); detail.update({"e1": e1, "es": es, "ec": ec})
    if e1 < T_REL and es < T_REL and not (ec < es): return {"status": "relations", "order": STD_SLOTS, "detail": detail}
    if e1 < T_REL and ec < T_REL: return {"status": "relations", "order": CAB_SLOTS, "detail": detail}
    if e1 < T_REL and es < T_REL: return {"status": "relations", "order": STD_SLOTS, "detail": detail}
    return {"status": "unverified", "order": None, "detail": detail}
