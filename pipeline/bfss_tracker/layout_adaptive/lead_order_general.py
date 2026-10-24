"""Lead order for every layout of the catalogue: accept on the labels, else check the signal relations, else flag. Own code. Generalises lead_order.py (3x4 + rhythm only).

    out = decide(img_bgr, bank, t_rel)  {"status": "labels"|"relations"|"unverified", "order": [12 names in the row-major order of the main slots] or None, "layout": catalogue name, "detail": {...}}

* Geometry: layout_infer.infer2 over the whole catalogue with glyph penalty 0 (names are not used to pick the layout); standard / Cabrera / time-shifted / simultaneous variants of one arrangement share geometry.
* Labels: slots of the detected layout (rhythm rows excluded), word per slot, template reading, one-to-one assignment, minus-sign check (-aVR). Accepted when all 12 words are found with similarity >= TAU_LABEL.
* Relations: leads of one column share one time window in every layout (shifted and simultaneous alike), so a column that holds >= 3 limb leads must satisfy the limb relations: with the limb leads written in the (I, II) basis
  I=(1,0) II=(0,1) III=(-1,1) aVR=(-.5,-.5) aVL=(1,-.5) aVF=(-.5,1), the signals S (k x n) of those leads must lie in the span of their vectors V (k x 2): residual = |S - V V+ S| / |S|.
  A hypothesis (standard or Cabrera names for the slots of the detected arrangement) is accepted when all its testable columns pass (residual < t_rel) and the other one fails at least one.
  Arrangements with no column holding >= 3 limb leads (4 rows x 3 columns grouped) cannot be checked.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import layout_infer as I, label_reader as R, label_blobs as B, ticks_fallback as T
import lead_order as LO3
from layouts import STANDARD, catalogue, build
from overlap_tracker import trace as _trace
def trace(ink): return _trace(ink > 0)  # frozen tracker copy of pipeline/bfss_tracker (same call as track_overlap_trace)

TAU_LABEL = 0.7
LIMB = {"I": (1, 0), "II": (0, 1), "III": (-1, 1), "aVR": (-.5, -.5), "aVL": (1, -.5), "aVF": (-.5, 1)}
CAT = catalogue()


def is_rhythm(lay, ri, seg):
    return bool(seg.ncols == 1 and not seg.shifted and lay.rhythm and ri >= len(lay.pages[0]) - lay.rhythm)


def main_segments(lay):
    """[(row, seg)] of the 12 main slots in row-major order (rhythm rows excluded)."""
    return [(ri, seg) for ri, row in enumerate(lay.pages[0]) for seg in row if not is_rhythm(lay, ri, seg)]


def vec(name):
    s = -1.0 if name.startswith("-") else 1.0
    return s * np.array(LIMB[name.lstrip("-")], float)


def rank_residual(S, names):
    V = np.stack([vec(n) for n in names]); P = V @ np.linalg.pinv(V)
    return float(np.linalg.norm(S - P @ S) / (np.linalg.norm(S) + 1e-9))


def segment_signals(ink, rows, lay, starts_by_ncols, end):
    """signal (up positive, median removed) of every main slot, in the order of main_segments(lay); from the detected rows and the column starts."""
    H = ink.shape[0]; pitch = float(np.median(np.diff(rows))) if len(rows) > 1 else H; half = int(0.5 * pitch); out = []
    for ri, seg in main_segments(lay):
        n = seg.ncols; e = [0.0] + (starts_by_ncols.get(n) or [k * end / n for k in range(1, n)]) + [end]
        xa, xb = int(e[seg.col]), int(e[seg.col + 1]); b = int(rows[ri]); y0, y1 = max(0, b - half), min(H, b + half + 1)
        y = np.asarray(trace(ink[y0:y1, xa:xb]), float)
        if np.all(np.isnan(y)): y = np.zeros(max(2, xb - xa))
        idx = np.arange(len(y)); ok = ~np.isnan(y); y = np.interp(idx, idx[ok], y[ok])
        v = -(y + y0 - b); out.append(v - np.median(v))
    return out


def column_groups(lay):
    """for each time-aligned group (the main slots of one column, same ncols): list of indices into main_segments(lay)."""
    groups = {}
    for i, (ri, seg) in enumerate(main_segments(lay)): groups.setdefault((seg.ncols, seg.col), []).append(i)
    return list(groups.values())


def hypothesis_residuals(signals, lay, names_for_slots):
    """residual of every testable group (>= 3 limb leads) under the claimed names (one per main slot)."""
    res = []
    for g in column_groups(lay):
        idx = [i for i in g if names_for_slots[i].lstrip("-") in LIMB]
        if len(idx) < 3: continue
        n = min(len(signals[i]) for i in idx); S = np.stack([signals[i][:n] for i in idx]); res.append(rank_residual(S, [names_for_slots[i] for i in idx]))
    return res


def hyp_names(arrangement, rhythm, order):
    lay = build(arrangement, order, rhythm); return [seg.lead for _, seg in main_segments(lay)]


def decide(img_bgr, bank, t_rel=0.35, return_residuals=False):
    o = I.infer2(img_bgr, glyph_penalty=0.0)
    if o.get("candidate") is None: return {"status": "unverified", "order": None, "layout": None, "detail": {"reason": "no layout"}}
    lay = CAT[o["candidate"]]; ink, ppm, rows = o["ink"], o["px_per_mm"], o["rows"]
    segs = main_segments(lay); assign = o["assign"]
    slot_ids = [k for k, (ri, row) in enumerate((ri, seg) for ri, rw in enumerate(lay.pages[0]) for seg in rw) if not is_rhythm(lay, ri, row)]
    slots = [assign[k] for k in slot_ids]
    names, found, weakest = LO3.read_labels(ink, ppm, slots, bank) if len(slots) == 12 else (None, 0, -1.0)
    detail = {"layout": o["candidate"], "found": found, "weakest_similarity": weakest}
    if names and found == 12 and weakest >= TAU_LABEL: return {"status": "labels", "order": names, "layout": o["candidate"], "detail": detail}
    cs = T.column_starts(img_bgr); end = T.content_end(ink, rows, ppm) or ink.shape[1] - 6 * ppm
    ncmax = max(len(r) for r in lay.pages[0]); starts = {ncmax: cs["starts_px"]} if len(cs["starts_px"]) == ncmax - 1 else {}
    signals = segment_signals(ink, rows, lay, starts, end)
    res = {order: hypothesis_residuals(signals, lay, hyp_names(lay.arrangement, lay.rhythm, order)) for order in ("standard", "cabrera")}
    detail["residuals"] = res
    ok = {k: (len(v) > 0 and max(v) < t_rel) for k, v in res.items()}
    if ok["standard"] != ok["cabrera"]:
        order = "standard" if ok["standard"] else "cabrera"
        return {"status": "relations", "order": hyp_names(lay.arrangement, lay.rhythm, order), "layout": o["candidate"], "detail": detail}
    return {"status": "unverified", "order": None, "layout": o["candidate"], "detail": detail}
