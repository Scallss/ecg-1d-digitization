"""Layout plan of one sheet for the adaptive preset (layout_method="adaptive"): rows, layout, column edges, lead name of every slot.

    plan = make_plan(img_bgr)  ->  dict
        layout, arrangement, rhythm            catalogue name and its parts ("3x4", 1); None when no layout could be inferred
        order_status                           "labels" | "relations" | "unverified" (lead_order_general.decide); unverified sheets get the standard names
        delegate                               True when the sheet is 3x4 + one rhythm strip in the standard order (or unverified): the caller then runs the fixed 3x4 path unchanged
        slots                                  [{name, sign, kind main|rhythm, row, col, ncols, shifted, x_edges (px), baseline (px)}]
        rows, px_per_mm, starts_source, edges  geometry; time_mode is always "assumed_shifted" (shifted vs simultaneous is not detected)
Frozen copy of experiments/layout_and_label_separators/detect (2026-10-17), see ../README.md.
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import layout_detect as D, label_detect as L, ticks_fallback as T, lead_order_general as G  # noqa: E402
from layouts import STANDARD  # noqa: E402

BANK = pickle.load(open(HERE / "real_label_bank.pkl", "rb"))["bank"]
T_REL = 0.40          # relation threshold fixed by calibrate_rel.py (ledger L9, L15)


def fine_px_per_mm(img_bgr: np.ndarray, kmax_frac: float = 0.6) -> float:
    """px per mm from the grid, refined over many grid periods: the coarse estimate (label_detect.estimate_px_per_mm) is the integer period of the first autocorrelation peaks and is
    1% off, which would stretch the time axis by 1%. Here the autocorrelation peaks of the column-median green channel at lags k x period (k up to 60% of the width) are fitted by a line through the origin."""
    p = L.estimate_px_per_mm(img_bgr)
    g = img_bgr[:, :, 1].astype(np.float32); col = np.median(g, axis=0); col = col - col.mean()
    ac = np.correlate(col, col, "full")[len(col) - 1:]; ac /= ac[0] + 1e-9
    W = len(col); ks, xs = [], []; k = 1
    while k * p < kmax_frac * W:
        c = k * p; lo, hi = int(c - 0.3 * p), int(c + 0.3 * p) + 1
        i = lo + int(np.argmax(ac[lo:hi]))
        if ac[i] > 0.05: ks.append(k); xs.append(i)
        if len(ks) >= 3: p = float(np.dot(ks, xs) / np.dot(ks, ks))
        k += 1
    return float(p)


def make_plan(img_bgr: np.ndarray, t_rel: float = T_REL) -> dict:
    d = G.decide(img_bgr, BANK, t_rel=t_rel)
    plan = {"layout": d["layout"], "order_status": d["status"], "detail": d["detail"], "slots": [], "delegate": False, "time_mode": "assumed_shifted"}
    if d["layout"] is None:
        plan["delegate"] = True; plan["reason"] = "no layout"; return plan
    lay = G.CAT[d["layout"]]; plan.update(arrangement=lay.arrangement, rhythm=lay.rhythm)
    main = G.main_segments(lay)
    names = d["order"] if d["order"] else [seg.name for _, seg in G.main_segments(G.build(lay.arrangement, "standard", lay.rhythm))]
    std3 = G.hyp_names("3x4", 1, "standard")
    if lay.arrangement == "3x4" and lay.rhythm == 1 and list(names) == std3:
        plan["delegate"] = True; plan["reason"] = "3x4+1 standard order" + ("" if d["order"] else " (unverified, default names)"); return plan
    ink = D.ink_mask(img_bgr); ppm = L.estimate_px_per_mm(img_bgr); H, W = ink.shape
    cs = T.column_starts(img_bgr); rows = cs["rows"]
    end = T.content_end(ink, rows, ppm) or (W - 6.0 * ppm)
    ncmax = max(len(r) for r in lay.pages[0]); starts = list(cs["starts_px"]) if ncmax > 1 else []
    if len(starts) != ncmax - 1:
        starts = [k * end / ncmax for k in range(1, ncmax)]; plan["starts_source"] = "equal_grid_recomputed"
    else:
        plan["starts_source"] = cs["source"]
    edges = [0.0] + starts + [float(end)]
    ri_of = {}
    for ri, row in enumerate(lay.pages[0]):
        for seg in row: ri_of[id(seg)] = ri
    slots = []
    for (ri, seg), nm in zip(main, names):
        sign = -1 if nm.startswith("-") else 1
        slots.append({"name": nm.lstrip("-"), "sign": sign, "kind": "main", "row": ri, "col": seg.col, "ncols": seg.ncols, "shifted": True,
                      "x_edges": [edges[seg.col], edges[seg.col + 1]], "baseline": float(rows[ri])})
    for ri, row in enumerate(lay.pages[0]):
        for seg in row:
            if G.is_rhythm(lay, ri, seg):
                slots.append({"name": seg.name, "sign": 1, "kind": "rhythm", "row": ri, "col": 0, "ncols": 1, "shifted": False, "x_edges": [0.0, float(end)], "baseline": float(rows[ri])})
    plan.update(px_per_mm_fine=fine_px_per_mm(img_bgr), slots=slots, rows=[float(r) for r in rows], px_per_mm=float(ppm), edges=edges, content_end_px=float(end), column_status=cs.get("status"))
    return plan
