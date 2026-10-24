# Self-contained copy of ecg_digitization/experiments/overlap_trace_tracking/core/tracker.py (default overlap tracker, decision 2026-10-11).
# Do not edit here; change the experiment folder and re-copy. Entry point: trace(ink) at the bottom.
"""Column-by-column trace trackers for overlapping ECG leads (no learning).

Every crop is a binary image (True = ink). In each column the ink forms one or more
vertical *runs* (a run = consecutive ink pixels). One run per column belongs to our
lead; the rest is a neighbour's trace. Trackers choose the run; the output y is then
read from that run.

Trackers
  median      production behaviour: median row of ALL ink in the column (the baseline).
  nearest     one pass, keep the run closest to a straight-line guess from the last two picks.
  greedy      user's idea, one hypothesis: predict from recent slope, pick the best run, never go back.
  beam        same cost, but keeps B alternatives alive and decides at the end (MHT / Viterbi-like).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import os

try:  # optional compiled version of track() (numba); identical paths, see overlap_tracker_fast.py. Off unless OVERLAP_TRACKER_FAST=1.
    import overlap_tracker_fast as _fast
except Exception:
    _fast = None
_FAST_ON = os.environ.get("OVERLAP_TRACKER_FAST", "0") == "1" and _fast is not None and _fast.AVAILABLE


@dataclass
class P:
    beam: int = 8            # hypotheses kept (1 = greedy)
    w_pred: float = 1.0      # cost per px of distance to the slope-predicted y
    w_gap: float = 2.0       # cost per px of vertical gap between consecutive runs (0 if touching)
    w_base: float = 0.0      # cost per column spent away from the lead baseline
    base_tol: float = 12.0   # px: closer than this to baseline is free
    base_cap: float = 40.0   # px: distance beyond this costs no more
    w_edge: float = 0.0      # cost for a run touching the crop top/bottom edge
    w_skip: float = 6.0      # cost for a column with no run chosen
    damp: float = 0.7        # fraction of the slope used for extrapolation
    vmax: float = 25.0       # px/column clip on the slope
    tall: float = 14.0       # runs taller than this are "merged/steep": y is read by prediction
    runs_gap: int = 1        # pixels of hole allowed inside a run
    jump_gate: float = 0.0   # >0: a run whose nearest point is farther than this from the prediction is not a candidate (column treated as missing)
    w_anchor: float = 0.0    # cost per px (up to anchor_cap) between a run and the trusted anchor y of this column; 0 = anchors unused
    anchor_cap: float = 40.0
    gap_gate: float = 0.0    # >0: a run separated from the previous run by more than this many px (ink not connected to where we were) is not a candidate
    slope_pts: int = 3       # number of past y values used for the slope (endpoint slope over the last slope_pts points); 3 = the original behaviour
    gate_mode: str = "gap"   # "gap": only the connectivity gap rule above; "or": a run is admissible if connected (gap rule) OR near the slope prediction; "and": both; "pred": slope prediction only
    pred_gate: float = 0.0   # tolerance (px) of the slope-prediction rule: distance from the predicted y to the run (0 if the prediction falls inside it)
    gate_damp: float = -1.0  # fraction of the recent slope used by the gate's own prediction (-1 = same as damp)
    gate_relax: float = 8.0  # the gate widens by 1/gate_relax of itself per consecutive missing column
    skip_cost: float = 5.0   # cost of a column with no admissible run when jump_gate > 0
    exit_skip_cost: float = -1.0  # >= 0: cost of a skipped column while the path is leaving the crop (last run touches the top/bottom edge within exit_margin px and the slope points to that edge); -1 = off
    exit_margin: float = 3.0
    ret_len: int = 0         # > 0: return-to-baseline gate: while a path is skipping columns, also admit runs near the level it had before the excursion (median of its last ret_len emitted y); 0 = off
    ret_gate: float = 40.0   # tolerance (px) around that level
    ret_cost: float = 20.0   # extra cost of a candidate admitted only through the return level (its distance to the level replaces the distance to the prediction, no gap term)
    ret_min: int = 10        # emitted columns needed before the return level exists
    skip_key_run: bool = False  # skipped hypotheses are recombined per last run instead of all into one shared 'skipped' state (a path that left through the top and one that sits at the baseline are different states)
    skip_cost_nofit: float = -1.0  # >= 0: price of a skipped column for a hypothesis none of whose candidate runs passes the gate in this column (nothing fits, so skipping is not a choice); -1 = off (the normal skip price applies)
    base_mem: int = 0        # > 0: the baseline pull uses the median y of the path's last base_mem emitted columns (once ret_min columns exist) instead of the crop centre; 0 = crop centre
    nofit_near: float = 0.0  # a run within this vertical distance (px) of the previous run counts as ink that fits for skip_cost_nofit, even when it is outside the gate window (a stroke continuing from the previous run); 0 = only gate-admitted runs count
    skip_key_edge: bool = False  # like skip_key_run, but only for skipped hypotheses whose last run touches the top/bottom edge within exit_margin px (paths that left through the crop edge); all other skips stay merged
    exit_confirm: int = 0   # > 0: the exit skip (first column and, up to exit_hold columns, the following ones) is cheap only if a run touches the same crop edge (within exit_margin px) in one of the next exit_confirm columns, i.e. the trace is seen coming back; otherwise a skip costs skip_cost even in the first column; 0 = off
    exit_hold_speed: float = 0.0   # > 0: the exit_hold columns after the first are free only if the last step before the path left was at least exit_hold_speed px per column (the trace was moving fast toward the edge); 0 = every path on the edge
    exit_hold: int = 0       # > 0: the exit-skip price also applies to the following skipped columns, up to exit_hold in a row, while the path's last run is on the crop edge (default 0: only the first skipped column, because the held slope is 0 afterwards)
    ret_after: int = 1       # the return level only applies once the path has skipped at least this many columns in a row
    tall_pred: float = 0.0   # >0: runs taller than this are read by prediction (clamped), not by their centre
    base_adapt: float = 0.0  # >0: per-path baseline follows own trace slowly (rate per column), so drift is not punished; 0 = fixed
    init_pull: float = 0.0   # cost per px of distance between a run and the baseline in the first inked column (breaks the tie between runs at the start of the crop, where nothing else has been seen yet; the crop is cut around the lead, so the baseline run is the lead); 0 = off
    base_tol_frac: float = 0.0  # > 0: the free zone of the baseline pull is base_tol_frac * crop height (px) instead of base_tol, so tall crops tolerate taller waves; 0 = base_tol
    cost_slope_pts: int = 0   # > 0: number of past points of the slope prediction used for the soft cost (distance to the predicted row, quadratic term); the gate and the baseline-pull waiver keep slope_pts; 0 = slope_pts for everything
    peak_read: float = 0.0   # > 0: output readout at peaks: a column whose chosen run is at least peak_read px tall and whose two neighbouring outputs lie on the same side of the run centre is a turning point of the trace; its output is the end of the run away from the neighbours (the tip) instead of the centre; the path itself is unchanged; 0 = centre
    gap_free: float = 0.0   # px of vertical gap between consecutive runs that cost nothing (a thin line steeper than its own thickness leaves a gap of 1-3 px, so only the gap beyond gap_free is charged w_gap per px); 0 = every px of gap is charged
    tall_exit_alt: float = 0.0   # with tall_exit: fixed extra cost of each end hypothesis (the centre reading, if kept by tall_exit_centre, stays free)
    tall_exit_rev: float = 0.0   # with tall_exit: extra cost of the hypothesis that leaves a tall run by the end it was entered from (the trace reverses inside one column, a narrow peak); the opposite end (the stroke is crossed) is free
    tall_exit_centre: bool = False   # with tall_exit: also keep the centre reading as a third hypothesis
    tall_exit: float = 0.0   # > 0: a run taller than tall_exit px is a vertical stroke the trace may leave by either end; the beam keeps two hypotheses per such run (history y = top end or bottom end) instead of one with the centre; 0 = off
    pred_gate_frac: float = 0.0   # > 0: the slope gate tolerance is pred_gate_frac * crop height (px) instead of pred_gate; 0 = pred_gate
    pred_quad_frac: float = 0.0   # > 0: the quadratic slope penalty is pred_quad_frac * (100 d / crop height)^2 (d in px), i.e. in percent of the crop height, instead of pred_quad * d^2; 0 = pred_quad
    pred_quad: float = 0.0   # > 0: extra cost pred_quad * d^2 for the distance d (px) between the run and the slope-predicted row, so a sudden change of slope (stopping or turning in one column) costs much more than the same change spread over several columns; 0 = off (linear cost only)
    pull_grace: int = 0      # > 0: duration-based baseline pull: a path may stay beyond the pull tolerance for this many consecutive columns without paying the pull (a QRS excursion is short; a neighbour's trace or a wrong lock-on stays away); after that the full pull applies; 0 = off
    base_free_v: float = 0.0  # > 0: no baseline pull for a path whose recent slope is at least this many px/column (a fast excursion, e.g. an R wave, is not a drift away from the baseline); 0 = pull always applies
    base_mode: str = "center"  # "center": the crop is cut around the lead's baseline row, so use the crop centre; "est": mode of clean columns


def column_runs(ink: np.ndarray, gap: int = 1):
    """List (per column) of (top, bottom) inclusive ink runs."""
    h, w = ink.shape
    t = np.ascontiguousarray(ink.T)
    idx = np.flatnonzero(t)  # ink pixels ordered by column, then row (same as np.nonzero(ink.T), faster)
    xs = idx // h; ys = idx - xs * h
    out = [[] for _ in range(w)]
    if xs.size == 0:
        return out
    new = np.r_[True, (xs[1:] != xs[:-1]) | (np.diff(ys) > gap + 1)]  # a run starts at a new column or after a gap
    s = np.flatnonzero(new); e = np.r_[s[1:] - 1, xs.size - 1]
    for x, a, b in zip(xs[s].tolist(), ys[s].tolist(), ys[e].tolist()):
        out[x].append((a, b))
    return out


def estimate_baseline(ink: np.ndarray, runs) -> float:
    """Most common y among single-run, short-run columns (where we are almost surely alone)."""
    ys = [(r[0][0] + r[0][1]) / 2 for r in runs if len(r) == 1 and r[0][1] - r[0][0] <= 6]
    if len(ys) < 10:
        ys = [(r[0][0] + r[0][1]) / 2 for r in runs if len(r) >= 1]
    if not ys:
        return ink.shape[0] / 2
    hist, edges = np.histogram(ys, bins=max(8, ink.shape[0] // 4), range=(0, ink.shape[0]))
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def read_y(run, pred, tall, tall_pred=0.0):
    top, bot = run
    if tall_pred and bot - top > tall_pred:
        return float(min(max(pred, top), bot))
    if bot - top <= tall:
        return (top + bot) / 2.0
    return float(min(max(pred, top), bot))


def median_trace(ink):
    h, w = ink.shape
    out = np.full(w, np.nan, np.float32)
    for x in range(w):
        ys = np.flatnonzero(ink[:, x])
        if ys.size:
            out[x] = np.median(ys)
    return out


def prod_cc_trace(ink, dilate_k=9):
    """Production extract_trace standard path: dilate horizontally, keep the largest connected
    component, per-column median of that component (the original, un-dilated ink inside it is not
    used by production either: it takes the dilated component mask)."""
    import cv2
    crop = ink.astype(np.uint8) * 255
    dil = cv2.dilate(crop, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    if n <= 1:
        return np.full(ink.shape[1], np.nan, np.float32)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return median_trace(labels == best)


def _predict(hist, damp, vmax, n=3):
    """Prediction = last y + damp * slope; slope = (newest - oldest) / (points - 1) over the last n recorded y values (n = 3 reproduces the original rule)."""
    h = hist[-n:]
    v = (h[-1] - h[0]) / (len(h) - 1) if len(h) >= 2 else 0.0
    v = float(min(max(v, -vmax), vmax))  # scalar clip without the numpy call overhead
    return h[-1] + damp * v


def track(ink: np.ndarray, p: P, baseline: float | None = None, runs=None, anchors=None):
    """Return (trace_y per column with NaN where skipped, chosen-run index per column)."""
    h, w = ink.shape
    runs = runs if runs is not None else column_runs(ink, p.runs_gap)
    if baseline is None:
        baseline = h / 2.0 if p.base_mode == "center" else estimate_baseline(ink, runs)

    if _FAST_ON and anchors is None:
        fast_path = _fast.track(runs, w, h, baseline, p)
        if fast_path is not None:
            return fast_path
    tol_eff = p.base_tol_frac * h if p.base_tol_frac else p.base_tol
    if p.exit_confirm > 0:   # does a run touch the top / bottom edge in the next exit_confirm columns?
        _top = np.array([any(r[0] <= p.exit_margin for r in rs) for rs in runs] + [False] * (p.exit_confirm + 1)); _bot = np.array([any(r[1] >= h - 1 - p.exit_margin for r in rs) for rs in runs] + [False] * (p.exit_confirm + 1))
        conf_top = [bool(_top[x + 1: x + 1 + p.exit_confirm].any()) for x in range(w)]; conf_bot = [bool(_bot[x + 1: x + 1 + p.exit_confirm].any()) for x in range(w)]
    pg = p.pred_gate_frac * h if p.pred_gate_frac else p.pred_gate; qd = p.pred_quad_frac * (100.0 / h) ** 2 if p.pred_quad_frac else p.pred_quad
    def is_far(run, b): return abs(min(max(b, run[0]), run[1]) - b) > tol_eff
    def unary(run, b=None, x=None, fast=False):
        top, bot = run
        b = baseline if b is None else b
        c = 0.0
        if anchors is not None and p.w_anchor and x is not None and not np.isnan(anchors[x]):
            ay = anchors[x]; c += p.w_anchor * min(0.0 if top <= ay <= bot else min(abs(ay - top), abs(ay - bot)), p.anchor_cap)
        if p.w_base and not fast:
            yc = float(min(max(b, top), bot))  # closest point of the run to the baseline
            c += p.w_base * min(max(abs(yc - b) - (p.base_tol_frac * h if p.base_tol_frac else p.base_tol), 0.0), p.base_cap)
        if p.w_edge and (top <= 0 or bot >= h - 1):
            c += p.w_edge
        return c

    # hypothesis: (cost, y_history(list of recent y), last_run, path(linked list (run idx or -1, parent) so extending it is O(1)))
    hyps = None
    for x in range(w):
        rs = runs[x]
        if not rs:
            if hyps is not None:
                hyps = [(c + p.w_skip, hy + [hy[-1]], lr, (-1, path), bb, ms + 1, mem, nf) for c, hy, lr, path, bb, ms, mem, nf in hyps]
            continue
        if hyps is None:
            hyps = []
            for i, r in enumerate(rs):
                y0 = read_y(r, baseline, p.tall, p.tall_pred)
                hyps.append((unary(r, None, x) + p.init_pull * abs(float(min(max(baseline, r[0]), r[1])) - baseline), [y0], r, (i, None), baseline, 0, [y0], int(is_far(r, baseline))))
            hyps.sort(key=lambda t: t[0])
            hyps = hyps[: p.beam]
            continue
        cand = {}
        for c, hy, lr, path, bb, ms, mem, nf in hyps:
            any_ok = False
            bmed = float(np.median(mem)) if (p.base_mem and len(mem) >= p.ret_min) else None
            ret = float(np.median(mem)) if (p.ret_len and ms >= p.ret_after and ms > 0 and len(mem) >= p.ret_min) else None
            pred = _predict(hy, p.damp, p.vmax, p.cost_slope_pts or p.slope_pts)
            fast = (p.pull_grace > 0 and nf < p.pull_grace) or p.base_free_v > 0 and len(hy) >= 2 and abs(hy[-1] - hy[-min(len(hy), p.slope_pts)]) / (min(len(hy), p.slope_pts) - 1) >= p.base_free_v
            gpred = _predict(hy, p.damp if p.gate_damp < 0 else p.gate_damp, p.vmax, p.slope_pts) if p.gate_mode != "gap" else pred
            for i, r in enumerate(rs):
                top, bot = r
                pick = min(max(pred, top), bot)
                if p.jump_gate and abs(pick - pred) > p.jump_gate * (1 + ms / p.gate_relax): continue
                via_ret = False
                gap = max(0, max(top, lr[0]) - min(bot, lr[1]))
                if p.nofit_near > 0 and gap <= p.nofit_near: any_ok = True
                if p.gate_mode == "gap":
                    if p.gap_gate and gap > p.gap_gate * (1 + ms / p.gate_relax): continue
                else:
                    wd = 1 + ms / p.gate_relax; near_prev = gap <= p.gap_gate * wd
                    dpred = 0.0 if top <= gpred <= bot else min(abs(gpred - top), abs(gpred - bot)); near_pred = dpred <= pg * wd
                    ok = (near_prev or near_pred) if p.gate_mode == "or" else (near_prev and near_pred) if p.gate_mode == "and" else near_pred
                    if not ok:
                        if ret is None: continue
                        if (0.0 if top <= ret <= bot else min(abs(ret - top), abs(ret - bot))) > p.ret_gate: continue
                        via_ret = True
                any_ok = True
                nc = (c + p.ret_cost + p.w_pred * abs(min(max(ret, top), bot) - ret) + unary(r, bb if bmed is None else bmed, x, fast)) if via_ret else (c + p.w_pred * abs(pick - pred) + qd * (pick - pred) ** 2 + p.w_gap * max(0.0, gap - p.gap_free) + unary(r, bb if bmed is None else bmed, x, fast))
                if p.tall_exit and bot - top > p.tall_exit:  # a tall run: one hypothesis per exit end (and optionally the centre)
                    near = 0 if abs(hy[-1] - top) <= abs(hy[-1] - bot) else 1
                    ys = [((i, 0), float(top), p.tall_exit_alt + (p.tall_exit_rev if near == 0 else 0.0)), ((i, 1), float(bot), p.tall_exit_alt + (p.tall_exit_rev if near == 1 else 0.0))] + ([((i, 2), read_y(r, pred, p.tall, p.tall_pred), 0.0)] if p.tall_exit_centre else [])
                else: ys = [(i, read_y(r, pred, p.tall, p.tall_pred), 0.0)]
                for key, y, ex in ys:  # key: recombine hypotheses that end on the same run (Viterbi merge)
                    if key not in cand or nc + ex < cand[key][0]:
                        cand[key] = (nc + ex, [y] if via_ret else hy[-7:] + [y], r, (i, path), bb + p.base_adapt * (y - bb), 0, (mem[-max(p.ret_len, p.base_mem):] + [y]) if (p.ret_len or p.base_mem) else mem, (nf + 1) if is_far(r, bb) else 0)
            # option: no run fits at all (e.g. ink occluded) -> skip, keep prediction
            sk = p.skip_cost if (p.jump_gate or p.gap_gate or pg) else p.w_skip
            if p.skip_cost_nofit >= 0 and not any_ok: sk = min(sk, p.skip_cost_nofit)
            if p.exit_skip_cost >= 0 and len(hy) >= 2 and p.exit_confirm > 0:
                on_top, on_bot = lr[0] <= p.exit_margin, lr[1] >= h - 1 - p.exit_margin; dy = hy[-1] - hy[-2]
                if ms == 0: free = (on_bot and dy > 0 and conf_bot[x]) or (on_top and dy < 0 and conf_top[x])
                else: free = 0 < ms <= p.exit_hold and ((on_top and conf_top[x]) or (on_bot and conf_bot[x]))
                if free: sk = min(sk, p.exit_skip_cost)
            elif p.exit_skip_cost >= 0 and len(hy) >= 2:
                dy = hy[-1] - hy[-2]
                if (lr[1] >= h - 1 - p.exit_margin and dy > 0) or (lr[0] <= p.exit_margin and dy < 0) or (0 < ms <= p.exit_hold and (lr[1] >= h - 1 - p.exit_margin or lr[0] <= p.exit_margin) and (p.exit_hold_speed <= 0 or (len(hy) > ms + 1 and abs(hy[-1 - ms] - hy[-2 - ms]) >= p.exit_hold_speed))): sk = min(sk, p.exit_skip_cost)
            nc = c + sk
            key = (-1, lr) if (p.skip_key_run or (p.skip_key_edge and (lr[0] <= p.exit_margin or lr[1] >= h - 1 - p.exit_margin))) else -1
            if key not in cand or nc < cand[key][0]:
                cand[key] = (nc, hy[-7:] + [hy[-1] if (p.jump_gate or p.gap_gate or pg) else pred], lr, (-1, path), bb, ms + 1, mem, nf)
        hyps = sorted(cand.values(), key=lambda t: t[0])[: p.beam]
        # fill paths for columns skipped for lack of runs are already appended above
    if hyps is None:
        return np.full(w, np.nan, np.float32), np.full(w, -1)
    best = hyps[0]
    node = best[3]; path = []
    while node is not None: path.append(node[0]); node = node[1]
    path.reverse()
    # path has one entry per column from the first inked column onward; left-pad
    first = next(i for i, r in enumerate(runs) if r)
    # (columns with no runs append -1 entries too, so lengths line up)
    path = [-1] * first + path
    path = path[:w] + [-1] * max(0, w - len(path))
    return path


def path_to_trace(ink, runs, path, p: P):
    """Convert a chosen-run path into y values. Tall runs are read by momentum."""
    w = ink.shape[1]
    y = np.full(w, np.nan, np.float32)
    hist = []
    for x in range(w):
        i = path[x]
        if i < 0 or not runs[x]:
            continue
        pred = _predict(hist, p.damp, p.vmax, p.slope_pts) if hist else (runs[x][i][0] + runs[x][i][1]) / 2
        v = read_y(runs[x][i], pred, p.tall, p.tall_pred)
        y[x] = v
        hist.append(v)
    return y


def tracker_trace(ink, p: P, baseline=None, anchors=None):
    runs = column_runs(ink, p.runs_gap)
    if baseline is None:
        baseline = ink.shape[0] / 2.0 if p.base_mode == "center" else estimate_baseline(ink, runs)
    path = track(ink, p, baseline, runs, anchors)
    y = path_to_trace(ink, runs, path, p)
    if p.peak_read > 0:
        c = y.copy()
        for x in range(1, len(y) - 1):
            i = path[x]
            if i < 0 or not runs[x] or np.isnan(c[x - 1]) or np.isnan(c[x + 1]) or np.isnan(c[x]): continue
            top, bot = runs[x][i]
            if bot - top < p.peak_read: continue
            if c[x - 1] > c[x] and c[x + 1] > c[x]: y[x] = top      # both neighbours lower on the page: a peak, the tip is the top end
            elif c[x - 1] < c[x] and c[x + 1] < c[x]: y[x] = bot    # both neighbours higher on the page: a trough, the tip is the bottom end
    return y, path, runs


# ---- pipeline entry point ----
import json as _json
from dataclasses import replace as _replace
from pathlib import Path as _Path

def default_params() -> P:
    """Tracker parameters of the frozen default (overlap_tracker_settings.json = tuned_settings_v3 + default_tracker_settings)."""
    return P(**_json.load(open(_Path(__file__).with_name("overlap_tracker_settings.json"))))

_DEFAULT = None
def trace(ink: np.ndarray) -> np.ndarray:
    """y (crop rows) of the lead in each column of a binary crop (ink > 0); NaN where the tracker skipped a column."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = default_params()
    if _FAST_ON:
        y = _fast.trace_arrays(ink, _DEFAULT)
        if y is not None:
            return y
    return tracker_trace(ink > 0, _DEFAULT)[0]
