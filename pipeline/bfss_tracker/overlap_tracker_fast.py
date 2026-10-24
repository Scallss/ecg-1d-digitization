"""Compiled (numba) version of overlap_tracker.track for the frozen default settings. Same beam search, same costs in the same order of floating point operations, same tie breaking
(insertion order of the candidates, stable sort by cost), so the chosen path is identical to the Python version; anything it does not support falls back to the Python code.

    path = track(runs, w, h, baseline, p)    # list of run indices (-1 = skipped column) like overlap_tracker.track, or None when this settings object / crop is not supported

Supported: the options of overlap_tracker_settings.json (gate_mode "pred", skip_key_edge, tall_exit with centre, exit_skip_cost without exit_confirm, base_free_v, init_pull, pred_quad(_frac), gap_free).
Not supported (-> None): anchors, jump_gate, gap_gate, ret_len, base_mem, base_adapt, pull_grace, exit_confirm, exit_hold, nofit_near, skip_key_run, skip_cost_nofit, tall_exit_rev, tall_pred,
cost_slope_pts, slope_pts outside 2..4, gate_mode other than "pred", a crop without any ink run.
"""
from __future__ import annotations
import math
import cv2
import numpy as np

try:
    import numba as nb
except Exception:  # numba not installed: the Python tracker is used
    nb = None

AVAILABLE = nb is not None


def supported(p, anchors=None) -> bool:
    return (anchors is None and p.gate_mode == "pred" and p.jump_gate == 0 and p.gap_gate == 0 and p.ret_len == 0 and p.base_mem == 0 and p.base_adapt == 0 and p.pull_grace == 0
            and p.exit_confirm == 0 and p.exit_hold == 0 and p.nofit_near == 0 and not p.skip_key_run and p.skip_cost_nofit < 0 and p.tall_exit_rev == 0 and p.tall_pred == 0
            and p.cost_slope_pts == 0 and 2 <= p.slope_pts <= 4 and p.w_anchor == 0 and p.beam >= 1 and p.exit_skip_cost >= 0 and p.w_skip >= 0)


def _make_kernel():
    @nb.njit(cache=True)
    def predict(hyv, k, n_hy, damp, vmax, n):
        m = min(n_hy, n)
        newest = hyv[k, 3]
        v = 0.0
        if m >= 2:
            v = (newest - hyv[k, 4 - m]) / (m - 1)
        v = min(max(v, -vmax), vmax)
        return newest + damp * v

    @nb.njit(cache=True)
    def predict_hv(hv, nv, damp, vmax, n):
        m = min(nv, n)
        v = 0.0
        if m >= 2:
            v = (hv[nv - 1] - hv[nv - m]) / (m - 1)
        v = min(max(v, -vmax), vmax)
        return hv[nv - 1] + damp * v

    @nb.njit(cache=True)
    def unary(top, bot, b, fast, w_base, tol_eff, base_cap, w_edge, hf):
        c = 0.0
        if w_base != 0.0 and not fast:
            yc = min(max(b, float(top)), float(bot))
            c += w_base * min(max(abs(yc - b) - tol_eff, 0.0), base_cap)
        if w_edge != 0.0 and (top <= 0 or bot >= hf - 1):
            c += w_edge
        return c

    @nb.njit(cache=True)
    def read_y(top, bot, pred, tall):
        if bot - top <= tall:
            return (top + bot) / 2.0
        return min(max(pred, float(top)), float(bot))

    @nb.njit(cache=True)
    def sort_order(cost, n):
        order = np.arange(n)
        for a in range(1, n):  # stable insertion sort by cost
            t = order[a]; b = a - 1
            while b >= 0 and cost[order[b]] > cost[t]:
                order[b + 1] = order[b]; b -= 1
            order[b + 1] = t
        return order

    @nb.njit(cache=True)
    def kernel(tops, bots, cnt, off, w, hf, baseline, beam, w_pred, w_gap, w_base, tol_eff, base_cap, w_edge, w_skip, damp, vmax, slope_pts, gate_relax, pg, qd, gap_free,
               skip_cost, exit_skip_cost, exit_margin, base_free_v, init_pull, tall, tall_exit, tall_exit_alt, tall_exit_centre, skip_key_edge, gd, two, maxr):
        K = 3 * maxr + beam + 4
        cc = np.zeros(beam); hyv = np.zeros((beam, 4)); hyn = np.zeros(beam, np.int64); ltop = np.zeros(beam, np.int64); lbot = np.zeros(beam, np.int64)
        node = np.zeros(beam, np.int64); ms = np.zeros(beam, np.int64)
        ncc = np.zeros(K); chy = np.zeros((K, 4)); chn = np.zeros(K, np.int64); ctop = np.zeros(K, np.int64); cbot = np.zeros(K, np.int64)
        cpar = np.zeros(K, np.int64); cval = np.zeros(K, np.int64); cms = np.zeros(K, np.int64); ckey = np.zeros(K, np.int64)
        nodeval = np.zeros(w * beam + beam + 8, np.int64); nodepar = np.zeros(w * beam + beam + 8, np.int64); nn = 0
        nh = 0; started = False; first = -1
        use_skip = pg != 0.0
        for x in range(w):
            nr = cnt[x]; r0 = off[x]
            if nr == 0:
                if started:
                    for k in range(nh):
                        cc[k] += w_skip
                        for j in range(3): hyv[k, j] = hyv[k, j + 1]
                        hyn[k] = min(hyn[k] + 1, 4)
                        nodeval[nn] = -1; nodepar[nn] = node[k]; node[k] = nn; nn += 1
                        ms[k] += 1
                continue
            if not started:
                started = True; first = x; n = 0
                for i in range(nr):
                    top = tops[r0 + i]; bot = bots[r0 + i]; y0 = (top + bot) / 2.0
                    ncc[n] = unary(top, bot, baseline, False, w_base, tol_eff, base_cap, w_edge, hf) + init_pull * abs(min(max(baseline, float(top)), float(bot)) - baseline)
                    chy[n, 3] = y0; chn[n] = 1; ctop[n] = top; cbot[n] = bot; cval[n] = i; n += 1
                order = sort_order(ncc, n); nh = min(n, beam)
                for a in range(nh):
                    s = order[a]; cc[a] = ncc[s]; hyv[a, 3] = chy[s, 3]; hyn[a] = 1; ltop[a] = ctop[s]; lbot[a] = cbot[s]; ms[a] = 0
                    nodeval[nn] = cval[s]; nodepar[nn] = -1; node[a] = nn; nn += 1
                continue
            ncand = 0
            for k in range(nh):
                c = cc[k]; n_hy = hyn[k]; lt = ltop[k]; lb = lbot[k]; msk = ms[k]
                pred = predict(hyv, k, n_hy, damp, vmax, slope_pts)
                fast = False
                if base_free_v > 0.0 and n_hy >= 2:
                    m = min(n_hy, slope_pts)
                    fast = abs(hyv[k, 3] - hyv[k, 4 - m]) / (m - 1) >= base_free_v
                gpred = predict(hyv, k, n_hy, gd, vmax, slope_pts)
                wd = 1 + msk / gate_relax
                for i in range(nr):
                    top = tops[r0 + i]; bot = bots[r0 + i]
                    dpred = 0.0
                    if not (top <= gpred <= bot):
                        dpred = min(abs(gpred - top), abs(gpred - bot))
                    if not (dpred <= pg * wd):
                        continue
                    pick = min(max(pred, float(top)), float(bot))
                    gap = max(0, max(top, lt) - min(bot, lb))
                    nc = c + w_pred * abs(pick - pred) + qd * math.pow(pick - pred, two) + w_gap * max(0.0, gap - gap_free) + unary(top, bot, baseline, fast, w_base, tol_eff, base_cap, w_edge, hf)
                    if tall_exit > 0.0 and bot - top > tall_exit:
                        ne = 3 if tall_exit_centre else 2
                    else:
                        ne = 1
                    for e in range(ne):
                        if ne == 1:
                            key = i * 4 + 3; y = read_y(top, bot, pred, tall); ex = 0.0
                        elif e == 0:
                            key = i * 4; y = float(top); ex = tall_exit_alt + 0.0
                        elif e == 1:
                            key = i * 4 + 1; y = float(bot); ex = tall_exit_alt + 0.0
                        else:
                            key = i * 4 + 2; y = read_y(top, bot, pred, tall); ex = 0.0
                        tot = nc + ex
                        slot = -1
                        for q in range(ncand):
                            if ckey[q] == key: slot = q; break
                        if slot < 0 or tot < ncc[slot]:
                            if slot < 0:
                                slot = ncand; ncand += 1
                            ncc[slot] = tot; ckey[slot] = key
                            for j in range(3): chy[slot, j] = hyv[k, j + 1]
                            chy[slot, 3] = y; chn[slot] = min(n_hy + 1, 4)
                            ctop[slot] = top; cbot[slot] = bot; cpar[slot] = node[k]; cval[slot] = i; cms[slot] = 0
                sk = skip_cost if use_skip else w_skip
                if exit_skip_cost >= 0.0 and n_hy >= 2:
                    dy = hyv[k, 3] - hyv[k, 2]
                    if (lb >= hf - 1 - exit_margin and dy > 0) or (lt <= exit_margin and dy < 0):
                        sk = min(sk, exit_skip_cost)
                nc = c + sk
                edge = lt <= exit_margin or lb >= hf - 1 - exit_margin
                if skip_key_edge and edge:
                    key = -2 - (lt * 100000 + lb)
                else:
                    key = -1
                slot = -1
                for q in range(ncand):
                    if ckey[q] == key: slot = q; break
                if slot < 0 or nc < ncc[slot]:
                    if slot < 0:
                        slot = ncand; ncand += 1
                    ncc[slot] = nc; ckey[slot] = key
                    for j in range(3): chy[slot, j] = hyv[k, j + 1]
                    chy[slot, 3] = hyv[k, 3] if use_skip else pred
                    chn[slot] = min(n_hy + 1, 4); ctop[slot] = lt; cbot[slot] = lb; cpar[slot] = node[k]; cval[slot] = -1; cms[slot] = msk + 1
            order = sort_order(ncc, ncand); nh = min(ncand, beam)
            for a in range(nh):
                s = order[a]; cc[a] = ncc[s]
                for j in range(4): hyv[a, j] = chy[s, j]
                hyn[a] = chn[s]; ltop[a] = ctop[s]; lbot[a] = cbot[s]; ms[a] = cms[s]
                nodeval[nn] = cval[s]; nodepar[nn] = cpar[s]; node[a] = nn; nn += 1
        path = np.full(w, -1, np.int64)
        if not started:
            return path, -1
        L = w - first; nd = node[0]; pos = w - 1
        while nd >= 0 and pos >= first:
            path[pos] = nodeval[nd]; nd = nodepar[nd]; pos -= 1
        return path, first

    @nb.njit(cache=True)
    def ytrace(path, cnt, off, tops, bots, w, damp, vmax, slope_pts, tall):
        # overlap_tracker.path_to_trace: tall runs are read by momentum, the history keeps the unrounded values
        y = np.full(w, np.nan, np.float32); hv = np.zeros(w); nv = 0
        for x in range(w):
            i = path[x]
            if i < 0 or cnt[x] == 0:
                continue
            top = tops[off[x] + i]; bot = bots[off[x] + i]
            if nv > 0:
                pred = predict_hv(hv, nv, damp, vmax, slope_pts)
            else:
                pred = (top + bot) / 2
            if bot - top <= tall:
                val = (top + bot) / 2.0
            else:
                val = min(max(pred, float(top)), float(bot))
            y[x] = val; hv[nv] = val; nv += 1
        return y
    return kernel, ytrace


_KERNEL = None
_YTRACE = None


def _args(p, h):
    tol_eff = p.base_tol_frac * h if p.base_tol_frac else p.base_tol
    pg = p.pred_gate_frac * h if p.pred_gate_frac else p.pred_gate
    qd = p.pred_quad_frac * (100.0 / h) ** 2 if p.pred_quad_frac else p.pred_quad
    gd = p.damp if p.gate_damp < 0 else p.gate_damp
    return tol_eff, pg, qd, gd


def _run_kernel(cnt, off, tops, bots, w, h, baseline, p):
    global _KERNEL, _YTRACE
    if _KERNEL is None:
        _KERNEL, _YTRACE = _make_kernel()
    tol_eff, pg, qd, gd = _args(p, h)
    path, first = _KERNEL(tops, bots, cnt, off, int(w), float(h), float(baseline), int(p.beam), float(p.w_pred), float(p.w_gap), float(p.w_base), float(tol_eff), float(p.base_cap), float(p.w_edge),
                          float(p.w_skip), float(p.damp), float(p.vmax), int(p.slope_pts), float(p.gate_relax), float(pg), float(qd), float(p.gap_free), float(p.skip_cost),
                          float(p.exit_skip_cost), float(p.exit_margin), float(p.base_free_v), float(p.init_pull), float(p.tall), float(p.tall_exit), float(p.tall_exit_alt), bool(p.tall_exit_centre),
                          bool(p.skip_key_edge), float(gd), 2.0, int(cnt.max()))
    return path


def track(runs, w, h, baseline, p):
    if nb is None or not supported(p):
        return None
    cnt = np.fromiter((len(r) for r in runs), np.int64, count=w)
    if not cnt.any():
        return None
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
    flat = [t for r in runs for t in r]
    tops = np.array([t[0] for t in flat], np.int64); bots = np.array([t[1] for t in flat], np.int64)
    return _run_kernel(cnt, off, tops, bots, w, h, baseline, p).tolist()


def trace_arrays(ink, p):
    """y per column of a binary crop (uint8 or bool) (column_runs + track + path_to_trace for baseline_mode "center" and no peak_read), or None when not supported / no ink at all."""
    if nb is None or not supported(p) or p.peak_read > 0 or p.base_mode != "center":
        return None
    h, w = ink.shape
    t = cv2.transpose(ink) if ink.dtype == np.uint8 else np.ascontiguousarray(ink.T)
    idx = np.flatnonzero(t if t.dtype == bool else t > 0)  # ink pixels ordered by column, then row
    if idx.size == 0:
        return None
    xs = idx // h; ys = idx - xs * h
    new = np.r_[True, (xs[1:] != xs[:-1]) | (np.diff(ys) > p.runs_gap + 1)]  # a run starts at a new column or after a gap
    s = np.flatnonzero(new); e = np.r_[s[1:] - 1, xs.size - 1]
    cnt = np.bincount(xs[s], minlength=w).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
    tops = ys[s].astype(np.int64); bots = ys[e].astype(np.int64)
    path = _run_kernel(cnt, off, tops, bots, w, h, h / 2.0, p)
    return _YTRACE(path, cnt, off, tops, bots, int(w), float(p.damp), float(p.vmax), int(p.slope_pts), float(p.tall))
