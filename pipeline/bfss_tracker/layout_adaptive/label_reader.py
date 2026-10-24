"""Read lead-label names from word crops by template matching with templates harvested from real scans (classical, no learning). Own code.

    bank = harvest(scans)                       # templates per name from scans whose layout is known (positions give the names)
    name, score = read_word(ink, word, ppm, bank, n_glyphs)
    names = assign_unique(sim_matrix)           # Hungarian assignment so that every lead is used once

Crops are rescaled to 10 px/mm, blurred (sigma 1 px) and compared by normalised cross-correlation with the templates of the names that have the same number of glyphs
(I=1; II, V1..V6=2; III, aVR, aVL, aVF=3; -aVR=4), allowing small shifts.
"""
from __future__ import annotations
import cv2, numpy as np

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "-aVR"]
NGLYPH = {n: len(n) for n in NAMES}
CANVAS = (56, 110)            # h, w at 10 px/mm
SIGMA = 1.0


def word_image(ink: np.ndarray, w: dict, ppm: float, pad: int = 2) -> np.ndarray:
    crop = ink[max(0, w["y0"] - pad): w["y1"] + pad + 1, max(0, w["x0"] - pad): w["x1"] + pad + 1].astype(np.float32)
    f = 10.0 / ppm
    crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(crop, (0, 0), SIGMA)


def to_canvas(img: np.ndarray) -> np.ndarray:
    c = np.zeros(CANVAS, np.float32); h, w = min(img.shape[0], CANVAS[0]), min(img.shape[1], CANVAS[1]); c[:h, :w] = img[:h, :w]; return c


def build_bank(samples: dict) -> dict:
    """samples: name -> list of word images (already scaled). Template = mean of the crops placed on a canvas (top-left aligned)."""
    return {n: np.mean([to_canvas(i) for i in imgs], 0) for n, imgs in samples.items() if imgs}


def similarity(img: np.ndarray, tpl: np.ndarray, shift: int = 3) -> float:
    a = to_canvas(img); best = -1.0
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            b = np.roll(np.roll(tpl, dy, 0), dx, 1)
            va = a - a.mean(); vb = b - b.mean(); d = np.linalg.norm(va) * np.linalg.norm(vb)
            if d > 0: best = max(best, float((va * vb).sum() / d))
    return best


def scores(ink, word, ppm, bank, n_glyphs=None):
    img = word_image(ink, word, ppm)
    return {n: similarity(img, t) for n, t in bank.items() if n_glyphs is None or NGLYPH[n] == n_glyphs}


def read_word(ink, word, ppm, bank, n_glyphs=None):
    sc = scores(ink, word, ppm, bank, n_glyphs)
    if not sc: return None, -1.0
    n = max(sc, key=sc.get); return n, sc[n]


def assign_unique(sim: np.ndarray):
    """sim[i, j] = similarity of slot i to name j (columns = the names to place); returns the best one-to-one assignment (indices of names)."""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-sim); return c


def tight(tpl: np.ndarray) -> np.ndarray:
    ys, xs = np.where(tpl > 0.05 * tpl.max()); return tpl[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def slot_scores(ink: np.ndarray, ppm: float, slot_xy, bank: dict, half_mm: float = 3.0):
    """Similarity of every name's template with the ink inside the slot window (no word box needed): max normalised cross-correlation over the window, so a trace stroke
    crossing part of the label does not break a box. The image is brought to 10 px/mm first."""
    f = 10.0 / ppm; x, y = slot_xy; hw = int(half_mm * ppm)
    out = {}
    for n, t in bank.items():
        tt = tight(t); th, tw = tt.shape
        x0 = int(max(0, x - hw)); y0 = int(max(0, y - th / f / 2 - hw)); x1 = int(min(ink.shape[1], x + tw / f + hw)); y1 = int(min(ink.shape[0], y + th / f / 2 + hw))
        win = cv2.resize(ink[y0:y1, x0:x1].astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_AREA); win = cv2.GaussianBlur(win, (0, 0), SIGMA)
        if win.shape[0] < th or win.shape[1] < tw: out[n] = -1.0; continue
        out[n] = float(cv2.matchTemplate(win, tt, cv2.TM_CCOEFF_NORMED).max())
    return out
