"""Lead-label detection by template matching (classical, no learning). Own code.

    px_per_mm = estimate_px_per_mm(img_bgr)                       from the printed grid
    dets = detect_labels(ink, px_per_mm, font="mono")             [{"name", "x0", "y0", "x1", "y1", "score"}], longest strings first, overlaps suppressed
Templates are the lead names ("I", "II", "III", "aVR", "aVL", "aVF", "V1".."V6", "-aVR") drawn in a reference font at the size printed on our scans
(about 4.2 mm font size, 3 mm cap height) and matched at several scales; the font of a real scan will differ, which is what the cross-font tests measure.
"""
from __future__ import annotations
from pathlib import Path
import cv2, matplotlib, numpy as np
from PIL import Image, ImageDraw, ImageFont

NAMES = ["III", "aVR", "aVL", "aVF", "-aVR", "II", "I", "V1", "V2", "V3", "V4", "V5", "V6"]
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONTS = {"mono": FONT_DIR / "DejaVuSansMono.ttf", "sans": FONT_DIR / "DejaVuSans.ttf", "serif": FONT_DIR / "DejaVuSerif.ttf", "courier": Path("/usr/share/fonts/urw-base35/NimbusMonoPS-Regular.otf"),
         "texgyre": Path("/usr/share/fonts/tex-gyre/texgyrecursor-regular.otf"), "liberation": Path("/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf")}


def estimate_px_per_mm(img_bgr: np.ndarray) -> float:
    """Period of the 1 mm grid from the autocorrelation of the column-mean green channel (grid lines are darker than the paper in G)."""
    g = img_bgr[:, :, 1].astype(np.float32)
    col = np.median(g, axis=0); col = col - col.mean()                     # median over rows suppresses trace ink
    ac = np.correlate(col, col, "full")[len(col) - 1:]; ac /= ac[0] + 1e-9
    lo, hi = 4, int(0.04 * len(col))
    pk = [i for i in range(lo, hi) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.1]
    if not pk: return float("nan")
    # the first strong peak is the 1 mm period (5 mm lines are harmonics at 5x); refine with the mean of the first few multiples
    p1 = pk[0]; mult = [(k, min(pk, key=lambda x: abs(x - k * p1))) for k in (1, 2, 3, 4)]
    return float(np.mean([m / k for k, m in mult if abs(m - k * p1) <= 0.1 * k * p1]))


_CACHE = {}
def _template(name: str, font: str, px: int) -> np.ndarray:
    key = (name, font, px)
    if key not in _CACHE:
        f = ImageFont.truetype(str(FONTS[font]), px)
        im = Image.new("L", (int(f.getlength(name)) + 8, px + 8), 255); ImageDraw.Draw(im).text((4, 2), name, font=f, fill=0)
        a = 255 - np.asarray(im); ys, xs = np.where(a > 80)
        _CACHE[key] = (a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1].astype(np.float32) / 255.0)
    return _CACHE[key]


THR = {"I": 0.88, "II": 0.82}          # short names are easy to confuse with plain vertical strokes (QRS, ticks): stricter threshold
THR_DEFAULT = 0.72


def detect_labels(ink: np.ndarray, px_per_mm: float, font: str = "mono", scales=(0.85, 1.0, 1.15), density=(0.7, 1.35), blur: float = 0.0):
    """All templates are matched, then accepted globally in order of score; a candidate is rejected when it overlaps an accepted one, or when the ink inside its box
    is far from the template's ink amount (a trace crossing the text, or a stroke that is not text)."""
    img = ink.astype(np.float32); cands = []
    if blur > 0: img_m = cv2.GaussianBlur(img, (0, 0), blur)
    else: img_m = img
    for name in NAMES:
        for sc in scales:
            t = _template(name, font, max(8, int(round(4.2 * px_per_mm * sc))))
            if t.shape[0] >= img.shape[0] or t.shape[1] >= img.shape[1]: continue
            tm = cv2.GaussianBlur(t, (0, 0), blur) if blur > 0 else t
            r = cv2.matchTemplate(img_m, tm, cv2.TM_CCOEFF_NORMED); thr = THR.get(name, THR_DEFAULT)
            ys, xs = np.where(r >= thr)
            tp = float((t > 0.3).sum())
            for y, x in zip(ys, xs):
                cands.append((float(r[y, x]), name, int(x), int(y), t.shape[1], t.shape[0], tp))
    cands.sort(reverse=True); dets = []
    for sc_, name, x, y, w, h, tp in cands:
        box = (x, y, x + w, y + h)
        if any(_overlap(box, d) >= 0.3 for d in dets): continue
        ratio = float(img[y:y + h, x:x + w].sum()) / max(1.0, tp)
        if not (density[0] <= ratio <= density[1]): continue
        dets.append({"name": name, "x0": x, "y0": y, "x1": x + w, "y1": y + h, "score": sc_})
    return dets


def _overlap(b, d):
    ix = max(0, min(b[2], d["x1"]) - max(b[0], d["x0"])); iy = max(0, min(b[3], d["y1"]) - max(b[1], d["y0"]))
    return ix * iy / max(1, min((b[2] - b[0]) * (b[3] - b[1]), (d["x1"] - d["x0"]) * (d["y1"] - d["y0"])))
