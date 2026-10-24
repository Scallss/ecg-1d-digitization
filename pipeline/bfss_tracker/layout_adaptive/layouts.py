"""Layout catalogue for synthetic 12-lead ECG sheets (our own code; the survey behind it is ../LAYOUTS.md).

A layout is a list of pages; a page is a list of rows; a row is a list of segments. A segment is one lead drawn over one slice of the page's row width:
    Seg(lead, col, ncols, shifted)   col of ncols equal slices; `shifted` = the slice shows the matching slice of the recording (time-shifted print),
                                     otherwise it shows the recording from 0 s (simultaneous print / rhythm strip).
`lead` is a label such as "aVL" or "-aVR" (the sign inverts the signal, as Cabrera prints -aVR).
"""
from __future__ import annotations
from dataclasses import dataclass

STANDARD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CABRERA = ["aVL", "I", "-aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
SOURCE = {"I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5, "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11}  # PTB-XL channel order
RHYTHM = {0: [], 1: ["II"], 2: ["II", "V5"], 3: ["II", "V1", "V5"]}


@dataclass(frozen=True)
class Seg:
    lead: str
    col: int = 0
    ncols: int = 1
    shifted: bool = True

    @property
    def name(self) -> str:      # signal name without the sign
        return self.lead.lstrip("-")

    @property
    def sign(self) -> int:
        return -1 if self.lead.startswith("-") else 1


@dataclass(frozen=True)
class Layout:
    name: str
    pages: tuple            # tuple[page], page = tuple[row], row = tuple[Seg]
    limb_order: str = "standard"
    arrangement: str = ""
    rhythm: int = 0
    time_mode: str = "shifted"

    def leads(self) -> list:
        return [s.lead for p in self.pages for r in p for s in r]


def _rhythm_rows(n: int) -> list:
    return [(Seg(l, 0, 1, False),) for l in RHYTHM[n]]


def build(arrangement: str, order: str = "standard", rhythm: int = 0, time_mode: str = "shifted") -> Layout:
    """arrangement: 3x4, 6x2, 12x1, 4x3g (one page each); order: standard | cabrera; rhythm: 0-3 long strips (single-page layouts only)."""
    L = STANDARD if order == "standard" else CABRERA
    shifted = time_mode == "shifted"
    if arrangement == "3x4":            # 3 rows x 4 columns, columns hold consecutive lead triples
        rows = [tuple(Seg(L[3 * c + r], c, 4, shifted) for c in range(4)) for r in range(3)]
        pages = [tuple(rows) + tuple(_rhythm_rows(rhythm))]
    elif arrangement == "6x2":          # left column = limb leads, right column = V1-V6
        rows = [tuple(Seg(L[r + 6 * c], c, 2, shifted) for c in range(2)) for r in range(6)]
        pages = [tuple(rows) + tuple(_rhythm_rows(rhythm))]
    elif arrangement == "12x1":
        pages = [tuple((Seg(l, 0, 1, True),) for l in L) + tuple(_rhythm_rows(rhythm))]
    elif arrangement == "4x3g":         # 4 rows x 3 columns, one row per lead group
        rows = [tuple(Seg(L[3 * r + c], c, 3, shifted) for c in range(3)) for r in range(4)]
        pages = [tuple(rows) + tuple(_rhythm_rows(rhythm))]
    else:
        raise ValueError(arrangement)
    tag = f"{arrangement}{'+%d' % rhythm if rhythm else ''}_{'cab' if order == 'cabrera' else 'std'}{'' if shifted else '_sim'}"
    return Layout(tag, tuple(pages), order, arrangement, rhythm, time_mode)


def catalogue() -> dict:
    """The layouts we generate (see ../LAYOUTS.md)."""
    out = {}
    for order in ("standard", "cabrera"):
        for a, rs in (("3x4", (0, 1, 3)), ("6x2", (0, 1, 3)), ("12x1", (0, 1)), ("4x3g", (0,))):
            for r in rs:
                lay = build(a, order, r); out[lay.name] = lay
        for a in ("3x4", "6x2"):
            lay = build(a, order, 0, "simultaneous"); out[lay.name] = lay
    return out


if __name__ == "__main__":
    for n, l in catalogue().items():
        print(f"{n:22s} pages {len(l.pages)} rows/page {[len(p) for p in l.pages]} leads {len(l.leads())}")
