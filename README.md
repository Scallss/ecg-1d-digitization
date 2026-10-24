# ECG Digitization

Converts scanned 12-lead paper ECGs into calibrated digital waveforms (`.npz`), and records where the signal is unreliable
via two aligned artifact masks:

- `overlap_mask`: samples affected by a detected overlap anomaly (traces crossing text or each other)
- `interpolated_mask`: samples inferred by explicit recovery logic rather than observed

Its outputs feed the downstream mitral valve classifier in `../mitral_classification/`. For the research overview see
`../RESEARCH_INDEX.md`. Some paths below (`experiments/`, `docs/`, `benchmarks/`, `archive/`, annotation logs) exist only in the local
workspace: this repository's `.gitignore` excludes them.

## Quick start

The workspace uses one shared uv environment at the workspace root (`../pyproject.toml`): run `uv sync` once, then

```bash
# from the workspace root (MLCV/)
uv run python ecg_digitization/pipeline/extract_all_npz.py --dataset mitral_images --limit 10 --output-dir /tmp/test_out
```

Single image from Python:

```python
import sys; sys.path.insert(0, "ecg_digitization/pipeline")
import ecg_digitizer as ed
params = ed.get_params("black165_stable_sep")      # or "original"
result = ed.digitize_ecg_image("path/to/scan.png", params)
```

## Extraction versions

Three extraction versions, each a self-contained copy of the digitizer in its own folder (same downstream code, same `.npz` format). Output of the older versions is reproduced bit for bit (checked on 40 scans each; see "Checks").

| Version | Folder / preset | Ink mask and separators | Lead trace |
|---|---|---|---|
| v1 | `pipeline/original/`, preset `original` (also the canonical `pipeline/ecg_digitizer.py`) | Red-grid whitening + dark-hybrid mask; morphology-based column separators | per-column median of the selected component |
| v2 | `pipeline/bfss/`, preset `black165_stable_sep_repair` (default); `black165_stable_sep` without the repair | Darkness threshold (HSV V <= 165) + per-image redness gate + border-line stripper; cross-row stability separator detector, with repair of a missed or wrong separator | per-column median |
| v3 | `pipeline/bfss_tracker/`, preset `black165_stable_sep_repair_tracker` | as v2 | column-wise beam-search tracker: follows the trace through overlapping strokes and marks the samples it had to interpolate |

Why v2: low-resolution scans (~1130x554 px against ~2528x1230 px for the same form) lose much of their trace with v1. The black filter plus the stability separator raised the mean valid-window fraction from 0.846 to 0.912 and removed samples with a completely empty lead (5.81% to 0%). Why v3: where traces of neighbouring leads or text overlap, the median trace jumps onto the wrong stroke; the tracker stays on the lead's own trace.

Optional, v3 only: `OVERLAP_TRACKER_FAST=1` uses a numba-compiled tracker (`overlap_tracker_fast.py`, identical paths; needs `numba`). The opt-in preset `black165_stable_sep_repair_tracker_adaptive` (`pipeline/bfss_tracker/layout_adaptive/`) reads layouts other than 3x4 + rhythm strip; sheets in the standard layout take the unchanged v3 path. It was validated on synthetic sheets only (see its README).

Speed (v2 / v3, output identical to earlier code): about 60 ms per scan for the digitizer on a desktop CPU; on a free Kaggle machine (4 vCPU) the digitizer takes 151 / 160 ms per scan and the whole image-to-probabilities chain about 0.22 / 0.23 s with a T4 GPU.

### Checks
Arrays `signal_matrix`, `overlap_mask`, `interpolated_mask` equal to the stored extractions on 40 evenly spaced scans each: `original` (canonical file and `pipeline/original/`) against the v1 export, `black165_stable_sep` (canonical file, `pipeline/bfss/`, `pipeline/bfss_tracker/`) against the v2 export without repair. The speed-ups of `pipeline/bfss/` and `pipeline/bfss_tracker/` were checked as identical on 2252 exports of 563 scans.

## Layout

- `pipeline/`
  - `ecg_digitizer.py`: reusable pipeline (library API, both presets)
  - `bfss_tracker/`: v3, copy of `bfss/` whose lead trace comes from the overlap beam tracker (default preset `black165_stable_sep_repair_tracker`; optional numba tracker and adaptive layout; see its README)
  - `original/` (v1), `bfss/` (v2): self-contained peer copies of the digitizer + extraction script (`bfss` default preset `black165_stable_sep_repair`)
  - `extract_all_npz.py`: batch exporter for all datasets (`--preset`, `--dataset`, `--limit`, `--overwrite`, `--workers`, `--output-dir`)
  - `ecg_waveform_extraction.ipynb`: notebook for visual debugging; implements the `original` behavior only (it does not use the presets)
- `annotation/`: manual anomaly-annotation GUI (`ecg_annotation_gui.py`), log migration tool, logs and analysis (local only; EF datasets)
- `experiments/`: dated investigations, each with its own findings file (local only; see `../research/experiment_registry.md`, IDs A1 to A12)
- `benchmarks/`: comparison against an external reference implementation (local only, undocumented)
- `docs/`: reference papers (local only)
- `archive/`: example single-run output

## Datasets

Raw scans live in `../data/raw/`; see `../research/data_manifest.md` for extraction versions and provenance.

| Folder | Images | Role |
|---|---|---|
| `mitral_images/` | 5626 | Mitral valve task (the research focus) |
| `0_normal_EF/`, `1_reduced_EF/` | 1949, 402 | Ejection-fraction cohorts, annotated with the GUI; not part of the current research scope |

## Pipeline stages

1. Preprocess the scan into a binary ink mask (preset dependent, see above).
2. Detect the four baseline rows and the three vertical separators of the 3x4 lead layout.
3. Crop the 12 short leads and the long rhythm strip.
4. Extract a 1-D trace from each crop (`extract_trace`: dilate, keep the largest connected component).
5. Recover clipped short-lead crops by adaptively expanding the crop height.
6. Recover clipped rhythm-strip crops with rhythm-specific bottom-threshold logic.
7. Interpolate the portions of the waveform that have explicit recovery logic.
8. Detect overlap anomalies on the selected trace mask.
9. Calibrate pixel traces into time and voltage units.
10. Export the canonical signal arrays and aligned masks.

### Calibration

Fixed standard assumption, not estimated from the printed grid: 25 mm/s and 10 mm/mV. The code assumes the usable width is 10.0 s and
`voltage_time_ratio_mV_per_ms = 0.0025`.

### Rhythm strip

A first-class output with its own crop and recovery path; it participates in overlap and interpolation masks. Exported Lead II comes
from the rhythm strip, not from the short-lead II slot.

### Overlap detection

Vertical closing (height 7) on the local mask; a lead or rhythm strip counts as overlapping when detected overlap columns exceed 3.
The exported `overlap_mask` is thresholded: only leads judged globally overlapping keep nonzero samples.

### Interpolation

Limited to places with explicit recovery logic (top-row short leads clipped at the top border; the rhythm strip clipped at the bottom
border). The `interpolated_mask` is sparse by design and is not a generic smoothing mask.

## Output format (`.npz`)

- one array per lead (`I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1` to `V6`)
- `signal_matrix` (5000 x 12), `overlap_mask` (5000 x 12), `interpolated_mask` (5000 x 12)
- `lead_order`, `t_ms`, `meta`
- conventions: 2 ms sampling, 10 s duration, voltage in uV; per-lead NaN outside the lead's own 2.5 s window is by design (except Lead II)

## Annotation GUI

`annotation/ecg_annotation_gui.py` browses raw scans, runs the digitizer on demand, shows Y-trace, lead-crop, overlap and interpolation
views, and lets an annotator tag `Overlap with Text`, `Overlap of Signals`, `Broken Signal` or `Others`. Manual labels are separate from
the automatic masks. Logs are written under `annotation/logs/`.
