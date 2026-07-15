# MLCV ECG Digitization

This is the working project for paper-ECG digitization, anomaly annotation, and downstream dataset preparation.

The core of the project is a custom ECG waveform extraction pipeline that converts scanned 12-lead ECG images into aligned digital signals, while also tracking two important artifact channels:
- `overlap_mask`: columns/samples affected by detected overlap anomalies
- `interpolated_mask`: columns/samples where the waveform was inferred by interpolation

## What This Project Contains

At a high level, this project combines five related activities:
- waveform digitization and calibration
- anomaly annotation with a GUI workflow
- analysis of merged annotation logs
- experimental branches for improving extraction quality
- benchmarking against an external reference implementation

The main reusable implementation is [`pipeline/ecg_digitizer.py`](./pipeline/ecg_digitizer.py). The main notebook for inspection and debugging is [`pipeline/ecg_waveform_extraction.ipynb`](./pipeline/ecg_waveform_extraction.ipynb).

## Layout

- `pipeline/`: main reusable digitization code (script + notebook)
- `annotation/`: annotation GUI, log migration tooling, annotation logs, and merged analysis

### `pipeline/`

- `pipeline/ecg_digitizer.py`: main reusable digitization pipeline
- `pipeline/ecg_waveform_extraction.ipynb`: notebook version of the main pipeline with visual debugging
- `pipeline/extract_all_npz.py`: batch exporter that digitizes all datasets into `.npz` files

### `annotation/`

- `annotation/ecg_annotation_gui.py`: annotation GUI that runs the current digitizer and saves anomaly labels
- `annotation/migrate_annotation_logs.py`: utility for normalizing and importing annotation logs
- `annotation/logs/`: GUI annotation outputs, lead-level anomaly snapshots, and annotation logs (was `anomalous_images/`)
- `annotation/analysis/`: merged annotation CSVs and simple analysis artifacts (was `anomaly_analysis/`)

## Datasets

Currently uses three image datasets:

- `0_normal_EF/`: `1949` images
- `1_reduced_EF/`: `402` images
- `mitral_images/`: `5626` images

### 1. `0_normal_EF/`
Used for the normal-ejection-fraction cohort in the EF task.

### 2. `1_reduced_EF/`
Used for the reduced-ejection-fraction cohort in the EF task.

### 3. `mitral_images/`
Used for a separate mitral-regurgitation task. This workspace shares the digitization tools with that dataset, but the downstream modeling work for mitral classification is intentionally documented elsewhere.


## Main Digitization Pipeline

There are two main entry points:

- [`pipeline/ecg_digitizer.py`](./pipeline/ecg_digitizer.py): reusable script/API implementation
- [`pipeline/ecg_waveform_extraction.ipynb`](./pipeline/ecg_waveform_extraction.ipynb): notebook version for inspection, visualization, and one-off debugging

The notebook and script are intended to stay behaviorally consistent. The notebook exists because many extraction failures are easiest to understand visually.

### Pipeline Stages

The current pipeline, in broad terms, does the following:

1. preprocess the ECG image and suppress the red grid/background
2. detect the four baseline rows and the three vertical separators of the 3x4 lead layout
3. crop the 12 short leads and the long rhythm strip
4. extract a 1D waveform trace from each crop
5. recover clipped short-lead crops by adaptively expanding vertical crop height when needed
6. recover clipped rhythm-strip crops with rhythm-specific bottom-threshold logic when the signal is cut by the bottom image boundary
7. interpolate specific missing portions of the waveform where the pipeline has explicit recovery logic
8. detect overlap anomalies on the selected trace mask
9. calibrate pixel traces into time and voltage units
10. export canonical signal arrays and aligned masks

### Calibration Assumptions

The pipeline currently uses a **fixed standard ECG calibration assumption**, not an image-derived grid calibration.

Assumed standard:
- `25 mm/s`
- `10 mm/mV`

Implementation-wise, the code assumes:
- total usable width corresponds to `10.0` seconds
- `voltage_time_ratio_mV_per_ms = 0.0025`

That is consistent with standard ECG paper settings, but it is still an assumption. The pipeline does **not** currently estimate paper speed or gain directly from the printed grid.

### Rhythm Strip Handling

The rhythm strip is treated as a first-class output, not just a plotting artifact.

Current behavior:
- it has its own crop and its own recovery path
- it participates in overlap detection
- it participates in interpolation-mask generation
- exported Lead `II` in the canonical 12-lead output comes from the rhythm strip, not from the short-lead `II` slot

### Overlap Detection

The current main overlap detector uses a vertical-closing-based local mask followed by a fixed anomaly threshold.

Current main settings:
- vertical closing height: `1x7`
- anomaly decision: a lead/rhythm strip is considered overlapping if detected overlap columns are `> 3`

The exported `overlap_mask` is **thresholded**:
- local overlap columns are projected to the canonical time axis
- but only leads judged globally overlapping by the current main rule keep nonzero overlap samples in the exported mask

### Interpolation Logic

The current pipeline also tracks where signal values were inferred rather than directly observed.

Interpolation is currently limited to the places where the pipeline has explicit recovery logic:
- top-row short leads with top-border clipping logic
- rhythm strip with bottom-border clipping logic

This is why the exported `interpolated_mask` is sparse by design. It is not a generic smoothing mask; it only marks samples that were actually inferred by recovery logic.

## Output Formats

### Notebook Single-Run Output

The notebook writes example single-run output into `archive/`:
- `archive/digitised_ecg.csv`
- `archive/digitised_ecg.npz`

The CSV is useful for inspection, but the `.npz` is the main machine-readable export.

### NPZ Export Schema

The current `.npz` export contains:
- one array per lead (`I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`-`V6`)
- `signal_matrix`: canonical `5000 x 12` matrix
- `overlap_mask`: canonical `5000 x 12` binary mask aligned to `signal_matrix`
- `interpolated_mask`: canonical `5000 x 12` binary mask aligned to `signal_matrix`
- `lead_order`
- `t_ms`
- `meta`

Important export conventions:
- canonical sampling interval is `2 ms`
- canonical duration is `10 seconds`
- voltage unit in exported arrays is `uV`
- `overlap_mask` is thresholded by the main overlap decision
- `interpolated_mask` only marks values that were inferred by explicit interpolation logic

## Batch Extraction

[`pipeline/extract_all_npz.py`](./pipeline/extract_all_npz.py) batch-processes all datasets in `../data/raw/` and writes `.npz` files into [`../data/extracted/`](../data/extracted/).

Output layout:
- [`../data/extracted/0_normal_EF`](../data/extracted/0_normal_EF)
- [`../data/extracted/1_reduced_EF`](../data/extracted/1_reduced_EF)
- [`../data/extracted/mitral_images`](../data/extracted/mitral_images)

Each output filename matches the image stem, for example:
- `B002305.jpg -> B002305.npz`
- `A012599.jpg -> A012599.npz`

Typical usage (defaults now point at `../data/raw/` and `../data/extracted/` automatically):

```bash
python pipeline/extract_all_npz.py
```

Useful options:

```bash
python pipeline/extract_all_npz.py --dataset 0_normal_EF --overwrite
python pipeline/extract_all_npz.py --dataset 1_reduced_EF --limit 10
python pipeline/extract_all_npz.py --dataset mitral_images
```

## Annotation Workflow

Manual anomaly labeling is handled by [`annotation/ecg_annotation_gui.py`](./annotation/ecg_annotation_gui.py).

### What the GUI Does

The GUI:
- browses ECG images from `../data/raw/`
- runs the current digitization pipeline on demand
- shows multiple synchronized visual views of the extracted result
- lets annotators tag anomalies
- writes persistent logs under [`annotation/logs/`](./annotation/logs/)


### Manual Anomaly Categories

The GUI supports four manual anomaly categories:
- `Overlap with Text`
- `Overlap of Signals`
- `Broken Signal`
- `Others`

These manual labels are separate from the automatic `overlap_mask` and `interpolated_mask` produced by the digitizer.

### GUI Visual Views

The current GUI includes dedicated tabs for:
- `Y-Trace Views`
- `Lead Crops`
- `Overlap Views`
- `Interpolation Views`

The GUI also includes:
- rhythm-strip visualizations
- highlighted overlap detections
- highlighted interpolation detections
- a small automatic summary showing which leads were flagged by the current pipeline
