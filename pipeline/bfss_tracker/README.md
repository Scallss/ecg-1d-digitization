# BFSS + overlap tracker pipeline

Self-contained copy of the BFSS digitizer (`../bfss/`: black filter 165 + stability separator) in which the lead trace comes from
the column-wise beam tracker instead of the per-column median of the selected component. Default preset: `black165_stable_sep_tracker`.

    uv run python ecg_digitization/pipeline/bfss_tracker/extract_all_npz.py --dataset mitral_images --output-dir data/bfss_tracker

Default output is `data/bfss_tracker/`, so nothing existing is overwritten. Peer of `../original/` and `../bfss/`.

## What differs from `../bfss/`
- `overlap_tracker.py` + `overlap_tracker_settings.json`: frozen copy of `experiments/overlap_trace_tracking/core/tracker.py` and its default settings
  (`tuned_settings_v3.json` + `default_tracker_settings.json`, no anchors). It reads only the crop ink. Do not edit here; change the experiment and re-copy.
- `ecg_digitizer.py`: new parameter `trace_method` (`"median"` default, `"overlap_tracker"`), new preset `black165_stable_sep_tracker`,
  and `track_overlap_trace()` called from `build_lead_result_from_crop`. The component selection still runs, so `connected_mask`,
  overlap detection and the debug views are unchanged; only `y_trace` and `interpolation_mask` come from the tracker.
  Columns the tracker skips are linearly filled when the gap is up to `max(8, 3% of width)` px (same limit as the median trace); longer gaps stay NaN.
  The median-trace special cases (top/bottom occlusion interpolation) are not applied to the tracker trace; the tracker handles crop-edge exits itself.
- `extract_all_npz.py`: default preset and output dir changed.

## Rhythm strip (lead II in the export)
The exported lead II comes from the rhythm strip, which `extract_all_npz.py` re-extracts itself. Until 2026-10-12 that path still used the median trace; it now calls
`track_overlap_trace()` too when `trace_method == "overlap_tracker"` (and `attach_overlap_annotations` takes the strip's interpolation mask from the tracker).
Exact-truth numbers for strips: `../../experiments/overlap_trace_tracking/results/rhythm_eval_summary.txt` (`synthetic_eval/rhythm_eval.py`).

## Checks done (2026-10-12)
- Preset `black165_stable_sep` in this folder gives arrays identical to `../bfss/` (30 `mitral_images`; only `meta.params` differs by the new `trace_method` key).
- Tracker preset on the same 30 images: median abs difference to the median trace 7 uV per lead, 90th percentile 18 uV, 99th 85 uV; skipped-and-filled columns 0.014% vs 0.001%.
  On a scan with heavy overlap (leads V4/V5) the tracker removes the jumps onto the neighbouring lead's strokes and keeps the full R peak (checked by eye on a plot).
- Rhythm strip, 60 mitral_images: preset `black165_stable_sep` still identical to `../bfss/` for all arrays including II; tracker II differs from the median II by 6 uV median, 27 uV p99, no extra NaN.
- NOT done: the full 3 datasets, a comparison of downstream classifier results, and the exact-benchmark numbers for this integration (the method's numbers are in `../../experiments/overlap_trace_tracking/report/main.pdf`).
