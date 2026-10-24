# layout_adaptive: layouts other than 3x4 + rhythm strip (opt-in, 2026-10-17)

Opt-in through `layout_method="adaptive"`, preset `black165_stable_sep_repair_tracker_adaptive` (= `black165_stable_sep_repair_tracker` + adaptive layout). The default presets and `data/bfss_tracker` are untouched.

    uv run python ecg_digitization/pipeline/bfss_tracker/extract_all_npz.py --preset black165_stable_sep_repair_tracker_adaptive --output-dir data/bfss_tracker_adaptive

## What it does
1. `adaptive_plan.make_plan(img)`: rows (comb fit), layout (label positions against the 22-layout catalogue in `layouts.py`), column starts (pipeline `stability_repair` separators when printed ticks verify, else an equal grid capped by the label edge), lead name of every slot
   (`lead_order_general.decide`: labels if all 12 words read with similarity >= 0.7, else limb-lead relations, else unverified = default standard names).
2. **3x4 + one rhythm strip in the standard order (or unverified on 3x4 + 1, or no layout found): the unchanged fixed path** (`digitize_ecg_image` as before; arrays identical to the non-adaptive preset by construction, checked).
3. Every other layout: `adaptive_digitizer.digitize_adaptive` (one crop per slot of the plan, beam tracker trace) and `build_adaptive_export` (canonical 10 s x 12 export, same npz keys).

## Differences of the general path (deliberate, see the module docstring)
- Time and voltage scale come from the paper (25 mm/s, 10 mm/mV) with the sheet's px/mm (`fine_px_per_mm`: autocorrelation peaks of the grid over many periods, 0.01% off on synthetic sheets, the coarse integer-period estimate is 1% off). The fixed path divides 10 s by the rhythm-strip width instead; on a sheet whose content is 9.76 s wide (our renderer, probably the real scans too) the two scales differ by 2.4%. **Not reconciled.**
- Time origin = left edge of the first crop (like the fixed path). Shifted vs simultaneous time structure is not detected: the export assumes shifted (`time_mode: assumed_shifted` in the meta); simultaneous layouts (3x4_sim, 6x2_sim) therefore carry wrong time offsets.
- A rhythm-strip slot wins over a cell of the same lead (like lead II in the fixed path); with 2-3 rhythm rows the rhythm leads are assumed II, V5 / II, V1, V5 (not read from labels).
- A slot printed with a minus sign (Cabrera -aVR) is stored as `aVR` with the sign undone.
- No iterative crop recovery, no top-of-page occlusion handling. Crop half-height 0.70 of the row pitch (as in the fixed path); 0.40-0.60 were worse on the tuning records.
- Unverified sheets of any layout other than 3x4 + 1 are exported with the standard names and `order_status: unverified` in the meta; the caller should treat those leads as unchecked.

## Frozen code
Copies of `experiments/layout_and_label_separators/detect/` (`layout_detect`, `label_detect`, `label_blobs`, `label_reader`, `layout_infer`, `ticks_fallback`, `lead_order`, `lead_order_general`), `render/layouts.py` and `real_label_bank.pkl` (label templates harvested from our real scans), with only the import paths and the tracker call changed (`overlap_tracker.trace(ink > 0)`). Change the experiment, then re-copy; do not edit here.
Evaluation: `experiments/layout_and_label_separators/detect/eval_adaptive_e2e.py` (synthetic sheets, 22 layouts, truth from the renderer), `eval_ahus.py` (real Ahus pages, component test). Numbers: `research/claims_ledger.md` L17-L19.
