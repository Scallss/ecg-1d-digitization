# BFSS pipeline (black filter + stability separator)

Self-contained copy of the digitizer with default preset `black165_stable_sep_repair` (since 2026-10-13: `black165_stable_sep` plus the separator repair; `--preset black165_stable_sep` reproduces the stored stable-separator export; darkness threshold 165 + redness gate +
border stripper; cross-row stability column-separator detector). Reproduces `data/extracted/mitral_images_blackfilter_stableseparator`
(used by `v7h_bfss`). Investigation: `../../experiments/simple_black_filter/FINDINGS.md`, `SEPARATOR_DETECTION_FINDINGS.md`.

    uv run python ecg_digitization/pipeline/bfss/extract_all_npz.py --dataset mitral_images --output-dir data/bfss

Default output is `data/bfss/` so it cannot overwrite the original extraction. Peer of `../original/`.
