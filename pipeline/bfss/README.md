# BFSS pipeline (black filter + stability separator)

Self-contained copy of the digitizer (extraction v2) with default preset `black165_stable_sep_repair`: darkness threshold 165 + redness gate +
border stripper; cross-row stability column-separator detector, with repair of a missed or wrong separator. Investigation: `../../experiments/simple_black_filter/FINDINGS.md`, `SEPARATOR_DETECTION_FINDINGS.md`.

    uv run python ecg_digitization/pipeline/bfss/extract_all_npz.py --dataset mitral_images --output-dir data/bfss

Default output is `data/bfss/` so it cannot overwrite the original extraction. Peer of `../original/`.
