# Original pipeline

Self-contained copy of the canonical digitizer (red-grid whitening + hybrid ink mask, morphology column-separator detector).
Reproduces `data/extracted/mitral_images` exactly. Default preset: `original`.

    uv run python ecg_digitization/pipeline/original/extract_all_npz.py --output-dir data/extracted

Frozen peer of `../bfss/`. The shared library with both presets remains at `../ecg_digitizer.py`.
