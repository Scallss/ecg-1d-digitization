from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

import ecg_digitizer as ed


DATASET_FOLDERS = ["0_normal_EF", "1_reduced_EF", "mitral_images"]
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_OFFSET_MS = {
    "I": 0,
    "II": 0,
    "III": 0,
    "aVR": 2500,
    "aVL": 2500,
    "aVF": 2500,
    "V1": 5000,
    "V2": 5000,
    "V3": 5000,
    "V4": 7500,
    "V5": 7500,
    "V6": 7500,
}


def build_canonical_export(
    result: ed.DigitizationResult,
    params: dict[str, Any],
    *,
    image_path_text: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dt_ms = float(params["interp_dt_ms"])
    canonical_rows = int((float(params["time_seconds_total"]) * 1000.0) / dt_ms)
    t_canonical = np.arange(canonical_rows, dtype=np.float32) * dt_ms

    lead_signals: dict[str, np.ndarray] = {}
    exclude_from_12lead = {"II"}
    paper_x0 = int(result.bounds[0] + int(float(params["lr_pad_ratio"]) * result.raw_image_bgr.shape[1]))
    time_res_ms = float(result.time_res_ms_per_px)
    volt_res_mV = float(result.volt_res_mV_per_px)

    for lead in LEAD_ORDER:
        if lead in exclude_from_12lead:
            continue
        lead_result = result.lead_results.get(lead)
        if lead_result is None or lead_result.y_trace is None:
            continue
        y = lead_result.y_trace
        valid = ~np.isnan(y)
        if valid.sum() < 20:
            continue
        y0 = np.nanmedian(y[valid])
        amp_px = y0 - y
        amp_mV = amp_px * volt_res_mV
        crop_x0 = int(lead_result.crop_box[0])
        t_global = ((crop_x0 - paper_x0) + np.arange(len(amp_mV), dtype=np.float32)) * time_res_ms
        v_mV = np.interp(t_canonical, t_global[valid], amp_mV[valid], left=np.nan, right=np.nan)
        lead_signals[lead] = v_mV * 1000.0

    if result.rhythm_strip is not None:
        rhythm_trace, _ = ed.extract_trace(
            result.rhythm_strip,
            dilate_k=int(params["dilate_k"]),
            lead=None,
            params=params,
        )
        if rhythm_trace is not None:
            valid = ~np.isnan(rhythm_trace)
            if valid.sum() >= 20:
                y0_r = np.nanmedian(rhythm_trace[valid])
                amp_px_r = y0_r - rhythm_trace
                amp_mV_r = amp_px_r * volt_res_mV
                t_local_r = np.arange(len(amp_mV_r), dtype=np.float32) * time_res_ms
                v_mV_r = np.interp(t_canonical, t_local_r[valid], amp_mV_r[valid], left=np.nan, right=np.nan)
                lead_signals["II"] = v_mV_r * 1000.0

    cols = [lead_signals.get(lead, np.full(canonical_rows, np.nan, dtype=np.float32)) for lead in LEAD_ORDER]
    signal_matrix = np.column_stack(cols)

    def paint_cols(active_cols: np.ndarray | None, t_centers_ms: np.ndarray | None, signal_vec_uV: np.ndarray) -> np.ndarray:
        mask = np.zeros(canonical_rows, dtype=np.uint8)
        if active_cols is None or t_centers_ms is None:
            return mask
        active_cols = np.asarray(active_cols)
        xs = np.where(active_cols > 0)[0]
        if xs.size == 0:
            return mask
        half_px_ms = 0.5 * time_res_ms
        for x_coord in xs:
            t0 = max(0.0, float(t_centers_ms[x_coord] - half_px_ms))
            t1 = min(float(t_canonical[-1] + dt_ms), float(t_centers_ms[x_coord] + half_px_ms))
            i0 = max(0, int(np.floor(t0 / dt_ms)))
            i1 = min(canonical_rows, int(np.ceil(t1 / dt_ms)))
            mask[i0:i1] = 1
        return (mask & (~np.isnan(signal_vec_uV)).astype(np.uint8)).astype(np.uint8)

    overlap_cols = []
    interpolated_cols = []
    for lead in LEAD_ORDER:
        signal_vec = lead_signals.get(lead, np.full(canonical_rows, np.nan, dtype=np.float32))
        if lead == "II":
            overlap_info = result.rhythm_overlap_result
            cluster_counts = None if overlap_info is None else overlap_info.get("cluster_counts")
            is_overlapping = bool(overlap_info is not None and overlap_info.get("is_overlapping", False))
            t_centers_ms = np.arange(len(cluster_counts), dtype=np.float32) * time_res_ms if cluster_counts is not None else None
            overlap_mask = paint_cols(
                (cluster_counts >= 2).astype(np.uint8) if (cluster_counts is not None and is_overlapping) else None,
                t_centers_ms,
                signal_vec,
            )
            interpolation_mask = paint_cols(
                result.rhythm_interpolation_mask,
                np.arange(len(result.rhythm_interpolation_mask), dtype=np.float32) * time_res_ms
                if result.rhythm_interpolation_mask is not None else None,
                signal_vec,
            )
        else:
            lead_result = result.lead_results.get(lead)
            overlap_info = lead_result.overlap_result if lead_result is not None else None
            cluster_counts = None if overlap_info is None else overlap_info.get("cluster_counts")
            is_overlapping = bool(overlap_info is not None and overlap_info.get("is_overlapping", False))
            if lead_result is not None:
                crop_x0 = int(lead_result.crop_box[0])
                t_centers_ms = ((crop_x0 - paper_x0) + np.arange(lead_result.crop_binary.shape[1], dtype=np.float32)) * time_res_ms
            else:
                t_centers_ms = None
            overlap_mask = paint_cols(
                (cluster_counts >= 2).astype(np.uint8) if (cluster_counts is not None and is_overlapping) else None,
                t_centers_ms,
                signal_vec,
            )
            interpolation_mask = paint_cols(
                None if lead_result is None else lead_result.interpolation_mask,
                t_centers_ms,
                signal_vec,
            )
        overlap_cols.append(overlap_mask)
        interpolated_cols.append(interpolation_mask)

    overlap_mask = np.column_stack(overlap_cols).astype(np.uint8)
    interpolated_mask = np.column_stack(interpolated_cols).astype(np.uint8)

    short_lead_overlap_results = {
        lead: {k: v for k, v in lead_result.overlap_result.items() if k != "cluster_counts"}
        for lead, lead_result in result.lead_results.items()
        if lead_result.overlap_result is not None
    }
    rhythm_strip_overlap_result = (
        {k: v for k, v in result.rhythm_overlap_result.items() if k != "cluster_counts"}
        if result.rhythm_overlap_result is not None else None
    )

    meta = {
        "params": params,
        "time_res_ms_per_px": time_res_ms,
        "volt_res_mV_per_px": volt_res_mV,
        "image_path": image_path_text,
        "canonical_rows": canonical_rows,
        "dt_ms": dt_ms,
        "units": "uV",
        "lead_order": LEAD_ORDER,
        "lead_offset_ms": LEAD_OFFSET_MS,
        "alignment_mode": "global_crop_x",
        "paper_x0_px": paper_x0,
        "excluded_from_12lead": list(exclude_from_12lead),
        "overlap_detection_mode": "vertical_close_column_count",
        "overlap_vertical_close_h": int(params["overlap_vertical_close_h"]),
        "overlap_column_count_threshold": int(params["overlap_column_count_threshold"]),
        "short_lead_overlap_results": short_lead_overlap_results,
        "rhythm_strip_overlap_result": rhythm_strip_overlap_result,
        "overlap_mask_array": "overlap_mask",
        "overlap_mask_description": "5000x12 canonical binary mask aligned to signal_matrix and lead_order, but only for leads marked overlapping by the main thresholded decision",
        "interpolated_mask_array": "interpolated_mask",
        "interpolated_mask_description": "5000x12 canonical binary mask aligned to signal_matrix and lead_order, marking only samples inferred by interpolation in top-row leads and the rhythm strip export",
        "exported_lead_ii_overlap_source": "rhythm_strip_overlap_result",
        "exported_lead_ii_interpolation_source": "rhythm_strip",
    }
    return lead_signals, signal_matrix, overlap_mask, interpolated_mask, t_canonical, meta


def export_image_to_npz(
    image_path: Path,
    out_path: Path,
    params: dict[str, Any],
    *,
    image_path_text: str,
) -> None:
    result = ed.digitize_ecg_image(image_path, params=params)
    lead_signals, signal_matrix, overlap_mask, interpolated_mask, t_ms, meta = build_canonical_export(
        result,
        params,
        image_path_text=image_path_text,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **{lead: signal_matrix[:, i] for i, lead in enumerate(LEAD_ORDER)},
        signal_matrix=signal_matrix,
        overlap_mask=overlap_mask,
        interpolated_mask=interpolated_mask,
        lead_order=np.array(LEAD_ORDER, dtype="<U4"),
        t_ms=t_ms,
        meta=json.dumps(meta),
    )


def iter_images(dataset_dir: Path) -> list[Path]:
    return sorted(
        [path for path in dataset_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}],
        key=lambda path: path.name.lower(),
    )


def export_image_task(
    image_path_str: str,
    out_path_str: str,
    params: dict[str, Any],
    image_path_text: str,
) -> tuple[str, str, str]:
    image_path = Path(image_path_str)
    out_path = Path(out_path_str)
    try:
        export_image_to_npz(
            image_path,
            out_path,
            params,
            image_path_text=image_path_text,
        )
    except Exception as exc:  # pragma: no cover - batch failure reporting
        return ("failed", image_path.name, str(exc))
    return ("wrote", out_path.name, "")


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extract NPZ exports for every ECG image dataset.")
    workspace_root = Path(__file__).resolve().parent.parent.parent.parent
    parser.add_argument("--root", type=Path, default=workspace_root / "data" / "raw", help="Directory containing dataset folders")
    parser.add_argument("--output-dir", type=Path, default=workspace_root / "data" / "extracted", help="Output root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ files")
    parser.add_argument("--dataset", action="append", choices=DATASET_FOLDERS, help="Restrict to one or more dataset folders")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset image limit for testing")
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of worker processes to use per dataset. Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(ed.PRESETS),
        default="original",
        help="Extraction preset. 'original' is the pipeline behind data/extracted/mitral_images; "
        "'black165_stable_sep' is the black filter + stability separator (data/extracted/mitral_images_blackfilter_stableseparator).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = args.output_dir.resolve()
    datasets = args.dataset or DATASET_FOLDERS
    params = ed.get_params(args.preset)

    output_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        dataset_dir = root / dataset
        dataset_out = output_root / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        images = iter_images(dataset_dir)
        if args.limit is not None:
            images = images[: args.limit]

        total = len(images)
        success = 0
        skipped = 0
        failed = 0
        workers = max(1, int(args.workers))
        print(f"[{dataset}] {total} images -> {dataset_out} (workers={workers})")

        pending_tasks: list[tuple[Path, Path]] = []
        for idx, image_path in enumerate(images, start=1):
            out_path = dataset_out / f"{image_path.stem}.npz"
            if out_path.exists() and not args.overwrite:
                skipped += 1
                print(f"  [{idx}/{total}] skip  {image_path.name}")
                continue
            pending_tasks.append((image_path, out_path))

        if workers == 1:
            for image_path, out_path in pending_tasks:
                try:
                    export_image_to_npz(
                        image_path,
                        out_path,
                        params,
                        image_path_text=str(image_path.relative_to(root)),
                    )
                    success += 1
                    print(f"  [{success + skipped + failed}/{total}] wrote {out_path.name}")
                except Exception as exc:  # pragma: no cover - batch failure reporting
                    failed += 1
                    print(f"  [{success + skipped + failed}/{total}] FAIL  {image_path.name}: {exc}")
        else:
            completed = skipped
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(
                        export_image_task,
                        str(image_path),
                        str(out_path),
                        params,
                        str(image_path.relative_to(root)),
                    ): (image_path, out_path)
                    for image_path, out_path in pending_tasks
                }
                for future in as_completed(future_to_task):
                    completed += 1
                    image_path, out_path = future_to_task[future]
                    try:
                        status, output_name, message = future.result()
                    except Exception as exc:  # pragma: no cover - worker failure reporting
                        failed += 1
                        print(f"  [{completed}/{total}] FAIL  {image_path.name}: {exc}")
                        continue

                    if status == "wrote":
                        success += 1
                        print(f"  [{completed}/{total}] wrote {output_name}")
                    else:
                        failed += 1
                        print(f"  [{completed}/{total}] FAIL  {image_path.name}: {message}")

        print(f"[{dataset}] done: wrote={success}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
