from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import shutil


SOURCE_FOLDERS = ["0_normal_EF", "1_reduced_EF", "mitral_images"]
MERGED_CSV_BY_FOLDER = {
    "0_normal_EF": Path("analysis/normal_EF_merged.csv"),
    "1_reduced_EF": Path("analysis/reduced_EF_merged.csv"),
}
ANNOTATION_LOG_FILENAME = "annotation_logs.csv"
LEGACY_ANNOTATION_LOG_FILENAME = "annotation_log.csv"
DEFAULT_ASSIGNED_TO = "Pascal"
ANOMALY_FIELDS = [
    ("Overlap with Text", "anomaly_overlap_with_text"),
    ("Overlap of Signals", "anomaly_overlap_of_signals"),
    ("Broken Signal", "anomaly_broken_signal"),
    ("Others", "anomaly_others"),
]
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_FIELDS = [(lead, f"lead_{lead}") for lead in LEAD_ORDER]
TARGET_FIELDS = [
    "timestamp",
    "filename",
    "assigned_to",
    "label",
    "notes",
    *[column for _, column in ANOMALY_FIELDS],
    *[column for _, column in LEAD_FIELDS],
]
MERGED_ANOMALY_FIELD_MAP = {
    "anomaly_overlap_with_text": "Text Overlap",
    "anomaly_overlap_of_signals": "Signals Overlap",
    "anomaly_broken_signal": "Broken Signal",
    "anomaly_others": "Others",
}


def csv_bool(value: str) -> bool:
    return value.strip().upper() == "TRUE"


def csv_bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def natural_sort_text(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def normalize_old_anomaly_values(raw_value: str) -> set[str]:
    normalized: set[str] = set()
    for value in raw_value.split(";"):
        value = value.strip().lower()
        if not value:
            continue
        normalized.add(value)
    return normalized


def normalize_old_lead_values(raw_value: str) -> set[str]:
    normalized: set[str] = set()
    for value in raw_value.split(";"):
        value = value.strip()
        if not value:
            continue
        normalized.add(value)
    return normalized


def legacy_combined_anomaly_selected(row: dict[str, str], old_anomalies: set[str]) -> bool:
    return csv_bool(row.get("anomaly_overlap_both_with_text_and_of_signals", "")) or (
        "overlap both with text and of signals" in old_anomalies
    )


def convert_row(row: dict[str, str]) -> dict[str, str]:
    filename = row.get("filename") or Path(row.get("image_path", "")).name
    label = (row.get("label") or row.get("image_label") or "Normal").strip() or "Normal"
    assigned_to = (row.get("assigned_to") or DEFAULT_ASSIGNED_TO).strip() or DEFAULT_ASSIGNED_TO
    notes = row.get("notes", "").strip()
    timestamp = row.get("timestamp", "").strip()

    old_anomalies = normalize_old_anomaly_values(row.get("anomaly_type", ""))
    old_leads = normalize_old_lead_values(row.get("anomalous_leads", ""))
    combined_selected = legacy_combined_anomaly_selected(row, old_anomalies)

    converted = {
        "timestamp": timestamp,
        "filename": filename,
        "assigned_to": assigned_to,
        "label": label,
        "notes": notes,
    }

    for label_text, column in ANOMALY_FIELDS:
        selected = csv_bool(row.get(column, "")) or label_text.lower() in old_anomalies
        if column in {"anomaly_overlap_with_text", "anomaly_overlap_of_signals"}:
            selected = selected or combined_selected
        converted[column] = csv_bool_text(selected)

    for lead, column in LEAD_FIELDS:
        converted[column] = csv_bool_text(
            csv_bool(row.get(column, "")) or lead in old_leads
        )

    if label != "Anomalous":
        for _, column in ANOMALY_FIELDS:
            converted[column] = "FALSE"
        for _, column in LEAD_FIELDS:
            converted[column] = "FALSE"

    return converted


def convert_merged_row(row: dict[str, str]) -> dict[str, str]:
    label = (row.get("Label") or "Normal").strip() or "Normal"
    converted = {
        "timestamp": "",
        "filename": (row.get("Filename") or "").strip(),
        "assigned_to": (row.get("Assignment") or DEFAULT_ASSIGNED_TO).strip() or DEFAULT_ASSIGNED_TO,
        "label": label,
        "notes": (row.get("Notes") or "").strip(),
    }
    for _, column in ANOMALY_FIELDS:
        merged_field = MERGED_ANOMALY_FIELD_MAP[column]
        converted[column] = csv_bool_text(label == "Anomalous" and csv_bool(row.get(merged_field, "")))
    for lead, column in LEAD_FIELDS:
        converted[column] = csv_bool_text(label == "Anomalous" and csv_bool(row.get(f"lead_{lead}", "")))
    return converted


def migrate_csv(csv_path: Path, apply_changes: bool, backup_suffix: str) -> tuple[int, bool]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [convert_row(row) for row in reader]

    rows.sort(key=lambda row: natural_sort_text(row["filename"]))

    already_migrated = True
    current_fields = reader.fieldnames or []
    if current_fields != TARGET_FIELDS:
        already_migrated = False
    if any(set(row.keys()) != set(TARGET_FIELDS) for row in rows):
        already_migrated = False

    if apply_changes:
        backup_path = csv_path.with_name(csv_path.name + backup_suffix)
        shutil.copy2(csv_path, backup_path)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TARGET_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    return len(rows), already_migrated


def import_merged_csv(merged_csv_path: Path, target_csv_path: Path, apply_changes: bool, backup_suffix: str) -> int:
    with merged_csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [convert_merged_row(row) for row in reader if (row.get("Filename") or "").strip()]
    rows.sort(key=lambda row: natural_sort_text(row["filename"]))

    if apply_changes:
        target_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if target_csv_path.exists():
            backup_path = target_csv_path.with_name(target_csv_path.name + backup_suffix)
            shutil.copy2(target_csv_path, backup_path)
        with target_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TARGET_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate annotation_log.csv files to the checkbox-friendly schema used by the GUI."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing logs/",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the CSV files in place. Without this flag, the script only prints what it would do.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix added to backups when --apply is used.",
    )
    parser.add_argument(
        "--import-merged",
        action="store_true",
        help="Import analysis/*_merged.csv into logs/<folder>/annotation_logs.csv.",
    )
    args = parser.parse_args()

    output_root = args.root / "logs"
    if args.import_merged:
        for folder, rel_path in MERGED_CSV_BY_FOLDER.items():
            merged_csv_path = args.root / rel_path
            target_csv_path = output_root / folder / ANNOTATION_LOG_FILENAME
            if not merged_csv_path.exists():
                print(f"{folder}: no merged CSV found at {merged_csv_path}")
                continue
            row_count = import_merged_csv(merged_csv_path, target_csv_path, apply_changes=args.apply, backup_suffix=args.backup_suffix)
            status = "imported" if args.apply else "would import"
            print(f"{folder}: {status} {row_count} rows from {merged_csv_path} to {target_csv_path}")
        return

    found_any = False
    for folder in SOURCE_FOLDERS:
        csv_path = output_root / folder / ANNOTATION_LOG_FILENAME
        if not csv_path.exists():
            csv_path = output_root / folder / LEGACY_ANNOTATION_LOG_FILENAME
        if not csv_path.exists():
            print(f"{folder}: no annotation log found")
            continue

        found_any = True
        row_count, already_migrated = migrate_csv(csv_path, apply_changes=args.apply, backup_suffix=args.backup_suffix)
        status = "rewritten" if args.apply else "would rewrite"
        if already_migrated:
            status = "normalized" if args.apply else "would normalize"
        print(f"{folder}: {status} {row_count} rows at {csv_path}")
        if args.apply:
            print(f"{folder}: backup saved to {csv_path}{args.backup_suffix}")

    if not found_any:
        print(f"No annotation logs found under {output_root}")


if __name__ == "__main__":
    main()
