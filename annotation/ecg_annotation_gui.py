from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import traceback
from typing import Any

import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from ecg_digitizer import (
    LEAD_ORDER,
    bgr_to_rgb,
    build_crop_figure,
    build_interpolation_figure,
    build_overlap_figure,
    build_trace_figure,
    digitize_ecg_image,
)


IMAGES_ROOT = WORKSPACE_ROOT / "data" / "raw"
SOURCE_FOLDERS = ["0_normal_EF", "1_reduced_EF", "mitral_images"]
OUTPUT_DIR = "logs"
ANNOTATION_LOG_FILENAME = "annotation_logs.csv"
LEGACY_ANNOTATION_LOG_FILENAME = "annotation_log.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_ASSIGNED_TO = "Pascal"
MERGED_CSV_BY_FOLDER = {
    "0_normal_EF": Path("analysis/normal_EF_merged.csv"),
    "1_reduced_EF": Path("analysis/reduced_EF_merged.csv"),
}
ANOMALY_FIELDS = [
    ("Overlap with Text", "anomaly_overlap_with_text"),
    ("Overlap of Signals", "anomaly_overlap_of_signals"),
    ("Broken Signal", "anomaly_broken_signal"),
    ("Others", "anomaly_others"),
]
LEAD_FIELDS = [(lead, f"lead_{lead}") for lead in LEAD_ORDER]
MERGED_ANOMALY_FIELD_MAP = {
    "anomaly_overlap_with_text": "Text Overlap",
    "anomaly_overlap_of_signals": "Signals Overlap",
    "anomaly_broken_signal": "Broken Signal",
    "anomaly_others": "Others",
}


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def natural_sort_text(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def csv_bool(value: str) -> bool:
    return value.strip().upper() == "TRUE"


def csv_bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def legacy_combined_anomaly_selected(record: dict[str, str], old_anomaly_values: set[str]) -> bool:
    return csv_bool(record.get("anomaly_overlap_both_with_text_and_of_signals", "")) or (
        "overlap both with text and of signals" in old_anomaly_values
    )


def merged_row_to_annotation_record(row: dict[str, str]) -> dict[str, str]:
    label = (row.get("Label") or "Normal").strip() or "Normal"
    record = {
        "timestamp": "",
        "filename": (row.get("Filename") or "").strip(),
        "assigned_to": (row.get("Assignment") or DEFAULT_ASSIGNED_TO).strip() or DEFAULT_ASSIGNED_TO,
        "label": label,
        "notes": (row.get("Notes") or "").strip(),
    }
    for _, column in ANOMALY_FIELDS:
        merged_field = MERGED_ANOMALY_FIELD_MAP[column]
        record[column] = csv_bool_text(label == "Anomalous" and csv_bool(row.get(merged_field, "")))
    for lead, column in LEAD_FIELDS:
        record[column] = csv_bool_text(label == "Anomalous" and csv_bool(row.get(f"lead_{lead}", "")))
    return record


class MatplotlibView(ttk.Frame):
    def __init__(self, master: tk.Misc, placeholder_text: str, *, scroll_y: bool = False, scroll_x: bool = False) -> None:
        super().__init__(master)
        self.placeholder_text = placeholder_text
        self.scroll_y_enabled = scroll_y
        self.scroll_x_enabled = scroll_x
        self.figure = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.toolbar: NavigationToolbar2Tk | None = None
        self.toolbar_frame: ttk.Frame | None = None
        self.scroll_canvas: tk.Canvas | None = None
        self.x_scrollbar: ttk.Scrollbar | None = None
        self.y_scrollbar: ttk.Scrollbar | None = None
        self.canvas_window_id: int | None = None
        self.placeholder = ttk.Label(self, text=placeholder_text)
        self.placeholder.pack(expand=True)

    def show_figure(self, figure) -> None:
        self.clear()
        self.figure = figure
        self.placeholder.pack_forget()
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(fill="x")
        if self.scroll_y_enabled or self.scroll_x_enabled:
            self.scroll_canvas = tk.Canvas(self, highlightthickness=0)
            if self.scroll_y_enabled:
                self.y_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.scroll_canvas.yview)
                self.y_scrollbar.pack(side="right", fill="y")
            if self.scroll_x_enabled:
                self.x_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.scroll_canvas.xview)
                self.x_scrollbar.pack(side="bottom", fill="x")
            self.scroll_canvas.configure(
                xscrollcommand=self.x_scrollbar.set if self.x_scrollbar is not None else "",
                yscrollcommand=self.y_scrollbar.set if self.y_scrollbar is not None else "",
            )
            self.scroll_canvas.pack(side="left", fill="both", expand=True)
            canvas_master = self.scroll_canvas
        else:
            canvas_master = self

        self.canvas = FigureCanvasTkAgg(figure, master=canvas_master)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")

        widget = self.canvas.get_tk_widget()
        if self.scroll_canvas is not None:
            self.canvas_window_id = self.scroll_canvas.create_window((0, 0), window=widget, anchor="nw")
            widget.bind("<Configure>", self._sync_scroll_region)
            self.scroll_canvas.bind("<Configure>", self._sync_scroll_region)
            self._bind_mousewheel(widget)
            self._sync_scroll_region()
        else:
            widget.pack(fill="both", expand=True)

    def clear(self, show_placeholder: bool = True) -> None:
        try:
            if self.canvas is not None:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None
        except tk.TclError:
            self.canvas = None
        try:
            if self.toolbar is not None:
                self.toolbar.destroy()
                self.toolbar = None
        except tk.TclError:
            self.toolbar = None
        try:
            if self.toolbar_frame is not None:
                self.toolbar_frame.destroy()
                self.toolbar_frame = None
        except tk.TclError:
            self.toolbar_frame = None
        try:
            if self.scroll_canvas is not None:
                self.scroll_canvas.destroy()
                self.scroll_canvas = None
        except tk.TclError:
            self.scroll_canvas = None
        try:
            if self.x_scrollbar is not None:
                self.x_scrollbar.destroy()
                self.x_scrollbar = None
        except tk.TclError:
            self.x_scrollbar = None
        try:
            if self.y_scrollbar is not None:
                self.y_scrollbar.destroy()
                self.y_scrollbar = None
        except tk.TclError:
            self.y_scrollbar = None
        self.canvas_window_id = None
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
        if show_placeholder and self.winfo_exists() and self.placeholder.winfo_exists():
            try:
                self.placeholder.pack(expand=True)
            except tk.TclError:
                pass

    def _sync_scroll_region(self, _event=None) -> None:
        if self.scroll_canvas is None or self.canvas is None:
            return
        try:
            self.scroll_canvas.update_idletasks()
            bbox = self.scroll_canvas.bbox("all")
            if bbox is not None:
                self.scroll_canvas.configure(scrollregion=bbox)
        except tk.TclError:
            return

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        if self.scroll_y_enabled:
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_linux_scroll_up)
            widget.bind("<Button-5>", self._on_linux_scroll_down)
        if self.scroll_x_enabled:
            widget.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

    def _on_mousewheel(self, event) -> str | None:
        if self.scroll_canvas is None:
            return None
        self.scroll_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _on_shift_mousewheel(self, event) -> str | None:
        if self.scroll_canvas is None:
            return None
        self.scroll_canvas.xview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _on_linux_scroll_up(self, _event) -> str | None:
        if self.scroll_canvas is None:
            return None
        self.scroll_canvas.yview_scroll(-1, "units")
        return "break"

    def _on_linux_scroll_down(self, _event) -> str | None:
        if self.scroll_canvas is None:
            return None
        self.scroll_canvas.yview_scroll(1, "units")
        return "break"


class ECGAnnotationApp(tk.Tk):
    def __init__(self, root_dir: Path) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.output_root = self.root_dir / OUTPUT_DIR
        self.output_root.mkdir(exist_ok=True)
        self.resume_state_path = self.output_root / "resume_state.json"
        self.resume_state = self.load_resume_state()
        self.is_closing = False

        self.title("ECG Annotation GUI")
        self.geometry("1600x980")
        self.minsize(1300, 860)

        initial_folder = self.resume_state.get("last_folder", SOURCE_FOLDERS[0])
        if initial_folder not in SOURCE_FOLDERS:
            initial_folder = SOURCE_FOLDERS[0]

        self.folder_var = tk.StringVar(value=initial_folder)
        self.search_var = tk.StringVar()
        self.label_var = tk.StringVar(value="Normal")
        self.assigned_to_var = tk.StringVar(value=DEFAULT_ASSIGNED_TO)
        self.notes_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.counter_var = tk.StringVar(value="")
        self.image_info_var = tk.StringVar(value="")
        self.pipeline_summary_var = tk.StringVar(value="Run digitization to inspect automatic overlap and interpolation summaries.")

        self.images_by_folder = {folder: self.discover_images(folder) for folder in SOURCE_FOLDERS}
        self.current_index = int(self.resume_state.get("folders", {}).get(initial_folder, 0))
        self.current_result = None

        self.anomaly_vars = {column: tk.BooleanVar(value=False) for _, column in ANOMALY_FIELDS}
        self.lead_vars = {column: tk.BooleanVar(value=False) for _, column in LEAD_FIELDS}
        self.bootstrap_annotation_logs()
        self.saved_annotations = self.load_annotations()

        self.build_ui()
        self.bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.load_current_image()

    def build_ui(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Folder").pack(side="left")
        folder_combo = ttk.Combobox(top, textvariable=self.folder_var, values=SOURCE_FOLDERS, state="readonly", width=18)
        folder_combo.pack(side="left", padx=(8, 14))
        folder_combo.bind("<<ComboboxSelected>>", self.on_folder_change)

        ttk.Label(top, text="Search Filename").pack(side="left")
        search_entry = ttk.Entry(top, textvariable=self.search_var, width=24)
        search_entry.pack(side="left", padx=(8, 8))
        search_entry.bind("<Return>", self.on_search_submit)
        ttk.Button(top, text="Find", command=self.search_current_folder).pack(side="left", padx=(0, 14))

        ttk.Button(top, text="Previous", command=self.show_previous).pack(side="left")
        ttk.Button(top, text="Next", command=self.show_next).pack(side="left", padx=(8, 14))
        ttk.Button(top, text="Run Digitization", command=self.run_digitization).pack(side="left")
        ttk.Button(top, text="Save Annotation", command=self.save_annotation).pack(side="left", padx=(8, 8))
        ttk.Button(top, text="Save + Next", command=self.save_and_next).pack(side="left")

        ttk.Label(top, text="Assigned To").pack(side="left", padx=(18, 6))
        ttk.Entry(top, textvariable=self.assigned_to_var, width=14).pack(side="left")
        ttk.Label(top, textvariable=self.counter_var).pack(side="left", padx=(18, 0))
        ttk.Label(top, textvariable=self.status_var, foreground="#0b5ed7").pack(side="right")

        content = ttk.Panedwindow(self, orient="horizontal")
        content.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(content, padding=8)
        right = ttk.Frame(content, padding=8)
        content.add(left, weight=2)
        content.add(right, weight=3)

        controls_frame = ttk.Frame(left)
        controls_frame.pack(fill="x")

        annotation_frame = ttk.LabelFrame(controls_frame, text="Annotation", padding=10)
        annotation_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(annotation_frame, text="Image Label").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            annotation_frame,
            text="Normal",
            value="Normal",
            variable=self.label_var,
            command=self.update_annotation_controls,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Radiobutton(
            annotation_frame,
            text="Anomalous",
            value="Anomalous",
            variable=self.label_var,
            command=self.update_annotation_controls,
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        ttk.Label(annotation_frame, text="Anomaly Types").grid(row=1, column=0, sticky="nw", pady=(8, 0))
        anomaly_box = ttk.Frame(annotation_frame)
        anomaly_box.grid(row=1, column=1, columnspan=2, sticky="we", padx=(8, 0), pady=(8, 0))
        self.anomaly_buttons: list[ttk.Checkbutton] = []
        for idx, (label, column) in enumerate(ANOMALY_FIELDS):
            button = ttk.Checkbutton(
                anomaly_box,
                text=label,
                variable=self.anomaly_vars[column],
                command=self.on_anomaly_toggle,
            )
            button.grid(row=idx, column=0, sticky="w", pady=2)
            self.anomaly_buttons.append(button)

        ttk.Label(annotation_frame, text="Notes").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(annotation_frame, textvariable=self.notes_var, width=45).grid(
            row=2,
            column=1,
            columnspan=2,
            sticky="we",
            padx=(8, 0),
            pady=(8, 0),
        )
        annotation_frame.columnconfigure(2, weight=1)

        lead_frame = ttk.LabelFrame(controls_frame, text="Anomalous Leads", padding=10)
        lead_frame.pack(fill="x", pady=(10, 0))
        self.lead_buttons: list[ttk.Checkbutton] = []
        for idx, (lead, column) in enumerate(LEAD_FIELDS):
            button = ttk.Checkbutton(
                lead_frame,
                text=lead,
                variable=self.lead_vars[column],
                command=self.on_lead_toggle,
            )
            row, col = divmod(idx, 4)
            button.grid(row=row, column=col, sticky="w", padx=(0, 12), pady=4)
            self.lead_buttons.append(button)

        summary_frame = ttk.LabelFrame(controls_frame, text="Current Pipeline Summary", padding=10)
        summary_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(
            summary_frame,
            textvariable=self.pipeline_summary_var,
            justify="left",
            foreground="#444444",
            wraplength=420,
        ).pack(fill="x")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Raw ECG Image").pack(anchor="w")
        ttk.Label(left, textvariable=self.image_info_var, foreground="#555555").pack(anchor="w", pady=(0, 6))

        self.raw_view = MatplotlibView(left, "No image loaded.")
        self.raw_view.pack(fill="both", expand=True)

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        self.trace_tab = ttk.Frame(notebook)
        self.crop_tab = ttk.Frame(notebook)
        self.overlap_tab = ttk.Frame(notebook)
        self.interpolation_tab = ttk.Frame(notebook)
        notebook.add(self.trace_tab, text="Y-Trace Views")
        notebook.add(self.crop_tab, text="Lead Crops")
        notebook.add(self.overlap_tab, text="Overlap Views")
        notebook.add(self.interpolation_tab, text="Interpolation Views")

        self.trace_view = MatplotlibView(
            self.trace_tab,
            "Run digitization to inspect y-trace overlays.",
            scroll_y=True,
            scroll_x=True,
        )
        self.trace_view.pack(fill="both", expand=True)
        self.crop_view = MatplotlibView(
            self.crop_tab,
            "Lead crop previews will appear here after digitization.",
            scroll_y=True,
            scroll_x=True,
        )
        self.crop_view.pack(fill="both", expand=True)
        self.overlap_view = MatplotlibView(
            self.overlap_tab,
            "Run digitization to inspect overlap views.",
            scroll_y=True,
            scroll_x=True,
        )
        self.overlap_view.pack(fill="both", expand=True)
        self.interpolation_view = MatplotlibView(
            self.interpolation_tab,
            "Run digitization to inspect interpolation views.",
            scroll_y=True,
            scroll_x=True,
        )
        self.interpolation_view.pack(fill="both", expand=True)

        self.update_annotation_controls()

    def bind_shortcuts(self) -> None:
        self.bind("<Left>", self.on_previous_shortcut)
        self.bind("<Right>", self.on_next_shortcut)
        self.bind("<Control-s>", self.on_save_shortcut)
        self.bind("<Control-Return>", self.on_save_next_shortcut)

    def is_typing_focus(self) -> bool:
        focused = self.focus_get()
        if focused is None:
            return False
        widget_class = focused.winfo_class()
        return widget_class in {"Entry", "TEntry", "Text", "TCombobox", "Combobox"}

    def on_previous_shortcut(self, _event=None) -> str | None:
        if self.is_typing_focus():
            return None
        self.show_previous()
        return "break"

    def on_next_shortcut(self, _event=None) -> str | None:
        if self.is_typing_focus():
            return None
        self.show_next()
        return "break"

    def on_save_shortcut(self, _event=None) -> str | None:
        if self.is_typing_focus():
            return None
        self.save_annotation()
        return "break"

    def on_save_next_shortcut(self, _event=None) -> str | None:
        if self.is_typing_focus():
            return None
        self.save_and_next()
        return "break"

    def discover_images(self, folder: str) -> list[Path]:
        folder_path = IMAGES_ROOT / folder
        if not folder_path.exists():
            return []
        return sorted(
            (path for path in folder_path.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()),
            key=natural_sort_key,
        )

    def current_image_list(self) -> list[Path]:
        return self.images_by_folder.get(self.folder_var.get(), [])

    def current_image_path(self) -> Path | None:
        images = self.current_image_list()
        if not images:
            return None
        self.current_index = max(0, min(self.current_index, len(images) - 1))
        return images[self.current_index]

    def annotation_key(self, image_path: Path) -> str:
        return f"{image_path.parent.name}/{image_path.name}"

    def on_folder_change(self, _event=None) -> None:
        self.current_index = int(self.resume_state.get("folders", {}).get(self.folder_var.get(), 0))
        self.load_current_image()

    def on_search_submit(self, _event=None) -> str:
        self.search_current_folder()
        return "break"

    def search_current_folder(self) -> None:
        query = self.search_var.get().strip().lower()
        if not query:
            self.set_status("Enter a filename to search in the current folder")
            return

        images = self.current_image_list()
        if not images:
            self.set_status("No images found in the current folder")
            return

        exact_match_index = None
        partial_match_index = None
        for index, image_path in enumerate(images):
            filename = image_path.name.lower()
            stem = image_path.stem.lower()
            if query == filename or query == stem:
                exact_match_index = index
                break
            if partial_match_index is None and query in filename:
                partial_match_index = index

        matched_index = exact_match_index if exact_match_index is not None else partial_match_index
        if matched_index is None:
            self.set_status(f"No match for '{self.search_var.get().strip()}' in {self.folder_var.get()}")
            return

        self.current_index = matched_index
        self.load_current_image()
        self.set_status(f"Found {images[matched_index].name} in {self.folder_var.get()}")

    def load_current_image(self) -> None:
        image_path = self.current_image_path()
        if image_path is None:
            self.counter_var.set("0 / 0")
            self.image_info_var.set("No images found in this folder.")
            self.raw_view.clear()
            self.clear_digitization_views()
            self.reset_annotation_form()
            self.set_status("No images found")
            return

        images = self.current_image_list()
        self.counter_var.set(f"{self.current_index + 1} / {len(images)}")
        self.image_info_var.set(str(image_path.relative_to(self.root_dir)))
        self.current_result = None
        self.remember_position()
        self.show_image(image_path)
        self.clear_digitization_views()
        self.restore_annotation(image_path)
        self.set_status(f"Loaded {image_path.name}")

    def show_image(self, image_path: Path) -> None:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            self.raw_view.clear()
            return

        figure, axis = plt.subplots(figsize=(7, 9), constrained_layout=True)
        axis.imshow(bgr_to_rgb(image_bgr))
        axis.set_title(image_path.name)
        axis.axis("off")
        self.raw_view.show_figure(figure)

    def clear_digitization_views(self) -> None:
        self.trace_view.clear()
        self.crop_view.clear()
        self.overlap_view.clear()
        self.interpolation_view.clear()
        self.pipeline_summary_var.set("Run digitization to inspect automatic overlap and interpolation summaries.")

    def run_digitization(self) -> None:
        image_path = self.current_image_path()
        if image_path is None:
            return

        self.set_status(f"Running digitization for {image_path.name}...")
        self.update_idletasks()
        try:
            self.current_result = digitize_ecg_image(image_path)
        except Exception as exc:
            self.current_result = None
            messagebox.showerror("Digitization failed", str(exc))
            self.set_status(f"Digitization failed for {image_path.name}")
            return

        self.render_digitization_figures()
        self.update_annotation_controls()
        self.set_status(f"Digitized {image_path.name}")

    def render_digitization_figures(self) -> None:
        if self.current_result is None:
            return

        self.trace_view.show_figure(build_trace_figure(self.current_result))
        self.crop_view.show_figure(build_crop_figure(self.current_result))
        self.overlap_view.show_figure(build_overlap_figure(self.current_result))
        self.interpolation_view.show_figure(build_interpolation_figure(self.current_result))
        self.pipeline_summary_var.set(self.build_pipeline_summary(self.current_result))

    def update_annotation_controls(self) -> None:
        is_anomalous = self.label_var.get() == "Anomalous"
        anomaly_state = "!disabled" if is_anomalous else "disabled"
        for button in self.anomaly_buttons:
            if anomaly_state == "disabled":
                button.state(["disabled"])
            else:
                button.state(["!disabled"])

        lead_state = "!disabled" if is_anomalous and self.current_result is not None else "disabled"
        for button in self.lead_buttons:
            if lead_state == "disabled":
                button.state(["disabled"])
            else:
                button.state(["!disabled"])

    def on_anomaly_toggle(self) -> None:
        if any(var.get() for var in self.anomaly_vars.values()):
            self.label_var.set("Anomalous")
        self.update_annotation_controls()

    def on_lead_toggle(self) -> None:
        if any(var.get() for var in self.lead_vars.values()):
            self.label_var.set("Anomalous")
        self.update_annotation_controls()

    def reset_annotation_form(self) -> None:
        self.label_var.set("Normal")
        self.notes_var.set("")
        for var in self.anomaly_vars.values():
            var.set(False)
        for var in self.lead_vars.values():
            var.set(False)
        self.update_annotation_controls()

    def restore_annotation(self, image_path: Path) -> None:
        self.reset_annotation_form()
        record = self.saved_annotations.get(self.annotation_key(image_path))
        if not record:
            return

        self.label_var.set(record.get("label", record.get("image_label", "Normal")) or "Normal")
        self.notes_var.set(record.get("notes", ""))
        old_anomaly_values = {
            value.strip().lower()
            for value in record.get("anomaly_type", "").split(";")
            if value.strip()
        }
        combined_selected = legacy_combined_anomaly_selected(record, old_anomaly_values)
        for label, column in ANOMALY_FIELDS:
            var = self.anomaly_vars[column]
            selected = csv_bool(record.get(column, "")) or label.lower() in old_anomaly_values
            if column in {"anomaly_overlap_with_text", "anomaly_overlap_of_signals"}:
                selected = selected or combined_selected
            var.set(selected)

        old_selected_leads = {
            lead.strip()
            for lead in record.get("anomalous_leads", "").split(";")
            if lead.strip()
        }
        for lead, column in LEAD_FIELDS:
            self.lead_vars[column].set(csv_bool(record.get(column, "")) or lead in old_selected_leads)
        self.update_annotation_controls()

    def save_annotation(self) -> bool:
        image_path = self.current_image_path()
        if image_path is None:
            return False

        image_label = self.label_var.get()
        selected_anomalies = [label for label, column in ANOMALY_FIELDS if self.anomaly_vars[column].get()]
        selected_leads = [lead for lead, column in LEAD_FIELDS if self.lead_vars[column].get()] if image_label == "Anomalous" else []

        if image_label == "Anomalous" and not selected_anomalies:
            messagebox.showwarning("Missing anomaly type", "Choose at least one anomaly type before saving.")
            return False

        if selected_leads and self.current_result is None:
            messagebox.showwarning(
                "Digitization required",
                "Run digitization before saving anomalous leads so their crops can be exported.",
            )
            return False

        if selected_leads:
            self.save_anomalous_crops(image_path, selected_leads)

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "filename": image_path.name,
            "assigned_to": self.assigned_to_var.get().strip() or DEFAULT_ASSIGNED_TO,
            "label": image_label,
            "notes": self.notes_var.get().strip(),
        }
        for _, column in ANOMALY_FIELDS:
            record[column] = csv_bool_text(image_label == "Anomalous" and self.anomaly_vars[column].get())
        for _, column in LEAD_FIELDS:
            record[column] = csv_bool_text(image_label == "Anomalous" and self.lead_vars[column].get())

        self.saved_annotations[self.annotation_key(image_path)] = record
        self.write_annotations(image_path.parent.name)
        self.set_status(f"Saved annotation for {image_path.name}")
        return True

    def save_and_next(self) -> None:
        if self.save_annotation():
            self.show_next()

    def save_anomalous_crops(self, image_path: Path, selected_leads: list[str]) -> list[str]:
        if self.current_result is None:
            return []

        output_dir = self.output_root / image_path.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[str] = []
        for lead in selected_leads:
            lead_result = self.current_result.lead_results.get(lead)
            if lead_result is None:
                continue

            out_name = f"{image_path.stem}_{lead}{image_path.suffix.lower()}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), lead_result.processed_trace_view_bgr)
            saved_files.append(str(out_path.relative_to(self.root_dir)))
        return saved_files

    def build_pipeline_summary(self, result) -> str:
        overlap_leads = [
            lead
            for lead, lead_result in result.lead_results.items()
            if lead_result.overlap_result is not None and lead_result.overlap_result.get("is_overlapping", False)
        ]
        interpolation_leads = [
            lead
            for lead, lead_result in result.lead_results.items()
            if lead_result.interpolation_mask is not None and int((lead_result.interpolation_mask > 0).sum()) > 0
        ]
        rhythm_overlap = result.rhythm_overlap_result or {}
        rhythm_overlap_text = (
            f"OVERLAP ({int(rhythm_overlap.get('overlap_columns', 0))} cols)"
            if rhythm_overlap.get("is_overlapping", False)
            else f"clean ({int(rhythm_overlap.get('overlap_columns', 0))} cols)"
        )
        rhythm_interp_cols = 0 if result.rhythm_interpolation_mask is None else int((result.rhythm_interpolation_mask > 0).sum())
        lines = [
            f"Auto overlap leads: {', '.join(overlap_leads) if overlap_leads else 'none'}",
            f"Auto interpolation leads: {', '.join(interpolation_leads) if interpolation_leads else 'none'}",
            f"Rhythm strip overlap: {rhythm_overlap_text}",
            f"Rhythm strip interpolated columns: {rhythm_interp_cols}",
        ]
        return "\n".join(lines)

    def annotation_csv_candidates(self, source_folder: str) -> list[Path]:
        return [
            self.output_root / source_folder / ANNOTATION_LOG_FILENAME,
            self.output_root / source_folder / LEGACY_ANNOTATION_LOG_FILENAME,
        ]

    def bootstrap_annotation_logs(self) -> None:
        for folder, rel_path in MERGED_CSV_BY_FOLDER.items():
            target_path = self.annotation_csv_path(folder)
            if target_path.exists():
                continue
            legacy_path = self.output_root / folder / LEGACY_ANNOTATION_LOG_FILENAME
            if legacy_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(legacy_path.read_text(encoding="utf-8"), encoding="utf-8")
                continue

            merged_path = self.root_dir / rel_path
            if not merged_path.exists():
                continue

            rows: list[dict[str, str]] = []
            with merged_path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    record = merged_row_to_annotation_record(row)
                    if record["filename"]:
                        rows.append(record)
            rows.sort(key=lambda item: natural_sort_text(item["filename"]))
            self.write_annotation_rows(folder, rows)

    def load_annotations(self) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        for folder in SOURCE_FOLDERS:
            csv_path = next((path for path in self.annotation_csv_candidates(folder) if path.exists()), None)
            if csv_path is None:
                continue
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if "label" not in row and "image_label" in row:
                        row["label"] = row.get("image_label", "")
                    filename = row.get("filename")
                    if not filename and row.get("image_path"):
                        filename = Path(row["image_path"]).name
                    if filename:
                        key = f"{folder}/{filename}"
                        row["filename"] = filename
                        records[key] = row
        return records

    def write_annotations(self, source_folder: str) -> None:
        rows = sorted(
            (
                row
                for key, row in self.saved_annotations.items()
                if key.startswith(f"{source_folder}/")
            ),
            key=lambda item: natural_sort_text(item["filename"]),
        )
        self.write_annotation_rows(source_folder, rows)

    def write_annotation_rows(self, source_folder: str, rows: list[dict[str, str]]) -> None:
        csv_path = self.annotation_csv_path(source_folder)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp",
            "filename",
            "assigned_to",
            "label",
            "notes",
        ]
        fieldnames.extend(column for _, column in ANOMALY_FIELDS)
        fieldnames.extend(column for _, column in LEAD_FIELDS)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])

    def annotation_csv_path(self, source_folder: str) -> Path:
        return self.output_root / source_folder / ANNOTATION_LOG_FILENAME

    def load_resume_state(self) -> dict[str, object]:
        default_state = {
            "last_folder": SOURCE_FOLDERS[0],
            "folders": {folder: 0 for folder in SOURCE_FOLDERS},
        }
        if not self.resume_state_path.exists():
            return default_state

        try:
            data = json.loads(self.resume_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default_state

        folders = default_state["folders"] | {
            folder: int(index)
            for folder, index in data.get("folders", {}).items()
            if folder in SOURCE_FOLDERS
        }
        last_folder = data.get("last_folder", SOURCE_FOLDERS[0])
        if last_folder not in SOURCE_FOLDERS:
            last_folder = SOURCE_FOLDERS[0]
        return {"last_folder": last_folder, "folders": folders}

    def remember_position(self) -> None:
        folder = self.folder_var.get()
        if folder not in SOURCE_FOLDERS:
            return
        folders = self.resume_state.setdefault("folders", {})
        folders[folder] = int(self.current_index)
        self.resume_state["last_folder"] = folder
        self.resume_state_path.write_text(json.dumps(self.resume_state, indent=2), encoding="utf-8")

    def show_previous(self) -> None:
        images = self.current_image_list()
        if not images:
            return
        self.current_index = (self.current_index - 1) % len(images)
        self.load_current_image()

    def show_next(self) -> None:
        images = self.current_image_list()
        if not images:
            return
        self.current_index = (self.current_index + 1) % len(images)
        self.load_current_image()

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def report_callback_exception(self, exc, value, tb) -> None:
        if self.is_closing and issubclass(exc, tk.TclError):
            return
        traceback.print_exception(exc, value, tb)
        try:
            messagebox.showerror("Application Error", f"{exc.__name__}: {value}")
        except tk.TclError:
            pass

    def on_close(self) -> None:
        if self.is_closing:
            return
        self.is_closing = True
        try:
            self.remember_position()
        except OSError:
            pass

        for view in (getattr(self, "raw_view", None), getattr(self, "trace_view", None), getattr(self, "crop_view", None)):
            if view is not None:
                view.clear(show_placeholder=False)

        plt.close("all")
        try:
            self.quit()
        except tk.TclError:
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    app = ECGAnnotationApp(root_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
