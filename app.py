#!/usr/bin/env python3
"""Modernised attendance demo app with a dashboard-style layout."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Optional
from uuid import uuid4

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageOps, ImageTk
from tkinter import messagebox, simpledialog, ttk

from MobileFaceNet import MobileFaceNetService
from pipelines.attendance import (
    AttendancePipeline,
    SessionCallbacks,
    DEFAULT_ATTENDANCE_LOG,
    DEFAULT_FACEBANK,
    DEFAULT_POWER_LOG,
    DEFAULT_SPOOF_WEIGHTS,
    DEFAULT_WEIGHTS,
)
from BlazeFace import BlazeFaceService
from utils.camera import open_video_source
from utils.cli import parse_main_args
from utils.device import select_device
from utils.overlay import PanelStyle, draw_text_panel
from utils.power import jetson_power_available
from utils.paths import logs_path
from utils.logging import CSV_FIELDS


SETTINGS_PATH = Path("app_settings.json")
POWER_AVAILABLE = jetson_power_available()


def _resolve_source(spec: str) -> int | str:
    text = (spec or "").strip()
    if not text:
        return 0
    return int(text) if text.isdigit() else text


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))


def _sanitize_identity_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name.strip())
    return safe or "person"


def _format_display_timestamp(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    normalized = text.replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(normalized)
    except ValueError:
        return text
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    local_time = moment.astimezone()
    return local_time.strftime("%b %d, %Y %I:%M %p")


def _next_facebank_index(face_dir: Path) -> int:
    prefix = "facebank_"
    max_index = 0
    files = sorted(face_dir.glob(f"{prefix}*.png")) + sorted(face_dir.glob(f"{prefix}*.jpg"))
    for image_path in files:
        suffix = image_path.stem[len(prefix) :]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


CLI_DEFAULTS = parse_main_args(
    default_weights=DEFAULT_WEIGHTS,
    default_facebank=DEFAULT_FACEBANK,
    default_spoof_weights=DEFAULT_SPOOF_WEIGHTS,
    default_attendance_log=DEFAULT_ATTENDANCE_LOG,
    default_power_log=DEFAULT_POWER_LOG,
    argv=[],
)

DEFAULT_CAMERA_WIDTH = int(getattr(CLI_DEFAULTS, "camera_width", 0) or 0)
DEFAULT_CAMERA_HEIGHT = int(getattr(CLI_DEFAULTS, "camera_height", 0) or 0)
DEFAULT_SOURCE = str(getattr(CLI_DEFAULTS, "source", "0"))
DEFAULT_DEVICE = getattr(CLI_DEFAULTS, "device", "cpu")
DEFAULT_IDENTITY_THR = float(getattr(CLI_DEFAULTS, "identity_thr", 0.9))
DEFAULT_SPOOF_THR = float(getattr(CLI_DEFAULTS, "spoof_thr", 0.9))
DEFAULT_DETECTOR_THR = float(getattr(CLI_DEFAULTS, "detector_thr", 0.8))

RESOLUTION_PRESETS: list[tuple[str, Optional[tuple[int, int]]]] = [
    ("Camera default", None),
    ("640 x 480 (VGA)", (640, 480)),
    ("1280 x 720 (HD)", (1280, 720)),
    ("1920 x 1080 (Full HD)", (1920, 1080)),
]

FPS_PRESETS: list[tuple[str, Optional[float]]] = [
    ("Camera default", None),
    ("24 FPS", 24.0),
    ("30 FPS", 30.0),
    ("60 FPS", 60.0),
]

LIVE_FEED_SIZE = (960, 540)
UI_PAD = 12


def _resolution_label_for_dims(width: Optional[int], height: Optional[int]) -> str:
    dims = None
    if width and height:
        dims = (width, height)
    for label, option in RESOLUTION_PRESETS:
        if option == dims:
            return label
    return RESOLUTION_PRESETS[0][0]


def _resolution_value(label: str) -> tuple[Optional[int], Optional[int]]:
    for text, dims in RESOLUTION_PRESETS:
        if text == label:
            return dims if dims is not None else (None, None)
    return (None, None)


def _fps_label_for_value(value: Optional[float]) -> str:
    for label, fps in FPS_PRESETS:
        if fps == value:
            return label
    return FPS_PRESETS[0][0]


def _fps_value(label: str) -> Optional[float]:
    for text, value in FPS_PRESETS:
        if text == label:
            return value
    try:
        return float(label)
    except ValueError:
        return None


DASHBOARD_EXPORT_DIR = logs_path("dashboard_runs")
LOG_BOOK_PATH = logs_path("dashboard_logbook.json")

METRICS_PANEL_STYLE = PanelStyle(
    bg_color=(18, 20, 25),
    alpha=0.75,
    padding=10,
    margin=16,
    font_scale=0.6,
    text_color=(240, 240, 240),
    title_color=(180, 220, 255),
    title_scale=0.65,
    title_thickness=2,
    thickness=1,
)


class FrameDisplay:
    """Lightweight OpenCV -> Tkinter bridge with optional letterboxing."""

    def __init__(self, target_label: ttk.Label, *, target_size: Optional[tuple[int, int]] = None) -> None:
        self.label = target_label
        self.target_size = target_size
        self._lock = threading.Lock()
        self._buffer: Optional[np.ndarray] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._active = False

    def start(self, interval_ms: int = 30) -> None:
        if self._active:
            return
        self._active = True
        self._pump(interval_ms)

    def stop(self) -> None:
        self._active = False

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buffer = frame.copy()

    def clear(self) -> None:
        with self._lock:
            self._buffer = None
        self._photo = None
        self.label.configure(image="")

    def _pump(self, interval_ms: int) -> None:
        if not self._active:
            return
        frame = None
        with self._lock:
            if self._buffer is not None:
                frame = self._buffer
                self._buffer = None
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if self.target_size:
                image = self._fit_to_target(image, self.target_size)
            self._photo = ImageTk.PhotoImage(image=image)
            self.label.configure(image=self._photo)
        self.label.after(interval_ms, lambda: self._pump(interval_ms))

    def _fit_to_target(self, image: Image.Image, target: tuple[int, int]) -> Image.Image:
        target_w, target_h = target
        if target_w <= 0 or target_h <= 0:
            return image
        contained = ImageOps.contain(image, target, method=Image.LANCZOS)
        if contained.size == target:
            return contained
        canvas = Image.new("RGB", target, color=(0, 0, 0))
        offset = ((target_w - contained.width) // 2, (target_h - contained.height) // 2)
        canvas.paste(contained, offset)
        return canvas



@dataclass
class DemoConfig:
    source: str | int
    device: str
    identity_threshold: float
    spoof_threshold: float
    enable_spoof: bool
    display_scores: bool
    show_metrics: bool
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    enable_power_logging: bool = False
    power_log_path: Path = Path(DEFAULT_POWER_LOG)
    power_interval: float = 1.0
    attendance_log: Path = Path(DEFAULT_ATTENDANCE_LOG)
    weights_path: Path = Path(DEFAULT_WEIGHTS)
    facebank_dir: Path = Path(DEFAULT_FACEBANK)
    spoof_weights_path: Path = Path(DEFAULT_SPOOF_WEIGHTS)
    frame_max_size: Optional[str] = None
    guidance_box_size: int = 0
    guidance_center_tolerance: float = 0.2
    guidance_size_tolerance: float = 0.2
    guidance_crop_padding: float = 0.2
    evaluation_duration: float = float(getattr(CLI_DEFAULTS, "evaluation_duration", 1.0))
    evaluation_mode: str = str(getattr(CLI_DEFAULTS, "evaluation_mode", "time"))
    evaluation_frames: int = int(getattr(CLI_DEFAULTS, "evaluation_frames", 30))

    def to_pipeline_args(self) -> SimpleNamespace:
        width = self.width or 0
        height = self.height or 0
        power_enabled = self.enable_power_logging and POWER_AVAILABLE
        defaults = CLI_DEFAULTS
        return SimpleNamespace(
            mode="guided",
            source=self.source,
            camera_width=width,
            camera_height=height,
            fps=self.fps,
            device=self.device,
            weights=str(self.weights_path),
            facebank=str(self.facebank_dir),
            update_facebank=False,
            tta=False,
            identity_thr=self.identity_threshold,
            detector_thr=DEFAULT_DETECTOR_THR,
            spoof_weights=str(self.spoof_weights_path),
            disable_spoof=not self.enable_spoof,
            spoof_thr=self.spoof_threshold,
            evaluation_duration=float(self.evaluation_duration),
            evaluation_mode=str(self.evaluation_mode),
            evaluation_frames=int(self.evaluation_frames),
            attendance_log=str(self.attendance_log),
            guidance_box_size=int(self.guidance_box_size),
            guidance_center_tolerance=float(self.guidance_center_tolerance),
            guidance_size_tolerance=float(self.guidance_size_tolerance),
            guidance_rotation_thr=float(getattr(defaults, "guidance_rotation_thr", 10.0)),
            guidance_hold_frames=int(getattr(defaults, "guidance_hold_frames", 10)),
            frame_max_size=self.frame_max_size,
            guidance_crop_padding=float(self.guidance_crop_padding),
            power_log=str(self.power_log_path),
            power_interval=self.power_interval,
            enable_power_log=power_enabled,
            disable_power_log=not power_enabled,
            quiet_power_log=True,
            show_summary_scores=self.display_scores,
        )


class AttendanceSessionController:
    """Wrap the AttendancePipeline for in-window sessions."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.pipeline: Optional[AttendancePipeline] = None
        self.capture_context = None
        self.session_runner = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._error_handler: Optional[Callable[[BaseException], None]] = None

    def start(self, callbacks: SessionCallbacks, *, on_error: Optional[Callable[[BaseException], None]] = None) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._error_handler = on_error
        args = self.config.to_pipeline_args()
        self.pipeline = AttendancePipeline(args)
        self.capture_context = self.pipeline.open_capture()
        self.pipeline.show_summary_scores = self.config.display_scores
        merged_callbacks = self._wrap_callbacks(callbacks)
        self.session_runner = self.pipeline.build_session_runner(
            self.capture_context.capture,
            window_name="Dashboard Session",
            window_limits=(self.capture_context.display_width, self.capture_context.display_height),
            callbacks=merged_callbacks,
        )
        self._thread = threading.Thread(target=self._run_session, args=(merged_callbacks,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self.pipeline and self.capture_context:
            self.pipeline.close_capture(self.capture_context)
        self.pipeline = None
        self.capture_context = None
        self.session_runner = None

    def _run_session(self, callbacks: SessionCallbacks) -> None:
        if self.pipeline is None or self.capture_context is None:
            return
        try:
            with self.pipeline.power_logger:
                self.pipeline.power_logger.set_activity("warmup")
                self.pipeline.warmup(
                    width=self.capture_context.resolved_width,
                    height=self.capture_context.resolved_height,
                )
                self.pipeline.power_logger.set_activity("ready")
                self.pipeline.run_guided_session(
                    self.session_runner,
                    context=self.capture_context,
                    window_name="Dashboard Session",
                    callbacks=callbacks,
                )
        except Exception as exc:
            self._handle_run_exception(exc)
        finally:
            if self.pipeline and self.capture_context:
                self.pipeline.close_capture(self.capture_context)
            self.capture_context = None

    def _wrap_callbacks(self, callbacks: SessionCallbacks) -> SessionCallbacks:
        def poll_cancel() -> bool:
            if self._stop.is_set():
                return True
            if callbacks.poll_cancel:
                return bool(callbacks.poll_cancel())
            return False

        def status_change(text: str) -> None:
            if self._stop.is_set():
                return
            if callbacks.on_status:
                callbacks.on_status(text)

        def stage_change(text: str) -> None:
            if self._stop.is_set():
                return
            if callbacks.on_stage_change:
                callbacks.on_stage_change(text)

        return SessionCallbacks(
            on_guidance_frame=callbacks.on_guidance_frame,
            on_verification_frame=callbacks.on_verification_frame,
            on_final_frame=callbacks.on_final_frame,
            poll_cancel=poll_cancel,
            wait_for_next_person=callbacks.wait_for_next_person,
            on_summary=callbacks.on_summary,
            on_status=status_change,
            on_stage_change=stage_change,
            on_metrics=callbacks.on_metrics,
        )

    def _handle_run_exception(self, exc: BaseException) -> None:
        self._stop.set()
        handler = self._error_handler
        if handler:
            try:
                handler(exc)
            finally:
                # One-time handler; prevent repeated notifications.
                self._error_handler = None

    def set_display_scores(self, value: bool) -> None:
        self.config.display_scores = value
        if self.pipeline is not None:
            self.pipeline.show_summary_scores = value

    def refresh_facebank(self) -> bool:
        """Reload the in-memory facebank embeddings for the active pipeline."""
        pipeline = self.pipeline
        if pipeline is None or pipeline.recogniser is None:
            return False
        pipeline.recogniser.rebuild_facebank(self.config.facebank_dir)
        return True


class StatusPanel(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.columnconfigure(1, weight=1)
        ttk.Label(self, text="Stage:", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self, text="Status:", font=("TkDefaultFont", 10, "bold")).grid(row=1, column=0, sticky="w")
        self.stage_var = tk.StringVar(value="Idle")
        self.status_var = tk.StringVar(value="Ready to start.")
        ttk.Label(self, textvariable=self.stage_var).grid(row=0, column=1, sticky="w")
        ttk.Label(self, textvariable=self.status_var).grid(row=1, column=1, sticky="w")

    def set_stage(self, text: str) -> None:
        self.stage_var.set(text)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)


class AttendanceLogBook:
    """Persist attendance sessions as named logs with book-style navigation."""

    def __init__(self, path: Path, *, max_entries: int = 500) -> None:
        self.path = path
        self.max_entries = max_entries
        self.pages: list[dict[str, object]] = []
        self.selected_page_id: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        raw_pages = payload.get("pages") if isinstance(payload, dict) else []
        normalized: list[dict[str, object]] = []
        if isinstance(raw_pages, list):
            for page in raw_pages:
                if isinstance(page, dict):
                    normalized.append(self._normalize_page(page))
        self.pages = normalized
        selected = payload.get("selected_page_id") if isinstance(payload, dict) else None
        if selected and self.get_page(selected):
            self.selected_page_id = selected
        elif self.pages:
            self.selected_page_id = self.pages[0]["id"]
        else:
            self.selected_page_id = None

    def _save(self) -> None:
        payload = {
            "version": 1,
            "selected_page_id": self.selected_page_id,
            "pages": self.pages,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_name(self.path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.path)
        except OSError:
            # Failed persistence should not crash the UI; warn via logs if needed.
            pass

    # ------------------------------------------------------------------ normalization
    def _normalize_page(self, raw: dict[str, object]) -> dict[str, object]:
        created = str(raw.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"))
        entries_raw = raw.get("entries")
        entries: list[dict[str, object]] = []
        if isinstance(entries_raw, list):
            for entry in entries_raw:
                if isinstance(entry, dict):
                    entries.append(self._normalize_entry(entry))
        return {
            "id": str(raw.get("id") or uuid4().hex),
            "name": str(raw.get("name") or "Session"),
            "created_at": created,
            "updated_at": str(raw.get("updated_at") or created),
            "entries": entries[: self.max_entries],
        }

    def _normalize_entry(self, entry: dict[str, object]) -> dict[str, object]:
        normalized = dict(entry)
        normalized["timestamp"] = str(entry.get("timestamp") or "")
        normalized["identity"] = str(entry.get("identity") or "Unknown")
        normalized["accepted"] = bool(entry.get("accepted"))
        normalized["source"] = str(entry.get("source") or "")
        return normalized

    # ------------------------------------------------------------------ queries / mutations
    @property
    def is_empty(self) -> bool:
        return not self.pages

    def list_pages(self) -> list[dict[str, object]]:
        return list(self.pages)

    def get_page(self, page_id: Optional[str]) -> Optional[dict[str, object]]:
        if not page_id:
            return None
        for page in self.pages:
            if page["id"] == page_id:
                return page
        return None

    def entries_for(self, page_id: Optional[str]) -> list[dict[str, object]]:
        page = self.get_page(page_id)
        if page is None:
            return []
        return list(page["entries"])

    def page_index(self, page_id: Optional[str]) -> int:
        if not page_id:
            return -1
        for idx, page in enumerate(self.pages):
            if page["id"] == page_id:
                return idx
        return -1

    def create_page(self, name: str) -> dict[str, object]:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        page = {
            "id": uuid4().hex,
            "name": name or "Session",
            "created_at": timestamp,
            "updated_at": timestamp,
            "entries": [],
        }
        self.pages.append(page)
        self.selected_page_id = page["id"]
        self._save()
        return page

    def rename_page(self, page_id: Optional[str], name: str) -> None:
        page = self.get_page(page_id)
        if page is None:
            return
        page["name"] = name or page["name"]
        page["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save()

    def add_entry(self, page_id: Optional[str], entry: dict[str, object]) -> Optional[dict[str, object]]:
        page = self.get_page(page_id)
        if page is None:
            return None
        normalized = self._normalize_entry(entry)
        page["entries"].insert(0, normalized)
        if len(page["entries"]) > self.max_entries:
            page["entries"] = page["entries"][: self.max_entries]
        page["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save()
        return normalized

    def delete_page(self, page_id: Optional[str]) -> None:
        if not page_id:
            return
        for idx, page in enumerate(self.pages):
            if page["id"] == page_id:
                self.pages.pop(idx)
                break
        else:
            return
        if self.selected_page_id == page_id:
            self.selected_page_id = self.pages[0]["id"] if self.pages else None
        self._save()

    def set_selected_page(self, page_id: Optional[str]) -> None:
        if not page_id:
            return
        if self.get_page(page_id) is None:
            return
        if self.selected_page_id == page_id:
            return
        self.selected_page_id = page_id
        self._save()


class AttendanceLog(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        columns = ("timestamp", "identity", "result", "source")
        self._columns = columns
        self._heading_labels = {
            "timestamp": "Timestamp",
            "identity": "Identity",
            "result": "Result",
            "source": "Source",
        }
        self._sort_column: Optional[str] = None
        self._sort_descending = False

        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=10)
        for col in columns:
            self.tree.heading(col, text=self._heading_labels[col], command=lambda c=col: self._sort_tree(c))
            self.tree.column(col, anchor="w")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def add_entry(self, *, timestamp: str, identity: str, accepted: bool, source: str, reapply_sort: bool = True) -> None:
        result = "Accepted" if accepted else "Rejected"
        display_time = _format_display_timestamp(timestamp)
        self.tree.insert("", 0, values=(display_time, identity, result, source))
        self._trim_rows()
        if reapply_sort:
            self._reapply_sort_if_needed()

    def _trim_rows(self) -> None:
        children = self.tree.get_children()
        for item in children[200:]:
            self.tree.delete(item)

    def clear(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

    def load_entries(self, entries: Iterable[dict[str, object]]) -> None:
        self.clear()
        buffered = list(entries)
        for entry in reversed(buffered):
            self.add_entry(
                timestamp=str(entry.get("timestamp", "")),
                identity=str(entry.get("identity", "")),
                accepted=bool(entry.get("accepted")),
                source=str(entry.get("source", "")),
                reapply_sort=False,
            )
        self._reapply_sort_if_needed()

    def _reapply_sort_if_needed(self) -> None:
        if self._sort_column:
            self._sort_tree(self._sort_column, toggle=False, descending=self._sort_descending)

    def _sort_tree(self, column: str, *, toggle: bool = True, descending: Optional[bool] = None) -> None:
        items = self.tree.get_children("")
        if not items:
            return
        if descending is None:
            if toggle and self._sort_column == column:
                descending = not self._sort_descending
            elif toggle:
                descending = False
            else:
                descending = self._sort_descending if self._sort_column == column else False
        sort_payload: list[tuple[tuple[int, object], str]] = []
        for item in items:
            value = self.tree.set(item, column)
            sort_payload.append((self._sort_key(column, value), item))
        sort_payload.sort(key=lambda entry: entry[0], reverse=bool(descending))
        for index, (_, item) in enumerate(sort_payload):
            self.tree.move(item, "", index)
        self._sort_column = column
        self._sort_descending = bool(descending)
        self._update_heading_labels()

    def _sort_key(self, column: str, raw_value: str) -> tuple[int, object]:
        text = (raw_value or "").strip()
        if column == "timestamp":
            parsed = self._parse_display_timestamp(text)
            if parsed is not None:
                return (0, parsed)
            return (1, text.lower())
        if column == "result":
            order = 0 if text.lower().startswith("accept") else 1
            return (order, text.lower())
        return (0, text.lower())

    @staticmethod
    def _parse_display_timestamp(value: str) -> Optional[datetime]:
        if not value:
            return None
        for fmt in ("%b %d, %Y %I:%M %p", "%b %d, %Y %I:%M%p"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _update_heading_labels(self) -> None:
        for col in self._columns:
            label = self._heading_labels[col]
            if col == self._sort_column:
                arrow = "▼" if self._sort_descending else "▲"
                label = f"{label} {arrow}"
            self.tree.heading(col, text=label)


class ControlPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        on_overlay_change=None,
    ) -> None:
        super().__init__(master)
        self.on_overlay_change = on_overlay_change or (lambda: None)
        self.source_var = tk.StringVar(value=str(DEFAULT_SOURCE))
        self.device_var = tk.StringVar(value=str(DEFAULT_DEVICE))
        self.identity_thr_var = tk.StringVar(value=f"{DEFAULT_IDENTITY_THR:.2f}")
        self.spoof_thr_var = tk.StringVar(value=f"{DEFAULT_SPOOF_THR:.2f}")
        self.enable_spoof_var = tk.BooleanVar(value=not getattr(CLI_DEFAULTS, "disable_spoof", False))
        self.display_scores_var = tk.BooleanVar(value=not getattr(CLI_DEFAULTS, "minimal_overlay", False))
        self.show_metrics_var = tk.BooleanVar(value=True)
        default_res_label = _resolution_label_for_dims(DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT)
        self.resolution_var = tk.StringVar(value=default_res_label)
        self.fps_var = tk.StringVar(value=FPS_PRESETS[0][0])
        self.power_enabled_var = tk.BooleanVar(value=bool(getattr(CLI_DEFAULTS, "enable_power_log", False)))
        self.power_path_var = tk.StringVar(value=str(getattr(CLI_DEFAULTS, "power_log", DEFAULT_POWER_LOG)))
        self.power_interval_var = tk.StringVar(value=str(getattr(CLI_DEFAULTS, "power_interval", 1.0)))
        self.guidance_box_size_var = tk.StringVar(value=str(int(getattr(CLI_DEFAULTS, "guidance_box_size", 224))))
        self.guidance_center_tol_var = tk.StringVar(value=str(getattr(CLI_DEFAULTS, "guidance_center_tolerance", 0.3)))
        self.guidance_size_tol_var = tk.StringVar(value=str(getattr(CLI_DEFAULTS, "guidance_size_tolerance", 0.3)))
        self.guidance_padding_var = tk.StringVar(value=str(getattr(CLI_DEFAULTS, "guidance_crop_padding", 0.5)))
        self.eval_mode_var = tk.StringVar(value=str(getattr(CLI_DEFAULTS, "evaluation_mode", "time")))
        self.eval_duration_var = tk.StringVar(value=str(getattr(CLI_DEFAULTS, "evaluation_duration", 1.0)))
        self.eval_frames_var = tk.StringVar(value=str(getattr(CLI_DEFAULTS, "evaluation_frames", 30)))
        self._load_settings()
        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)

        capture_frame = ttk.LabelFrame(left, text="Capture")
        capture_frame.grid(row=0, column=0, sticky="ew")
        capture_frame.columnconfigure(0, weight=1)
        ttk.Label(capture_frame, text="Camera Source").grid(row=0, column=0, sticky="w")
        ttk.Entry(capture_frame, textvariable=self.source_var).grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(capture_frame, text="Device").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            capture_frame,
            textvariable=self.device_var,
            values=("cpu", "cuda"),
            state="readonly",
        ).grid(row=3, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(capture_frame, text="Resolution preset").grid(row=4, column=0, sticky="w")
        ttk.Combobox(
            capture_frame,
            textvariable=self.resolution_var,
            values=[label for label, _ in RESOLUTION_PRESETS],
            state="readonly",
        ).grid(row=5, column=0, sticky="ew")
        ttk.Label(capture_frame, text="Target FPS").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            capture_frame,
            textvariable=self.fps_var,
            values=[label for label, _ in FPS_PRESETS],
            state="readonly",
        ).grid(row=7, column=0, sticky="ew")

        thresholds_frame = ttk.LabelFrame(right, text="Thresholds & Toggles")
        thresholds_frame.grid(row=0, column=0, sticky="ew")
        thresholds_frame.columnconfigure(0, weight=1)
        ttk.Label(thresholds_frame, text="Identity threshold").grid(row=0, column=0, sticky="w")
        ttk.Entry(thresholds_frame, textvariable=self.identity_thr_var).grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(thresholds_frame, text="Spoof threshold").grid(row=2, column=0, sticky="w")
        ttk.Entry(thresholds_frame, textvariable=self.spoof_thr_var).grid(row=3, column=0, sticky="ew", pady=(0, 6))
        ttk.Checkbutton(thresholds_frame, text="Enable anti-spoof", variable=self.enable_spoof_var).grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Checkbutton(
            thresholds_frame,
            text="Show identity scores in summary",
            variable=self.display_scores_var,
            command=self.on_overlay_change,
        ).grid(row=5, column=0, sticky="w")
        ttk.Checkbutton(
            thresholds_frame,
            text="Show metrics overlay (FPS/timings)",
            variable=self.show_metrics_var,
            command=self.on_overlay_change,
        ).grid(row=6, column=0, sticky="w")

        guidance_frame = ttk.LabelFrame(right, text="Guidance Settings")
        guidance_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        guidance_frame.columnconfigure(0, weight=1)
        ttk.Label(guidance_frame, text="Box size (px)").grid(row=0, column=0, sticky="w")
        ttk.Entry(guidance_frame, textvariable=self.guidance_box_size_var).grid(row=1, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(guidance_frame, text="Center tolerance (0-1)").grid(row=2, column=0, sticky="w")
        ttk.Entry(guidance_frame, textvariable=self.guidance_center_tol_var).grid(row=3, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(guidance_frame, text="Size tolerance (0-1)").grid(row=4, column=0, sticky="w")
        ttk.Entry(guidance_frame, textvariable=self.guidance_size_tol_var).grid(row=5, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(guidance_frame, text="Crop padding (0-1)").grid(row=6, column=0, sticky="w")
        ttk.Entry(guidance_frame, textvariable=self.guidance_padding_var).grid(row=7, column=0, sticky="ew")

        verification_frame = ttk.LabelFrame(left, text="Verification Capture")
        verification_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        verification_frame.columnconfigure(0, weight=1)
        ttk.Label(verification_frame, text="Mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            verification_frame,
            textvariable=self.eval_mode_var,
            values=("time", "frames"),
            state="readonly",
        ).grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(verification_frame, text="Duration (s)").grid(row=2, column=0, sticky="w")
        ttk.Entry(verification_frame, textvariable=self.eval_duration_var).grid(row=3, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(verification_frame, text="Frame limit").grid(row=4, column=0, sticky="w")
        ttk.Entry(verification_frame, textvariable=self.eval_frames_var).grid(row=5, column=0, sticky="ew")

        power_frame = ttk.LabelFrame(left, text="Jetson Power Logging")
        power_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        power_frame.columnconfigure(0, weight=1)
        power_toggle = ttk.Checkbutton(power_frame, text="Enable power logging", variable=self.power_enabled_var)
        power_toggle.grid(row=0, column=0, sticky="w")
        ttk.Label(power_frame, text="Log path").grid(row=1, column=0, sticky="w", pady=(4, 0))
        power_path_entry = ttk.Entry(power_frame, textvariable=self.power_path_var)
        power_path_entry.grid(row=2, column=0, sticky="ew")
        ttk.Label(power_frame, text="Interval (s)").grid(row=3, column=0, sticky="w", pady=(4, 0))
        power_interval_entry = ttk.Entry(power_frame, textvariable=self.power_interval_var)
        power_interval_entry.grid(row=4, column=0, sticky="ew")
        if not POWER_AVAILABLE:
            power_toggle.state(["disabled"])
            power_path_entry.state(["disabled"])
            power_interval_entry.state(["disabled"])
            self.power_enabled_var.set(False)
            ttk.Label(
                power_frame,
                text="Power logging unavailable on this device.",
                foreground="#aa0000",
            ).grid(row=5, column=0, sticky="w", pady=(6, 0))

        save_row = ttk.Frame(self)
        save_row.grid(row=1, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(save_row, text="Save Settings", command=self._handle_save_settings, width=18).pack()

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.source_var.set(str(data.get("source", self.source_var.get())))
        device_value = str(data.get("device", self.device_var.get()))
        if device_value == "cuda:0":
            device_value = "cuda"
        self.device_var.set(device_value)
        preset = data.get("resolution_preset")
        resolution_labels = [label for label, _ in RESOLUTION_PRESETS]
        if preset in resolution_labels:
            self.resolution_var.set(preset)
        elif isinstance(preset, str) and "x" in preset.lower():
            try:
                width_str, height_str = preset.lower().split("x", 1)
                width = int(width_str.strip())
                height = int(height_str.strip())
                self.resolution_var.set(_resolution_label_for_dims(width, height))
            except ValueError:
                pass
        fps_value = data.get("fps")
        if fps_value:
            fps_labels = [label for label, _ in FPS_PRESETS]
            if fps_value in fps_labels:
                self.fps_var.set(fps_value)
            else:
                try:
                    numeric = float(fps_value)
                    self.fps_var.set(_fps_label_for_value(numeric))
                except (TypeError, ValueError):
                    pass
        self.identity_thr_var.set(str(data.get("identity_threshold", self.identity_thr_var.get())))
        self.spoof_thr_var.set(str(data.get("spoof_threshold", self.spoof_thr_var.get())))
        self.enable_spoof_var.set(bool(data.get("enable_spoof", True)))
        self.display_scores_var.set(bool(data.get("display_scores", True)))
        self.show_metrics_var.set(bool(data.get("show_metrics", True)))
        eval_mode = str(data.get("evaluation_mode", self.eval_mode_var.get())).strip().lower()
        if eval_mode in ("time", "frames"):
            self.eval_mode_var.set(eval_mode)
        self.eval_duration_var.set(str(data.get("evaluation_duration", self.eval_duration_var.get())))
        self.eval_frames_var.set(str(data.get("evaluation_frames", self.eval_frames_var.get())))
        self.power_enabled_var.set(bool(data.get("power_enabled", False)) and POWER_AVAILABLE)
        self.power_path_var.set(str(data.get("power_path", self.power_path_var.get())))
        self.power_interval_var.set(str(data.get("power_interval", self.power_interval_var.get())))
        self.guidance_box_size_var.set(str(data.get("guidance_box_size", self.guidance_box_size_var.get())))
        self.guidance_center_tol_var.set(str(data.get("guidance_center_tolerance", self.guidance_center_tol_var.get())))
        self.guidance_size_tol_var.set(str(data.get("guidance_size_tolerance", self.guidance_size_tol_var.get())))
        self.guidance_padding_var.set(str(data.get("guidance_crop_padding", self.guidance_padding_var.get())))

    def _handle_save_settings(self) -> None:
        payload = {
            "source": self.source_var.get(),
            "device": self.device_var.get(),
            "resolution_preset": self.resolution_var.get(),
            "fps": self.fps_var.get(),
            "identity_threshold": self.identity_thr_var.get(),
            "spoof_threshold": self.spoof_thr_var.get(),
            "enable_spoof": bool(self.enable_spoof_var.get()),
            "display_scores": bool(self.display_scores_var.get()),
            "show_metrics": bool(self.show_metrics_var.get()),
            "evaluation_mode": self.eval_mode_var.get(),
            "evaluation_duration": self.eval_duration_var.get(),
            "evaluation_frames": self.eval_frames_var.get(),
            "power_enabled": bool(self.power_enabled_var.get()),
            "power_path": self.power_path_var.get(),
            "power_interval": self.power_interval_var.get(),
            "guidance_box_size": self.guidance_box_size_var.get(),
            "guidance_center_tolerance": self.guidance_center_tol_var.get(),
            "guidance_size_tolerance": self.guidance_size_tol_var.get(),
            "guidance_crop_padding": self.guidance_padding_var.get(),
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            messagebox.showinfo("Settings saved", f"Settings stored in {SETTINGS_PATH.resolve()}", parent=self)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Unable to save settings: {exc}", parent=self)

    def build_demo_config(self) -> Optional[DemoConfig]:
        try:
            identity_thr = float(self.identity_thr_var.get())
            spoof_thr = float(self.spoof_thr_var.get())
            power_interval = float(self.power_interval_var.get() or "1.0")
            box_size = int(float(self.guidance_box_size_var.get() or "0"))
            center_tol = float(self.guidance_center_tol_var.get() or "0.3")
            size_tol = float(self.guidance_size_tol_var.get() or "0.3")
            crop_padding = float(self.guidance_padding_var.get() or "0.5")
            eval_duration = float(self.eval_duration_var.get() or "0.9")
            eval_frames = int(float(self.eval_frames_var.get() or "30"))
        except ValueError as exc:
            messagebox.showerror("Invalid settings", f"Numeric format error: {exc}", parent=self)
            return None
        box_size = max(0, box_size)
        center_tol = max(0.0, center_tol)
        size_tol = max(0.0, size_tol)
        crop_padding = max(0.0, crop_padding)
        eval_duration = max(0.1, eval_duration)
        eval_frames = max(1, eval_frames)
        eval_mode = self.eval_mode_var.get().strip().lower()
        if eval_mode not in ("time", "frames"):
            eval_mode = "time"
        width, height = self._current_resolution()
        fps_value = self._current_fps()
        return DemoConfig(
            source=_resolve_source(self.source_var.get()),
            device=self.device_var.get(),
            identity_threshold=identity_thr,
            spoof_threshold=spoof_thr,
            enable_spoof=self.enable_spoof_var.get(),
            display_scores=self.display_scores_var.get(),
            show_metrics=self.show_metrics_var.get(),
            width=width,
            height=height,
            fps=fps_value,
            enable_power_logging=self.power_enabled_var.get() and POWER_AVAILABLE,
            power_log_path=Path(self.power_path_var.get().strip() or DEFAULT_POWER_LOG),
            power_interval=max(0.1, power_interval),
            frame_max_size=None,
            guidance_box_size=box_size,
            guidance_center_tolerance=center_tol,
            guidance_size_tolerance=size_tol,
            guidance_crop_padding=crop_padding,
            evaluation_duration=eval_duration,
            evaluation_mode=eval_mode,
            evaluation_frames=eval_frames,
        )

    def _current_resolution(self) -> tuple[Optional[int], Optional[int]]:
        label = self.resolution_var.get()
        width, height = _resolution_value(label)
        return width, height

    def _current_fps(self) -> Optional[float]:
        label = self.fps_var.get()
        value = _fps_value(label)
        if value is not None or label in [text for text, _ in FPS_PRESETS]:
            return value
        try:
            return float(label)
        except ValueError:
            return None

class FacebankPanel(ttk.Frame):
    """Display and manage facebank identities."""

    def __init__(self, master: tk.Misc, *, on_register, on_refresh) -> None:
        super().__init__(master)
        self.on_register = on_register
        self.on_refresh = on_refresh
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._active_identity: Optional[str] = None
        self._thumb_labels: dict[Path, tk.Label] = {}
        self._selected_samples: set[Path] = set()

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=UI_PAD, pady=(UI_PAD, UI_PAD // 2))
        header.columnconfigure(0, weight=1)

        self.count_var = tk.StringVar(value="Identities: 0")
        ttk.Label(
            header,
            textvariable=self.count_var,
            font=("TkDefaultFont", 11, "bold"),
            anchor="center",
        ).grid(row=0, column=0, sticky="ew")

        content = ttk.Frame(self)
        content.grid(row=1, column=0, sticky="nsew", padx=UI_PAD, pady=(UI_PAD, UI_PAD // 2))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        list_frame = ttk.Frame(content)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, UI_PAD // 2))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(list_frame, height=18)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_identity_select)

        thumbs_frame = ttk.LabelFrame(content, text="Samples")
        thumbs_frame.grid(row=0, column=1, sticky="nsew", padx=(UI_PAD // 2, 0))
        thumbs_frame.columnconfigure(0, weight=1)
        thumbs_frame.rowconfigure(0, weight=1)
        self.thumbs_canvas = tk.Canvas(thumbs_frame, highlightthickness=0)
        self.thumbs_canvas.grid(row=0, column=0, sticky="nsew")
        vscroll = ttk.Scrollbar(thumbs_frame, orient="vertical", command=self.thumbs_canvas.yview)
        vscroll.grid(row=0, column=1, sticky="ns")
        self.thumbs_canvas.configure(yscrollcommand=vscroll.set)
        thumbs_inner = ttk.Frame(self.thumbs_canvas)
        self._thumb_window = self.thumbs_canvas.create_window((0, 0), window=thumbs_inner, anchor="nw")
        self.thumbs_inner = thumbs_inner
        self._thumb_images: list[ImageTk.PhotoImage] = []
        self._thumb_labels: dict[Path, tk.Label] = {}
        self._selected_samples: set[Path] = set()
        temp_label = tk.Label(self)
        self._sample_default_bg = temp_label.cget("background")
        temp_label.destroy()
        self.thumbs_canvas.bind("<Configure>", lambda event: self._relayout_thumbnails())
        self.thumbs_canvas.bind_all("<MouseWheel>", self._on_thumb_scroll)
        self.thumbs_inner.bind("<Configure>", lambda event: self._relayout_thumbnails())

        footer = ttk.Frame(self)
        footer.grid(row=2, column=0, sticky="ew", padx=UI_PAD, pady=(UI_PAD, UI_PAD))
        footer.columnconfigure(0, weight=1)
        button_row = ttk.Frame(footer)
        button_row.pack(side="left")
        ttk.Button(button_row, text="Register User", command=self.on_register, width=18).grid(row=0, column=0, padx=4)
        ttk.Button(button_row, text="Rename User", command=self._rename_selected, width=18).grid(row=0, column=1, padx=4)
        ttk.Button(button_row, text="Delete User", command=self._delete_selected, width=18).grid(row=0, column=2, padx=4)
        samples_row = ttk.Frame(footer)
        samples_row.pack(side="right")
        self.delete_samples_button = ttk.Button(
            samples_row, text="Delete Selected", width=18, command=self._delete_selected_samples, state="disabled"
        )
        self.delete_samples_button.grid(row=0, column=0, padx=(0, 6))
        self.clear_selection_button = ttk.Button(
            samples_row, text="Cancel Selection", width=18, command=self._clear_sample_selection, state="disabled"
        )
        self.clear_selection_button.grid(row=0, column=1)

        self.refresh()

    def refresh(self) -> None:
        names: List[str] = []
        DEFAULT_FACEBANK.mkdir(parents=True, exist_ok=True)
        if DEFAULT_FACEBANK.exists():
            for entry in sorted(DEFAULT_FACEBANK.iterdir()):
                if entry.is_dir() and not entry.name.startswith("."):
                    names.append(entry.name)
        self.listbox.delete(0, tk.END)
        for name in names:
            self.listbox.insert(tk.END, name)
        self.count_var.set(f"Identities: {len(names)}")
        if names:
            if self._active_identity in names:
                target = self._active_identity
            else:
                target = names[0]
                self._active_identity = target
            idx = names.index(target)
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
        else:
            self._active_identity = None
        self._load_thumbnails()

    def _selected_names(self) -> List[str]:
        indices = self.listbox.curselection()
        return [self.listbox.get(idx) for idx in indices]

    def _on_identity_select(self, event=None) -> None:
        names = self._selected_names()
        self._active_identity = names[0] if names else None
        self._load_thumbnails()

    def _load_thumbnails(self) -> None:
        for widget in self.thumbs_inner.winfo_children():
            widget.destroy()
        self._thumb_images.clear()
        self._thumb_labels = {}
        self._selected_samples.clear()
        selection = self._selected_names()
        if len(selection) != 1:
            self._update_sample_action_state()
            self._relayout_thumbnails()
            return
        person = selection[0]
        person_dir = DEFAULT_FACEBANK / person
        if not person_dir.exists():
            self._update_sample_action_state()
            self._relayout_thumbnails()
            return
        image_paths = sorted(
            [p for p in person_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
        if not image_paths:
            self._update_sample_action_state()
            self._relayout_thumbnails()
            return
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.fit(img, (120, 120), method=Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._thumb_images.append(photo)
                label = tk.Label(self.thumbs_inner, image=photo, borderwidth=2, relief="flat", cursor="hand2")
                label.grid(row=0, column=0, padx=4, pady=4)
                label.bind("<Button-1>", lambda event, p=img_path: self._toggle_sample_selection(p))
                self._thumb_labels[img_path] = label
            except Exception:
                continue
        self._relayout_thumbnails()
        self._update_sample_action_state()

    def _update_thumb_scrollregion(self) -> None:
        self.thumbs_canvas.configure(scrollregion=self.thumbs_canvas.bbox("all"))
        self.thumbs_canvas.itemconfigure(self._thumb_window, width=self.thumbs_canvas.winfo_width())

    def _relayout_thumbnails(self) -> None:
        children = self.thumbs_inner.winfo_children()
        if not children:
            self._update_thumb_scrollregion()
            return
        width = max(self.thumbs_canvas.winfo_width(), 1)
        thumb_w = 120 + 8  # image width + padding
        cols = max(1, int(width / thumb_w))
        for idx, child in enumerate(children):
            row = idx // cols
            col = idx % cols
            child.grid_configure(row=row, column=col, padx=4, pady=4)
        self.thumbs_inner.update_idletasks()
        self._update_thumb_scrollregion()

    def _on_thumb_scroll(self, event) -> None:
        widget_path = str(event.widget)
        if widget_path.startswith(str(self.thumbs_canvas)) or widget_path.startswith(str(self.thumbs_inner)):
            self.thumbs_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _toggle_sample_selection(self, path: Path) -> None:
        label = self._thumb_labels.get(path)
        if label is None:
            return
        if path in self._selected_samples:
            self._selected_samples.remove(path)
            label.config(relief="flat", background=self._sample_default_bg)
        else:
            self._selected_samples.add(path)
            label.config(relief="solid", background="#ffedc2")
        self._update_sample_action_state()

    def _clear_sample_selection(self) -> None:
        for path in list(self._selected_samples):
            label = self._thumb_labels.get(path)
            if label:
                label.config(relief="flat", background=self._sample_default_bg)
        self._selected_samples.clear()
        self._update_sample_action_state()

    def _update_sample_action_state(self) -> None:
        if self._selected_samples:
            self.delete_samples_button.state(("!disabled",))
            self.clear_selection_button.state(("!disabled",))
        else:
            self.delete_samples_button.state(("disabled",))
            self.clear_selection_button.state(("disabled",))

    def _delete_selected_samples(self) -> None:
        if not self._selected_samples:
            messagebox.showinfo("Delete samples", "Select at least one sample.", parent=self)
            return
        count = len(self._selected_samples)
        person = self._active_identity or "identity"
        if not messagebox.askyesno(
            "Delete samples",
            f"Delete {count} sample(s) for '{person}'? This cannot be undone.",
            parent=self,
        ):
            return
        errors: list[str] = []
        for path in list(self._selected_samples):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"{Path(path).name}: {exc}")
        if errors:
            messagebox.showerror("Delete samples", "\n".join(errors), parent=self)
        self._selected_samples.clear()
        self._load_thumbnails()
        self._request_refresh(
            status_message="Deleting samples... refreshing facebank.",
            success_message="Facebank refreshed after deleting samples.",
        )

    def _rename_selected(self) -> None:
        names = self._selected_names()
        if len(names) != 1:
            messagebox.showinfo("Rename identity", "Select a single identity to rename.", parent=self)
            return
        current = names[0]
        new_name = simpledialog.askstring("Rename identity", f"Enter new name for '{current}':", parent=self)
        if not new_name:
            return
        sanitized = _sanitize_identity_name(new_name)
        if not sanitized:
            messagebox.showerror("Rename identity", "Name cannot be empty.", parent=self)
            return
        if sanitized == current:
            return
        src = DEFAULT_FACEBANK / current
        dst = DEFAULT_FACEBANK / sanitized
        if dst.exists():
            messagebox.showerror("Rename identity", f"'{sanitized}' already exists.", parent=self)
            return
        try:
            src.rename(dst)
        except OSError as exc:
            messagebox.showerror("Rename identity", f"Unable to rename: {exc}", parent=self)
            return
        self.refresh()
        self._request_refresh(
            status_message=f"Renaming '{current}'... refreshing facebank.",
            success_message=f"Facebank refreshed after renaming '{current}' to '{sanitized}'.",
        )

    def _delete_selected(self) -> None:
        names = self._selected_names()
        if not names:
            messagebox.showinfo("Delete identity", "Select at least one identity to delete.", parent=self)
            return
        count = len(names)
        plural = "ies" if count != 1 else "y"
        confirm = messagebox.askyesno(
            "Delete identities",
            f"Delete {count} identit{plural}? This cannot be undone.",
            parent=self,
        )
        if not confirm:
            return
        for name in names:
            target = DEFAULT_FACEBANK / name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        self.refresh()
        word = "identity" if count == 1 else "identities"
        self._request_refresh(
            status_message=f"Deleting {count} {word}... refreshing facebank.",
            success_message=f"Facebank refreshed after deleting {count} {word}.",
        )

    def _request_refresh(self, *, status_message: str, success_message: str) -> None:
        if callable(self.on_refresh):
            try:
                self.on_refresh(status_message=status_message, success_message=success_message)
            except TypeError:
                # Backward compatibility if callback signature has not been updated.
                self.on_refresh()


class RegistrationSession:
    """Background camera capture used during dashboard registration mode."""

    MAX_SAMPLES: Optional[int] = None

    def __init__(self, config: DemoConfig, submit_frame: Callable[[np.ndarray], None]) -> None:
        self.config = config
        self.submit_frame = submit_frame
        self.capture: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.samples: List[np.ndarray] = []
        self.latest_frame: Optional[np.ndarray] = None
        self.detector: Optional[BlazeFaceService] = None
        self._device = select_device(self.config.device)
        self._frame_lock = threading.Lock()

    def start(self) -> None:
        self.capture = open_video_source(
            self.config.source,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
        )
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError("Unable to open camera source for registration.")
        self.detector = BlazeFaceService(score_threshold=DEFAULT_DETECTOR_THR, device=self._device)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self) -> None:
        capture = self.capture
        if capture is None:
            return
        while not self.stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                continue
            with self._frame_lock:
                self.latest_frame = frame.copy()
            self.submit_frame(frame)

    def capture_sample(self) -> int:
        with self._frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            raise RuntimeError("Camera not ready yet.")
        if self.max_samples and len(self.samples) >= self.max_samples:
            raise RuntimeError("Maximum samples captured.")
        if self.detector is None:
            raise RuntimeError("Face detector unavailable.")
        detections = self.detector.detect(frame)
        best = max(detections, key=lambda det: det.score) if detections else None
        if best is None:
            raise RuntimeError("No face detected; align before capturing.")
        aligned = self.detector.detector.align_face(frame, best)
        if aligned is None:
            raise RuntimeError("Unable to align face; adjust your pose and try again.")
        self.samples.append(aligned)
        return len(self.samples)

    def save_samples(self, identity: str) -> int:
        if not self.samples:
            raise RuntimeError("No samples captured.")
        face_dir = DEFAULT_FACEBANK / identity
        face_dir.mkdir(parents=True, exist_ok=True)
        start_index = _next_facebank_index(face_dir)
        saved = 0
        for idx, frame in enumerate(self.samples, start=start_index):
            target = face_dir / f"facebank_{idx:03d}.png"
            if cv2.imwrite(str(target), frame):
                saved += 1
        self.samples.clear()
        return saved

    def stop(self) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=1.0)
            self.thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.detector = None

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def max_samples(self) -> int:
        return self.MAX_SAMPLES or 0

class DashboardApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Attendance Dashboard")
        self.root.geometry("1250x720")
        style = ttk.Style(self.root)
        style.configure("TNotebook.Tab", padding=(14, 6), font=("TkDefaultFont", 10, "bold"))
        self.controller: Optional[AttendanceSessionController] = None
        self.log_book = AttendanceLogBook(LOG_BOOK_PATH)
        self._current_log_id: Optional[str] = None
        self._active_log_id: Optional[str] = None
        self._next_session_log_id: Optional[str] = None

        self.status_panel = StatusPanel(self.root)
        self.status_panel.pack(fill="x", padx=12, pady=(12, 6))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.notebook = notebook

        self.session_tab = ttk.Frame(notebook)
        self.session_tab.columnconfigure(0, weight=1)
        self.session_tab.rowconfigure(0, weight=1)
        notebook.add(self.session_tab, text="Live Session")

        video_frame = ttk.LabelFrame(self.session_tab, text="Live Feed")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=UI_PAD, pady=(UI_PAD, UI_PAD))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        feed_container = ttk.Frame(video_frame, width=LIVE_FEED_SIZE[0], height=LIVE_FEED_SIZE[1])
        feed_container.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        feed_container.grid_propagate(False)
        feed_container.rowconfigure(0, weight=1)
        feed_container.columnconfigure(0, weight=1)
        video_label = ttk.Label(feed_container, anchor="center")
        video_label.place(relx=0.5, rely=0.5, anchor="center")
        self.video_display = FrameDisplay(video_label, target_size=LIVE_FEED_SIZE)
        self.video_display.start()

        session_footer = ttk.Frame(self.session_tab)
        session_footer.grid(row=1, column=0, sticky="ew", pady=(0, 12), padx=12)
        session_footer.columnconfigure(0, weight=1)
        self.session_buttons_frame = ttk.Frame(session_footer)
        self.session_buttons_frame.grid(row=0, column=0)
        self.start_button = ttk.Button(
            self.session_buttons_frame,
            text="Start Session",
            command=self._start_session_from_controls,
            width=18,
        )
        self.start_button.grid(row=0, column=0, padx=(0, 6))
        self.pause_button = ttk.Button(
            self.session_buttons_frame,
            text="Stop Session",
            command=self._stop_session,
            width=18,
        )
        self.pause_button.grid(row=0, column=1, padx=6)
        self.pause_button.state(["disabled"])
        self.resume_button = ttk.Button(
            self.session_buttons_frame,
            text="Resume Session",
            command=self._resume_session_from_log,
            width=18,
        )
        self.resume_button.grid(row=0, column=2, padx=(6, 0))
        self.session_buttons_frame.columnconfigure((0, 1, 2), weight=1)

        self.registration_frame = ttk.Frame(session_footer)
        self.registration_frame.grid(row=0, column=0, sticky="ew")
        self.registration_frame.columnconfigure(0, weight=1)
        self.registration_frame.columnconfigure(1, weight=1)
        self.registration_frame.grid_remove()
        self.registration_name_var = tk.StringVar()
        ttk.Label(self.registration_frame, text="Identity name", anchor="center").grid(row=0, column=0, columnspan=2, sticky="ew")
        name_entry = ttk.Entry(self.registration_frame, textvariable=self.registration_name_var, justify="center", width=30)
        name_entry.grid(row=1, column=0, columnspan=2, pady=(0, 6))
        reg_buttons = ttk.Frame(self.registration_frame)
        reg_buttons.grid(row=2, column=0, columnspan=2, pady=(0, 6))
        for col in range(3):
            reg_buttons.columnconfigure(col, weight=0)
        ttk.Button(reg_buttons, text="Capture Sample", command=self._capture_registration_sample, width=18).grid(row=0, column=0, padx=4)
        ttk.Button(reg_buttons, text="Save Samples", command=self._save_registration_samples, width=18).grid(row=0, column=1, padx=4)
        ttk.Button(reg_buttons, text="Cancel", command=self._cancel_registration, width=18).grid(row=0, column=2, padx=4)
        self.registration_status_var = tk.StringVar(value="Registration idle.")
        ttk.Label(self.registration_frame, textvariable=self.registration_status_var, anchor="center").grid(row=3, column=0, columnspan=2, sticky="ew")

        facebank_tab = ttk.Frame(notebook)
        notebook.add(facebank_tab, text="Facebank")
        self.facebank_panel = FacebankPanel(
            facebank_tab,
            on_register=self._start_registration,
            on_refresh=self._on_facebank_refresh,
        )
        self.facebank_panel.pack(fill="both", expand=True, padx=12, pady=12)

        log_tab = ttk.Frame(notebook)
        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(1, weight=1)
        notebook.add(log_tab, text="Attendance Logs")

        log_header = ttk.Frame(log_tab)
        log_header.grid(row=0, column=0, sticky="ew", padx=UI_PAD, pady=(UI_PAD, UI_PAD // 2))
        log_header.columnconfigure(0, weight=1)
        self.log_title_var = tk.StringVar(value="Loading logs...")
        ttk.Label(
            log_header,
            textvariable=self.log_title_var,
            font=("TkDefaultFont", 11, "bold"),
            anchor="center",
        ).grid(row=0, column=0, sticky="ew")

        self.log_panel = AttendanceLog(log_tab)
        self.log_panel.grid(row=1, column=0, sticky="nsew", padx=UI_PAD, pady=(0, UI_PAD // 2))

        log_footer = ttk.Frame(log_tab)
        log_footer.grid(row=2, column=0, sticky="ew", padx=UI_PAD, pady=(UI_PAD // 2, UI_PAD))
        log_footer.columnconfigure(0, weight=1)
        log_footer.columnconfigure(1, weight=1)

        nav_frame = ttk.Frame(log_footer)
        nav_frame.grid(row=0, column=0, sticky="w")
        self.prev_log_button = ttk.Button(nav_frame, text="<< Previous Log", width=18, command=self._select_prev_log)
        self.prev_log_button.grid(row=0, column=0, padx=(0, 6))
        self.next_log_button = ttk.Button(nav_frame, text="Next Log >>", width=18, command=self._select_next_log)
        self.next_log_button.grid(row=0, column=1, padx=(6, 0))

        actions_frame = ttk.Frame(log_footer)
        actions_frame.grid(row=0, column=1, sticky="e")
        ttk.Button(actions_frame, text="New Log", command=self._create_log).grid(row=0, column=0, padx=4)
        ttk.Button(actions_frame, text="Rename Log", command=self._rename_log).grid(row=0, column=1, padx=4)
        ttk.Button(actions_frame, text="Delete Log", command=self._delete_log).grid(row=0, column=2, padx=4)
        ttk.Button(actions_frame, text="Export CSV", command=self._export_log).grid(row=0, column=3, padx=4)

        settings_tab = ttk.Frame(notebook)
        settings_tab.columnconfigure(0, weight=1)
        notebook.add(settings_tab, text="Settings")
        self.settings_tab = settings_tab
        self.control_panel = ControlPanel(
            settings_tab,
            on_overlay_change=self._on_overlay_change,
        )
        self.control_panel.pack(fill="both", expand=True, padx=12, pady=12)

        self._latest_metrics: Dict[str, float] = {}
        self._show_metrics_overlay = self.control_panel.show_metrics_var.get()
        self._display_scores = self.control_panel.display_scores_var.get()
        self._session_active = False
        self._session_identities: set[str] = set()
        self._next_person_event = threading.Event()
        self._awaiting_next = False
        self.registration_session: Optional[RegistrationSession] = None
        self._session_button_mode = "start"
        self._failed_entries_by_log: dict[str, dict[str, dict[str, object]]] = {}
        self._set_start_button_mode("start")
        self._initialize_logbook_state()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Ensure embeddings reflect the latest facebank on startup.
        self.root.after(
            100,
            lambda: self._on_facebank_refresh(
                status_message="Preparing facebank...",
                success_message="Facebank ready.",
            ),
        )

    def run(self) -> None:
        self.root.mainloop()

    # ------------------------------------------------------------------ logbook helpers
    def _initialize_logbook_state(self) -> None:
        selected = self.log_book.selected_page_id
        if selected and self.log_book.get_page(selected):
            self._current_log_id = selected
        elif not self.log_book.is_empty:
            first = self.log_book.list_pages()[0]
            self._current_log_id = first["id"]
            self.log_book.set_selected_page(self._current_log_id)
        else:
            self._current_log_id = None
        self._active_log_id = None
        if self._current_log_id is not None:
            self._failed_entries_by_log.setdefault(self._current_log_id, {})
        self._refresh_log_display()

    def _refresh_log_display(self) -> None:
        if self._current_log_id is None:
            self.log_panel.clear()
        else:
            entries = self.log_book.entries_for(self._current_log_id)
            self.log_panel.load_entries(entries)
        self.log_title_var.set(self._format_log_label())
        self._update_log_nav_state()

    def _format_log_label(self) -> str:
        pages = self.log_book.list_pages()
        if not pages or not self._current_log_id:
            return "No logs yet"
        index = self.log_book.page_index(self._current_log_id)
        if index < 0:
            return "No logs yet"
        page = pages[index]
        badges: list[str] = []
        if self._session_active and self._active_log_id == page["id"]:
            badges.append("live")
        suffix = f" ({', '.join(badges)})" if badges else ""
        return f"Log {index + 1}/{len(pages)}: {page['name']}{suffix}"

    def _update_log_nav_state(self) -> None:
        pages = self.log_book.list_pages()
        index = self.log_book.page_index(self._current_log_id)
        if len(pages) <= 1 or index <= 0:
            self.prev_log_button.state(["disabled"])
        else:
            self.prev_log_button.state(["!disabled"])
        if len(pages) <= 1 or index == len(pages) - 1 or index == -1:
            self.next_log_button.state(["disabled"])
        else:
            self.next_log_button.state(["!disabled"])

    def _select_log_dialog(self, pages: list[dict[str, object]], current_index: int) -> Optional[str]:
        dialog = tk.Toplevel(self.root)
        dialog.title("Resume Session")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("420x320")
        dialog.update_idletasks()
        self._center_window(dialog)
        ttk.Label(dialog, text="Select a log to resume:", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 8))
        listbox = tk.Listbox(dialog, height=min(12, len(pages)), exportselection=False)
        listbox.grid(row=1, column=0, columnspan=2, padx=16, sticky="nsew")
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=1)
        for idx, page in enumerate(pages):
            entry_count = len(page.get("entries", []))
            created = page.get("created_at", "")[:16]
            label = f"{page['name']}  ({entry_count} entries, {created})"
            listbox.insert("end", label)
        if 0 <= current_index < len(pages):
            listbox.selection_set(current_index)
            listbox.see(current_index)
        selection: dict[str, Optional[str]] = {"value": None}

        def confirm(event=None):
            chosen = listbox.curselection()
            if not chosen:
                return
            selection["value"] = pages[chosen[0]]["id"]
            dialog.destroy()

        def cancel() -> None:
            selection["value"] = None
            dialog.destroy()

        listbox.bind("<Double-Button-1>", confirm)
        ttk.Button(dialog, text="Cancel", command=cancel).grid(row=2, column=0, pady=16, padx=16, sticky="w")
        ttk.Button(dialog, text="Resume", command=confirm).grid(row=2, column=1, pady=16, padx=16, sticky="e")
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        self.root.wait_window(dialog)
        return selection["value"]

    def _center_window(self, window: tk.Toplevel) -> None:
        window.update_idletasks()
        parent = self.root
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
        win_w = window.winfo_width()
        win_h = window.winfo_height()
        x = parent_x + (parent_w // 2) - (win_w // 2)
        y = parent_y + (parent_h // 2) - (win_h // 2)
        window.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _set_current_log(self, page_id: Optional[str]) -> None:
        if not page_id:
            return
        if self.log_book.get_page(page_id) is None:
            return
        self._current_log_id = page_id
        self.log_book.set_selected_page(page_id)
        self._ensure_log_bucket(page_id)
        self._refresh_log_display()

    def _select_prev_log(self) -> None:
        pages = self.log_book.list_pages()
        index = self.log_book.page_index(self._current_log_id)
        if index > 0:
            self._set_current_log(pages[index - 1]["id"])

    def _select_next_log(self) -> None:
        pages = self.log_book.list_pages()
        index = self.log_book.page_index(self._current_log_id)
        if 0 <= index < len(pages) - 1:
            self._set_current_log(pages[index + 1]["id"])

    def _generate_default_log_name(self) -> str:
        return datetime.now().strftime("Session %Y-%m-%d %H:%M")

    def _ensure_log_bucket(self, log_id: str) -> dict[str, dict[str, object]]:
        return self._failed_entries_by_log.setdefault(log_id, {})

    def _failure_key(self, identity: str, source: str) -> str:
        return f"{identity}::{source}"

    def _require_active_log_id(self) -> str:
        target_page_id = self._active_log_id or self._current_log_id
        if target_page_id is None:
            fallback_page = self.log_book.create_page(self._generate_default_log_name())
            target_page_id = fallback_page["id"]
            self._active_log_id = target_page_id
            self._set_current_log(target_page_id)
        self._ensure_log_bucket(target_page_id)
        return target_page_id

    def _create_log(self) -> None:
        suggested = self._generate_default_log_name()
        name = simpledialog.askstring(
            "New attendance log",
            "Name this log:",
            parent=self.root,
            initialvalue=suggested,
        )
        if name is None:
            return
        final_name = name.strip() or suggested
        page = self.log_book.create_page(final_name)
        self._set_current_log(page["id"])
        self._ensure_log_bucket(page["id"])
        self.status_panel.set_status(f"Created log '{final_name}'.")

    def _rename_log(self) -> None:
        page = self.log_book.get_page(self._current_log_id)
        if page is None:
            messagebox.showinfo("No log", "Create a log before renaming.", parent=self.root)
            return
        new_name = simpledialog.askstring(
            "Rename log",
            "Enter a new name for this log:",
            parent=self.root,
            initialvalue=page["name"],
        )
        if new_name is None:
            return
        final_name = new_name.strip()
        if not final_name:
            return
        self.log_book.rename_page(page["id"], final_name)
        self._refresh_log_display()
        self.status_panel.set_status(f"Renamed log to '{final_name}'.")

    def _resume_session_from_log(self) -> None:
        pages = self.log_book.list_pages()
        if not pages:
            messagebox.showinfo("Resume session", "No logs available to resume.", parent=self.root)
            return
        current_index = self.log_book.page_index(self._current_log_id)
        selection = self._select_log_dialog(pages, current_index)
        if selection is None:
            return
        page = self.log_book.get_page(selection)
        if page is None:
            messagebox.showerror("Resume session", "Unable to load the selected log.", parent=self.root)
            return
        if self._session_active and self._active_log_id == page["id"]:
            messagebox.showinfo("Resume session", f"'{page['name']}' is already live.", parent=self.root)
            return
        self._next_session_log_id = page["id"]
        self._set_current_log(page["id"])
        self._ensure_log_bucket(page["id"])
        self.status_panel.set_status(f"Next session will resume '{page['name']}'.")
        self._refresh_log_display()

    def _prepare_log_for_new_session(self) -> bool:
        if self._next_session_log_id:
            page = self.log_book.get_page(self._next_session_log_id)
            if page is not None:
                self._active_log_id = page["id"]
                self._set_current_log(page["id"])
                self._ensure_log_bucket(page["id"])
                self.status_panel.set_status(f"Resuming log '{page['name']}'.")
                self._next_session_log_id = None
                return True
            self._next_session_log_id = None
        suggested = self._generate_default_log_name()
        prompt = (
            "Name this session's attendance log.\n"
            "A new log will be created so you can revisit it later."
        )
        name = simpledialog.askstring(
            "New session log",
            prompt,
            parent=self.root,
            initialvalue=suggested,
        )
        if name is None:
            return False
        final_name = name.strip() or suggested
        page = self.log_book.create_page(final_name)
        self._active_log_id = page["id"]
        self._set_current_log(page["id"])
        self._ensure_log_bucket(page["id"])
        self.status_panel.set_status(f"Logging session '{final_name}'.")
        return True

    def _seed_session_identities(self) -> None:
        """Populate the duplicate tracker using existing accepted entries in the active log."""
        self._session_identities.clear()
        target_page_id = self._active_log_id or self._current_log_id
        if not target_page_id:
            return
        entries = self.log_book.entries_for(target_page_id)
        for entry in entries:
            identity = entry.get("identity")
            if entry.get("accepted") and identity and identity != "Unknown":
                self._session_identities.add(str(identity))

    def _delete_log(self) -> None:
        page = self.log_book.get_page(self._current_log_id)
        if page is None:
            messagebox.showinfo("Delete log", "Select a log to delete.", parent=self.root)
            return
        if self._session_active and self._active_log_id == page["id"]:
            messagebox.showinfo("Delete log", "Stop the active session before deleting its log.", parent=self.root)
            return
        confirm = messagebox.askyesno(
            "Delete log",
            f"Delete '{page['name']}' and all attendance entries?",
            parent=self.root,
        )
        if not confirm:
            return
        index = self.log_book.page_index(page["id"])
        self.log_book.delete_page(page["id"])
        self._failed_entries_by_log.pop(page["id"], None)
        if self._next_session_log_id == page["id"]:
            self._next_session_log_id = None
        if self._active_log_id == page["id"]:
            self._active_log_id = None
        pages = self.log_book.list_pages()
        if pages:
            new_index = min(max(index, 0), len(pages) - 1)
            self._set_current_log(pages[new_index]["id"])
        else:
            self._current_log_id = None
            self._refresh_log_display()
        self.status_panel.set_status(f"Deleted log '{page['name']}'.")

    def _start_session_from_controls(self) -> None:
        if self.registration_session is not None:
            messagebox.showinfo("Registration active", "Finish or cancel registration before starting a session.", parent=self.root)
            return
        if self.controller is not None:
            return
        config = self.control_panel.build_demo_config()
        if config is None:
            self.notebook.select(self.settings_tab)
            return
        try:
            select_device(config.device)
        except Exception as exc:
            messagebox.showerror("Device error", str(exc), parent=self.root)
            return
        if not self._prepare_log_for_new_session():
            return
        self._start_session(config)

    def _start_session(self, config: DemoConfig) -> None:
        self._stop_session(preserve_active_log=True)
        self.controller = AttendanceSessionController(config)
        self._show_metrics_overlay = config.show_metrics
        self._display_scores = config.display_scores
        self._seed_session_identities()
        self._next_person_event = threading.Event()
        self._awaiting_next = False
        self._session_active = True
        self._set_start_button_mode("disabled")
        self.pause_button.state(["!disabled"])
        self._show_session_controls()
        callbacks = SessionCallbacks(
            on_guidance_frame=self._submit_frame,
            on_verification_frame=self._submit_frame,
            on_final_frame=self._submit_frame,
            poll_cancel=lambda: False,
            wait_for_next_person=self._wait_for_next,
            on_summary=self._handle_summary,
            on_status=lambda text: self._ui_call(self.status_panel.set_status, text),
            on_stage_change=lambda text: self._ui_call(self.status_panel.set_stage, text),
            on_metrics=self._handle_metrics,
        )
        try:
            self.controller.start(callbacks, on_error=self._handle_session_error)
            self.status_panel.set_status("Align face to begin verification.")
        except Exception as exc:
            self._stop_session()
            messagebox.showerror("Session error", _format_exception(exc), parent=self.root)

    def _stop_session(self, *, preserve_active_log: bool = False) -> None:
        retained_log = self._active_log_id if preserve_active_log else None
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        self._session_active = False
        self._active_log_id = retained_log
        self._awaiting_next = False
        self._next_person_event.set()
        self.video_display.clear()
        self.status_panel.set_stage("Idle")
        self.status_panel.set_status("Session stopped.")
        self.pause_button.state(["disabled"])
        self._set_start_button_mode("start")
        self._latest_metrics = {}
        self._session_identities.clear()

    def _handle_session_error(self, exc: BaseException) -> None:
        self.root.after(0, lambda: self._report_session_error(exc))

    def _report_session_error(self, exc: BaseException) -> None:
        self._stop_session()
        self.status_panel.set_status("Session ended due to an error.")
        messagebox.showerror("Session error", _format_exception(exc), parent=self.root)

    def _wait_for_next(self) -> bool:
        self._awaiting_next = True
        while self.controller is not None and not self.controller._stop.is_set() and self._session_active:
            if self._next_person_event.wait(0.2):
                self._next_person_event.clear()
                self._awaiting_next = False
                return True
        return False

    def _handle_summary(self, summary: Dict[str, object]) -> None:
        self.root.after(0, lambda: self._process_summary(summary))

    def _process_summary(self, summary: Dict[str, object]) -> None:
        timestamp = summary.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")
        identity = summary.get("identity") or "Unknown"
        accepted = bool(summary.get("accepted"))
        source = summary.get("source") or self.control_panel.source_var.get()
        duplicate = accepted and identity != "Unknown" and identity in self._session_identities
        if accepted and not duplicate and identity != "Unknown":
            self._session_identities.add(identity)
        if not duplicate:
            entry = dict(summary)
            entry.update(
                {
                    "timestamp": timestamp,
                    "identity": identity,
                    "accepted": accepted,
                    "source": str(source),
                }
            )
            if accepted:
                target_page_id = self._require_active_log_id()
                self.log_book.add_entry(target_page_id, entry)
                if self._current_log_id == target_page_id:
                    self.log_panel.add_entry(
                        timestamp=str(entry.get("timestamp", "")),
                        identity=str(entry.get("identity", "")),
                        accepted=True,
                        source=str(entry.get("source", "")),
                    )
                failures = self._failed_entries_by_log.get(target_page_id)
                if failures:
                    failures.pop(self._failure_key(identity, str(source)), None)
            else:
                target_page_id = self._require_active_log_id()
                failures = self._ensure_log_bucket(target_page_id)
                failures[self._failure_key(identity, str(source))] = entry
        result_text: str
        if duplicate:
            result_text = f"{identity}: Already attended"
        elif accepted:
            result_text = f"{identity}: Attended"
        else:
            result_text = f"{identity}: Try again"
        self.status_panel.set_status(result_text)
        self._awaiting_next = True
        self._set_start_button_mode("next")

    def _handle_metrics(self, metrics: Dict[str, float]) -> None:
        self._latest_metrics = metrics

    def _submit_frame(self, frame: np.ndarray) -> None:
        if self.registration_session is not None:
            self.video_display.submit(frame)
            return
        annotated = frame
        if self._show_metrics_overlay:
            annotated = self._apply_metrics_overlay(annotated)
        self.video_display.submit(annotated)

    def _signal_next_person(self) -> None:
        if not self._awaiting_next:
            return
        self._set_start_button_mode("disabled")
        self._awaiting_next = False
        self._next_person_event.set()

    def _start_registration(self) -> None:
        if self.controller is not None:
            self._stop_session()
        if self.registration_session is not None:
            return
        config = self.control_panel.build_demo_config()
        if config is None:
            self.notebook.select(self.settings_tab)
            return
        try:
            session = RegistrationSession(config, self._submit_frame)
            session.start()
        except Exception as exc:
            messagebox.showerror("Registration", f"Unable to start registration: {exc}", parent=self.root)
            return
        self.registration_session = session
        self.registration_name_var.set("")
        self.registration_status_var.set("Capture samples using the controls buttons.")
        self._show_registration_controls()
        self.notebook.select(self.session_tab)

    def _capture_registration_sample(self) -> None:
        session = self.registration_session
        if session is None:
            return
        try:
            count = session.capture_sample()
        except RuntimeError as exc:
            messagebox.showerror("Capture sample", str(exc), parent=self.root)
            return
        self.registration_status_var.set(f"Captured {count} sample(s).")

    def _save_registration_samples(self) -> None:
        session = self.registration_session
        if session is None:
            return
        identity_raw = self.registration_name_var.get().strip()
        if not identity_raw:
            messagebox.showinfo("Registration", "Enter an identity name before saving.", parent=self.root)
            return
        sanitized = _sanitize_identity_name(identity_raw)
        if not sanitized:
            messagebox.showerror("Registration", "Identity name is invalid.", parent=self.root)
            return
        if session.sample_count == 0:
            messagebox.showinfo("Registration", "Capture at least one sample before saving.", parent=self.root)
            return
        face_dir = DEFAULT_FACEBANK / sanitized
        if face_dir.exists():
            proceed = messagebox.askyesno(
                "Append samples",
                f"The identity '{sanitized}' exists. Append new samples?",
                parent=self.root,
            )
            if not proceed:
                return
        try:
            saved = session.save_samples(sanitized)
        except RuntimeError as exc:
            messagebox.showerror("Registration", str(exc), parent=self.root)
            return
        self.registration_status_var.set(f"Saved {saved} samples for {sanitized}.")
        messagebox.showinfo("Registration complete", f"Saved {saved} samples for {sanitized}.", parent=self.root)
        self.status_panel.set_status("Updating facebank with new registration...")
        self._rebuild_facebank_async(
            session.config,
            success_message=f"Facebank refreshed with new samples for {sanitized}.",
        )
        self._end_registration()
        self._refresh_facebank()

    def _cancel_registration(self) -> None:
        if self.registration_session is None:
            return
        self._end_registration(message="Registration cancelled.")

    def _end_registration(self, message: Optional[str] = None) -> None:
        if self.registration_session is not None:
            self.registration_session.stop()
            self.registration_session = None
        if message:
            self.registration_status_var.set(message)
        else:
            self.registration_status_var.set("Registration idle.")
        self.registration_name_var.set("")
        self.video_display.clear()
        self._show_session_controls()

    def _apply_metrics_overlay(self, frame: np.ndarray) -> np.ndarray:
        metrics = self._latest_metrics
        if not metrics:
            return frame
        lines: list[tuple[str, tuple[int, int, int]]] = []
        fps = metrics.get("avg_fps") or metrics.get("fps")
        if fps is not None:
            lines.append((f"FPS: {fps:5.1f}", (255, 255, 0)))
        detector = metrics.get("avg_detector_ms") or metrics.get("detector_ms")
        if detector is not None:
            lines.append((f"Detector: {detector:.1f} ms", (0, 255, 255)))
        recog = metrics.get("avg_recognition_ms") or metrics.get("recognition_ms")
        if recog is not None:
            lines.append((f"Recogniser: {recog:.1f} ms", (0, 255, 255)))
        spoof = metrics.get("avg_spoof_ms") or metrics.get("spoof_ms")
        if spoof is not None:
            lines.append((f"Spoof: {spoof:.1f} ms", (0, 255, 255)))
        if not lines:
            return frame
        return draw_text_panel(
            frame,
            lines,
            anchor="top-right",
            title="Performance",
            style=METRICS_PANEL_STYLE,
        )

    def _ui_call(self, func, *args, **kwargs) -> None:
        if func is None:
            return
        self.root.after(0, lambda f=func, a=args, kw=kwargs: f(*a, **kw))

    def _set_start_button_mode(self, mode: str) -> None:
        self._session_button_mode = mode
        if mode == "start":
            self.start_button.config(text="Start Session", command=self._start_session_from_controls)
            self.start_button.state(["!disabled"])
        elif mode == "next":
            self.start_button.config(text="Next Person", command=self._signal_next_person)
            self.start_button.state(["!disabled"])
        elif mode == "disabled":
            self.start_button.state(["disabled"])

    def _show_session_controls(self) -> None:
        self.registration_frame.grid_remove()
        self.session_buttons_frame.grid()
        if not self._session_active:
            self._set_start_button_mode("start")
            self.pause_button.state(["disabled"])

    def _show_registration_controls(self) -> None:
        self.session_buttons_frame.grid_remove()
        self.registration_frame.grid()
        self.pause_button.state(["disabled"])
        self._set_start_button_mode("start")

    def _on_overlay_change(self) -> None:
        new_metrics = self.control_panel.show_metrics_var.get()
        self._show_metrics_overlay = new_metrics
        new_scores = self.control_panel.display_scores_var.get()
        self._display_scores = new_scores
        if self.controller is not None:
            self.controller.set_display_scores(new_scores)

    def _on_register(self) -> None:
        self._start_registration()

    def _export_log(self) -> None:
        page = self.log_book.get_page(self._current_log_id)
        if page is None:
            messagebox.showinfo("Export attendance", "Select a log to export.", parent=self.root)
            return
        page_id = page["id"]
        success_entries = list(self.log_book.entries_for(page_id))
        failure_entries = list(self._failed_entries_by_log.get(page_id, {}).values())
        combined_entries = success_entries + failure_entries
        if not combined_entries:
            messagebox.showinfo("Export attendance", "No entries to export for this log.", parent=self.root)
            return
        export_dir = DASHBOARD_EXPORT_DIR
        export_dir.mkdir(parents=True, exist_ok=True)
        page_slug = _sanitize_identity_name(page["name"])
        date_stamp = datetime.now().strftime("%Y%m%d")
        export_path = export_dir / f"{page_slug}_{date_stamp}.csv"
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            detail_fields = [field for field in CSV_FIELDS if field not in ("timestamp", "identity", "source")]
            header = ["timestamp", "identity", "result", "source", *detail_fields]
            with export_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                ordered_entries = sorted(
                    combined_entries,
                    key=lambda item: str(item.get("timestamp", "")),
                )
                for entry in ordered_entries:
                    result = "Accepted" if entry.get("accepted") else "Rejected"
                    row_values = [
                        entry.get("timestamp", ""),
                        entry.get("identity", ""),
                        result,
                        entry.get("source", ""),
                    ]
                    for field in detail_fields:
                        row_values.append(entry.get(field, ""))
                    writer.writerow(row_values)
            messagebox.showinfo("Export attendance", f"CSV for '{page['name']}' written to {export_path}", parent=self.root)
        except Exception as exc:
            messagebox.showerror("Export failed", f"Unable to export log: {exc}", parent=self.root)

    def _refresh_facebank(self) -> None:
        if getattr(self, "facebank_panel", None) is not None:
            self.facebank_panel.refresh()

    def _on_facebank_refresh(
        self,
        *,
        status_message: Optional[str] = None,
        success_message: Optional[str] = None,
    ) -> None:
        config = self.control_panel.build_demo_config()
        if config is None:
            self.notebook.select(self.settings_tab)
            return
        self.status_panel.set_status(status_message or "Refreshing facebank...")
        self._rebuild_facebank_async(config, success_message=success_message or "Facebank refreshed.")

    def _rebuild_facebank_async(self, config: DemoConfig, *, success_message: str = "Facebank refreshed.") -> None:
        def worker() -> None:
            try:
                MobileFaceNetService(
                    weights_path=config.weights_path,
                    facebank_dir=config.facebank_dir,
                    recognition_threshold=config.identity_threshold,
                    refresh_facebank=True,
                )
                controller = self.controller
                reload_warning: Optional[str] = None
                if controller is not None:
                    try:
                        controller.refresh_facebank()
                    except Exception as refresh_exc:  # pragma: no cover - UI thread handles message
                        reload_warning = _format_exception(refresh_exc)

                def notify() -> None:
                    if reload_warning:
                        messagebox.showwarning(
                            "Facebank refresh",
                            "Facebank files were rebuilt, but the live session could not reload them automatically.\n"
                            "Restart the session to apply the latest embeddings.\n\n"
                            f"Details:\n{reload_warning}",
                            parent=self.root,
                        )
                        self._facebank_refresh_complete("Facebank refreshed (restart session to apply).")
                    else:
                        self._facebank_refresh_complete(success_message)

                self.root.after(0, notify)
            except Exception as exc:
                err = _format_exception(exc)
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Facebank refresh failed",
                        err,
                        parent=self.root,
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _facebank_refresh_complete(self, message: str) -> None:
        self.status_panel.set_status(message)
        self._refresh_facebank()

    def _on_close(self) -> None:
        self._stop_session()
        if self.registration_session is not None:
            self._end_registration()
        self.video_display.stop()
        self.root.destroy()


def main() -> None:
    DashboardApp().run()


if __name__ == "__main__":
    main()
