#!/usr/bin/env python3
"""Modernised attendance demo app with a dashboard-style layout."""
from __future__ import annotations

import json
import shutil
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
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
    ("2560 x 1440 (QHD)", (2560, 1440)),
]

FPS_PRESETS: list[tuple[str, Optional[float]]] = [
    ("Camera default", None),
    ("24 FPS", 24.0),
    ("30 FPS", 30.0),
    ("60 FPS", 60.0),
]


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
    """Lightweight OpenCV -> Tkinter bridge."""

    def __init__(self, target_label: ttk.Label) -> None:
        self.label = target_label
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
        self.label.configure(image="", text="Feed paused")

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
            self._photo = ImageTk.PhotoImage(image=image)
            self.label.configure(image=self._photo)
        self.label.after(interval_ms, lambda: self._pump(interval_ms))



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
            evaluation_duration=float(getattr(defaults, "evaluation_duration", 1.0)),
            evaluation_mode=str(getattr(defaults, "evaluation_mode", "time")),
            evaluation_frames=int(getattr(defaults, "evaluation_frames", 30)),
            attendance_log=str(self.attendance_log),
            guidance_box_size=int(getattr(defaults, "guidance_box_size", 0)),
            guidance_center_tolerance=float(getattr(defaults, "guidance_center_tolerance", 0.2)),
            guidance_size_tolerance=float(getattr(defaults, "guidance_size_tolerance", 0.2)),
            guidance_rotation_thr=float(getattr(defaults, "guidance_rotation_thr", 10.0)),
            guidance_hold_frames=int(getattr(defaults, "guidance_hold_frames", 10)),
            frame_max_size=str(getattr(defaults, "frame_max_size", "")),
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

    def start(self, callbacks: SessionCallbacks) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
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

    def set_display_scores(self, value: bool) -> None:
        self.config.display_scores = value
        if self.pipeline is not None:
            self.pipeline.show_summary_scores = value


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


class AttendanceLog(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        columns = ("timestamp", "identity", "result", "source")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=10)
        for col, heading in zip(columns, ("Timestamp", "Identity", "Result", "Source")):
            self.tree.heading(col, text=heading)
            self.tree.column(col, anchor="w")
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def add_entry(self, *, timestamp: str, identity: str, accepted: bool, source: str) -> None:
        result = "Accepted" if accepted else "Rejected"
        self.tree.insert("", 0, values=(timestamp, identity, result, source))
        # Limit rows
        children = self.tree.get_children()
        for item in children[200:]:
            self.tree.delete(item)

    def clear(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)



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
        self._load_settings()
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Camera Source").grid(row=0, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.source_var).grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(self, text="Device").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            self,
            textvariable=self.device_var,
            values=("cpu", "cuda", "cuda:0"),
            state="readonly",
        ).grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(self, text="Resolution preset").grid(row=4, column=0, sticky="w")
        ttk.Combobox(
            self,
            textvariable=self.resolution_var,
            values=[label for label, _ in RESOLUTION_PRESETS],
            state="readonly",
        ).grid(row=5, column=0, sticky="ew")

        ttk.Label(self, text="Target FPS").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            self,
            textvariable=self.fps_var,
            values=[label for label, _ in FPS_PRESETS],
            state="readonly",
        ).grid(row=7, column=0, sticky="ew")

        ttk.Label(self, text="Identity threshold").grid(row=8, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self, textvariable=self.identity_thr_var).grid(row=9, column=0, sticky="ew")
        ttk.Label(self, text="Spoof threshold").grid(row=10, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self, textvariable=self.spoof_thr_var).grid(row=11, column=0, sticky="ew")

        ttk.Checkbutton(self, text="Enable anti-spoof", variable=self.enable_spoof_var).grid(row=12, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(self, text="Show identity scores in summary", variable=self.display_scores_var, command=self.on_overlay_change).grid(row=13, column=0, sticky="w")
        ttk.Checkbutton(self, text="Show metrics overlay (FPS/timings)", variable=self.show_metrics_var, command=self.on_overlay_change).grid(row=14, column=0, sticky="w")

        power_frame = ttk.LabelFrame(self, text="Jetson Power Logging")
        power_frame.grid(row=15, column=0, sticky="ew", pady=(8, 0))
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

        ttk.Button(self, text="Save Settings", command=self._handle_save_settings).grid(row=16, column=0, sticky="ew", pady=(12, 0))
        self.columnconfigure(0, weight=1)

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.source_var.set(str(data.get("source", self.source_var.get())))
        self.device_var.set(str(data.get("device", self.device_var.get())))
        preset = data.get("resolution_preset")
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
        self.power_enabled_var.set(bool(data.get("power_enabled", False)) and POWER_AVAILABLE)
        self.power_path_var.set(str(data.get("power_path", self.power_path_var.get())))
        self.power_interval_var.set(str(data.get("power_interval", self.power_interval_var.get())))

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
            "power_enabled": bool(self.power_enabled_var.get()),
            "power_path": self.power_path_var.get(),
            "power_interval": self.power_interval_var.get(),
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
        except ValueError as exc:
            messagebox.showerror("Invalid settings", f"Numeric format error: {exc}", parent=self)
            return None
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
        self.listbox = tk.Listbox(self, height=18)
        self.listbox.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=4)
        self.listbox.configure(yscrollcommand=scrollbar.set)

        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        controls.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(controls, text="Register User", command=self.on_register).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(controls, text="Refresh Facebank", command=self._handle_refresh).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(controls, text="Rename", command=self._rename_selected).grid(row=0, column=2, sticky="ew", padx=(4, 0))
        ttk.Button(controls, text="Delete Selected", command=self._delete_selected).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self.columnconfigure(0, weight=1)
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

    def _selected_names(self) -> List[str]:
        indices = self.listbox.curselection()
        return [self.listbox.get(idx) for idx in indices]

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

    def _handle_refresh(self) -> None:
        self.refresh()
        if callable(self.on_refresh):
            self.on_refresh()


class RegistrationSession:
    """Background camera capture used during dashboard registration mode."""

    MAX_SAMPLES = 8

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
        assert self.capture is not None
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok:
                continue
            self.latest_frame = frame.copy()
            self.submit_frame(frame)

    def capture_sample(self) -> int:
        if self.latest_frame is None:
            raise RuntimeError("Camera not ready yet.")
        if len(self.samples) >= self.MAX_SAMPLES:
            raise RuntimeError("Maximum samples captured.")
        if self.detector is None:
            raise RuntimeError("Face detector unavailable.")
        detections = self.detector.detect(self.latest_frame)
        best = max(detections, key=lambda det: det.score) if detections else None
        if best is None:
            raise RuntimeError("No face detected; align before capturing.")
        self.samples.append(self.latest_frame.copy())
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
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        self.detector = None

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def max_samples(self) -> int:
        return self.MAX_SAMPLES

class DashboardApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Attendance Dashboard")
        self.root.geometry("1250x720")
        self.controller: Optional[AttendanceSessionController] = None

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
        video_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 8))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        video_label = ttk.Label(video_frame, anchor="center")
        video_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.video_display = FrameDisplay(video_label)
        self.video_display.start()

        controls_container = ttk.Frame(self.session_tab)
        controls_container.grid(row=1, column=0, pady=(0, 12))
        controls_container.columnconfigure(0, weight=1)

        self.session_buttons_frame = ttk.Frame(controls_container)
        self.session_buttons_frame.grid(row=0, column=0)
        self.start_button = ttk.Button(
            self.session_buttons_frame,
            text="Start Session",
            command=self._start_session_from_controls,
            width=18,
        )
        self.start_button.grid(row=0, column=0, padx=10)
        self.pause_button = ttk.Button(
            self.session_buttons_frame,
            text="Stop Session",
            command=self._stop_session,
            width=18,
        )
        self.pause_button.grid(row=0, column=1, padx=10)
        self.pause_button.state(["disabled"])
        self.session_buttons_frame.columnconfigure((0, 1), weight=1)

        self.registration_frame = ttk.Frame(controls_container)
        self.registration_frame.grid(row=0, column=0, sticky="ew")
        self.registration_frame.columnconfigure(0, weight=1)
        self.registration_frame.columnconfigure(1, weight=1)
        self.registration_frame.grid_remove()
        self.registration_name_var = tk.StringVar()
        ttk.Label(self.registration_frame, text="Identity name").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(self.registration_frame, textvariable=self.registration_name_var).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        reg_buttons = ttk.Frame(self.registration_frame)
        reg_buttons.grid(row=2, column=0, columnspan=2, pady=(0, 6))
        ttk.Button(reg_buttons, text="Capture Sample", command=self._capture_registration_sample).grid(row=0, column=0, padx=4)
        ttk.Button(reg_buttons, text="Save Samples", command=self._save_registration_samples).grid(row=0, column=1, padx=4)
        ttk.Button(reg_buttons, text="Cancel", command=self._cancel_registration).grid(row=0, column=2, padx=4)
        self.registration_status_var = tk.StringVar(value="Registration idle.")
        ttk.Label(self.registration_frame, textvariable=self.registration_status_var).grid(row=3, column=0, columnspan=2, sticky="w")

        settings_tab = ttk.Frame(notebook)
        settings_tab.columnconfigure(0, weight=1)
        notebook.add(settings_tab, text="Settings")
        self.settings_tab = settings_tab
        self.control_panel = ControlPanel(
            settings_tab,
            on_overlay_change=self._on_overlay_change,
        )
        self.control_panel.pack(fill="both", expand=True, padx=12, pady=12)

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
        log_tab.rowconfigure(0, weight=1)
        notebook.add(log_tab, text="Attendance Log")
        self.log_panel = AttendanceLog(log_tab)
        self.log_panel.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 4))
        log_buttons = ttk.Frame(log_tab)
        log_buttons.grid(row=1, column=0, sticky="e", padx=12, pady=(0, 12))
        ttk.Button(log_buttons, text="Export CSV", command=self._export_log).grid(row=0, column=0, padx=4)
        ttk.Button(log_buttons, text="Clear Log", command=self._clear_log_entries).grid(row=0, column=1, padx=4)

        self._latest_metrics: Dict[str, float] = {}
        self._show_metrics_overlay = self.control_panel.show_metrics_var.get()
        self._display_scores = self.control_panel.display_scores_var.get()
        self._last_entries: list[dict[str, object]] = []
        self._session_active = False
        self._session_identities: set[str] = set()
        self._next_person_event = threading.Event()
        self._awaiting_next = False
        self.registration_session: Optional[RegistrationSession] = None
        self._session_button_mode = "start"
        self._set_start_button_mode("start")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self) -> None:
        self.root.mainloop()

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
        self._start_session(config)

    def _start_session(self, config: DemoConfig) -> None:
        self._stop_session()
        self.controller = AttendanceSessionController(config)
        self._show_metrics_overlay = config.show_metrics
        self._display_scores = config.display_scores
        self._session_identities.clear()
        self._last_entries.clear()
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
            self.controller.start(callbacks)
            self.status_panel.set_status("Align face to begin verification.")
        except Exception as exc:
            self._stop_session()
            messagebox.showerror("Session error", str(exc), parent=self.root)

    def _stop_session(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        self._session_active = False
        self._awaiting_next = False
        self._next_person_event.set()
        self.video_display.clear()
        self.status_panel.set_stage("Idle")
        self.status_panel.set_status("Session stopped.")
        self.pause_button.state(["disabled"])
        self._set_start_button_mode("start")
        self._latest_metrics = {}
        self._session_identities.clear()

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
            self.log_panel.add_entry(timestamp=timestamp, identity=identity, accepted=accepted, source=str(source))
            self._last_entries.insert(
                0,
                {
                    "timestamp": timestamp,
                    "identity": identity,
                    "accepted": accepted,
                    "source": str(source),
                },
            )
            if len(self._last_entries) > 500:
                self._last_entries.pop()
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
        self.registration_status_var.set(f"Captured {count} / {session.max_samples} samples.")

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
        if not self._last_entries:
            messagebox.showinfo("Export attendance", "No entries to export.", parent=self.root)
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = DASHBOARD_EXPORT_DIR
        export_dir.mkdir(parents=True, exist_ok=True)
        source_value = self.control_panel.source_var.get().strip()
        if not source_value:
            source_slug = "camera"
        else:
            resolved = _resolve_source(source_value)
            if isinstance(resolved, int):
                source_slug = f"camera_{resolved}"
            else:
                source_slug = Path(str(resolved)).stem or "source"
        safe_slug = _sanitize_identity_name(source_slug)
        export_path = export_dir / f"dashboard_run_{timestamp}_{safe_slug}.csv"
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with export_path.open("w", encoding="utf-8") as handle:
                handle.write("timestamp,identity,result,source\n")
                for entry in reversed(self._last_entries):
                    result = "Accepted" if entry["accepted"] else "Rejected"
                    line = f"{entry['timestamp']},{entry['identity']},{result},{entry['source']}\n"
                    handle.write(line)
            messagebox.showinfo("Export attendance", f"CSV written to {export_path}", parent=self.root)
        except Exception as exc:
            messagebox.showerror("Export failed", f"Unable to export log: {exc}", parent=self.root)

    def _refresh_facebank(self) -> None:
        if getattr(self, "facebank_panel", None) is not None:
            self.facebank_panel.refresh()

    def _on_facebank_refresh(self) -> None:
        config = self.control_panel.build_demo_config()
        if config is None:
            self.notebook.select(self.settings_tab)
            return
        self.status_panel.set_status("Refreshing facebank...")
        self._rebuild_facebank_async(config)

    def _rebuild_facebank_async(self, config: DemoConfig) -> None:
        def worker() -> None:
            try:
                MobileFaceNetService(
                    weights_path=config.weights_path,
                    facebank_dir=config.facebank_dir,
                    recognition_threshold=config.identity_threshold,
                    refresh_facebank=True,
                )
                self.root.after(0, lambda: self._facebank_refresh_complete("Facebank refreshed."))
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

    def _clear_log_entries(self) -> None:
        self._last_entries.clear()
        self.log_panel.clear()
        self.status_panel.set_status("Attendance log cleared.")

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
