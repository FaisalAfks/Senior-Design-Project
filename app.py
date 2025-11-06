#!/usr/bin/env python3
"""Tkinter front-end for the Senior Design attendance pipeline."""
from __future__ import annotations

import json
import math
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple
import traceback

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk

from BlazeFace import BlazeFaceService
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService
from main import (
    BLAZEFACE_MIN_FACE,
    DEFAULT_ATTENDANCE_LOG,
    DEFAULT_FACEBANK,
    DEFAULT_POWER_LOG,
    DEFAULT_SPOOF_WEIGHTS,
    DEFAULT_WEIGHTS,
    DEEPIX_TARGET_SIZE,
    MOBILEFACENET_INPUT,
)
from utils.camera import open_video_source
from utils.device import select_device
from utils.guidance import run_guidance_phase
from utils.logging import append_attendance_log
from utils.power import JetsonPowerLogger, jetson_power_available
from utils.verification import (
    FaceObservation,
    aggregate_observations,
    compose_final_display,
    evaluate_frame_with_timing,
    run_verification_phase,
)


FRAME_INTERVAL_MS = 30
REGISTRATION_SAMPLE_COUNT = 8
LOG_THROTTLE_SECONDS = 120.0
GUIDANCE_MIN_SIDE = max(BLAZEFACE_MIN_FACE, MOBILEFACENET_INPUT, DEEPIX_TARGET_SIZE)
GUIDANCE_BOX_SCALE = 0.5
GUIDANCE_HOLD_FRAMES = 15

SETTINGS_PATH = Path("app_settings.json")

RESOLUTION_PRESETS: List[tuple[str, tuple[int, int]]] = [
    ("Default (Camera)", (0, 0)),
    ("640 x 480", (640, 480)),
    ("800 x 600", (800, 600)),
    ("1280 x 720", (1280, 720)),
    ("1920 x 1080", (1920, 1080)),
]


def _resolve_source(spec: str) -> int | str:
    text = spec.strip()
    if not text:
        return 0
    return int(text) if text.isdigit() else text


def _safe_float(value: str) -> Optional[float]:
    text = value.strip()
    if not text:
        return None
    return float(text)


def _sanitize_identity_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name.strip())
    return safe or "person"


def _next_facebank_index(face_dir: Path) -> int:
    prefix = "facebank_"
    max_index = 0
    for image_path in sorted(face_dir.glob(f"{prefix}*.png")) + sorted(face_dir.glob(f"{prefix}*.jpg")):
        suffix = image_path.stem[len(prefix):]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))


@dataclass
class AppConfig:
    source: int | str
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    device_name: str
    identity_threshold: float
    spoof_threshold: float
    weights_path: Path = DEFAULT_WEIGHTS
    facebank_dir: Path = DEFAULT_FACEBANK
    spoof_weights_path: Path = DEFAULT_SPOOF_WEIGHTS
    attendance_log: Path = DEFAULT_ATTENDANCE_LOG
    enable_power_logging: bool = False
    power_log_path: Path = DEFAULT_POWER_LOG
    power_interval: float = 1.0
    guidance_box_size: int = 0
    guidance_hold_frames: int = GUIDANCE_HOLD_FRAMES

    def resolve_device(self):
        return select_device(self.device_name)


class _FrameDisplayMixin:
    """Mixin that renders OpenCV frames inside a Tkinter Label."""

    def _init_display_loop(self) -> None:
        self._display_lock = threading.Lock()
        self._display_frame: Optional[np.ndarray] = None
        self._display_active = False
        self._frame_shape: Optional[Tuple[int, int]] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None

    def _start_display_loop(self) -> None:
        if getattr(self, "_display_active", False):
            return
        self._display_active = True
        self.after(FRAME_INTERVAL_MS, self._drain_display_queue)

    def _stop_display_loop(self) -> None:
        self._display_active = False

    def _queue_display_frame(self, frame: np.ndarray) -> None:
        if getattr(self, "_closed", False):
            return
        with self._display_lock:
            self._display_frame = frame.copy()

    def _drain_display_queue(self) -> None:
        if not getattr(self, "_display_active", False) or getattr(self, "_closed", False):
            return
        frame = None
        with self._display_lock:
            if self._display_frame is not None:
                frame = self._display_frame
                self._display_frame = None
        if frame is not None:
            self._render_frame(frame)
        self.after(FRAME_INTERVAL_MS, self._drain_display_queue)

    def _render_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        height, width = frame.shape[:2]
        if self._frame_shape != (height, width):
            self._frame_shape = (height, width)
            if hasattr(self, "_adjust_window_size"):
                try:
                    self._adjust_window_size(width, height)
                except Exception:
                    pass
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo_image = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo_image)


class AttendanceApp:
    """Main Tkinter window hosting configuration, registration, and attendance tabs."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Attendance Demo")
        self.root.geometry("1000x720")

        self.name_var = tk.StringVar()
        self.source_var = tk.StringVar(value="0")
        self.fps_var = tk.StringVar(value="30")
        self.device_var = tk.StringVar(value="cpu")
        self.resolution_preset_var = tk.StringVar(value=RESOLUTION_PRESETS[0][0])
        self.identity_thr_var = tk.StringVar(value="0.86")
        self.spoof_thr_var = tk.StringVar(value="0.90")
        self.status_var = tk.StringVar(value="Ready.")
        self.power_enabled_var = tk.BooleanVar(value=jetson_power_available())
        self.power_path_var = tk.StringVar(value=str(DEFAULT_POWER_LOG))
        self.power_interval_var = tk.StringVar(value="1.0")
        self.guidance_box_var = tk.StringVar(value="0")
        self.guidance_hold_var = tk.StringVar(value=str(GUIDANCE_HOLD_FRAMES))

        self.attendance_list: Optional[tk.Listbox] = None
        self.facebank_list: Optional[tk.Listbox] = None
        self.notebook: Optional[ttk.Notebook] = None
        self.settings_tab: Optional[ttk.Frame] = None
        self._latest_session_log: Optional[Path] = None

        self._load_settings()
        self._build_layout()
        self.refresh_facebank()

    # ------------------------------------------------------------------ UI helpers
    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        self.source_var.set(str(data.get("source", self.source_var.get())))
        self.fps_var.set(str(data.get("fps", self.fps_var.get())))
        self.device_var.set(str(data.get("device", self.device_var.get())))
        preset = data.get("resolution_preset")
        if preset in [label for label, _ in RESOLUTION_PRESETS]:
            self.resolution_preset_var.set(preset)
        self.identity_thr_var.set(str(data.get("identity_threshold", self.identity_thr_var.get())))
        self.spoof_thr_var.set(str(data.get("spoof_threshold", self.spoof_thr_var.get())))
        self.power_enabled_var.set(bool(data.get("power_enabled", self.power_enabled_var.get())))
        self.power_path_var.set(str(data.get("power_path", self.power_path_var.get())))
        self.power_interval_var.set(str(data.get("power_interval", self.power_interval_var.get())))
        self.guidance_box_var.set(str(data.get("guidance_box_size", self.guidance_box_var.get())))
        self.guidance_hold_var.set(str(data.get("guidance_hold_frames", self.guidance_hold_var.get())))

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.notebook = notebook

        attendance_tab = ttk.Frame(notebook)
        registration_tab = ttk.Frame(notebook)
        settings_tab = ttk.Frame(notebook)

        notebook.add(attendance_tab, text="Attendance")
        notebook.add(registration_tab, text="Registration")
        notebook.add(settings_tab, text="Settings")
        self.settings_tab = settings_tab

        self._build_attendance_tab(attendance_tab)
        self._build_registration_tab(registration_tab)
        self._build_settings_tab(settings_tab)

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status_bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))

        self._update_resolution_label()

    def _build_attendance_tab(self, tab: ttk.Frame) -> None:
        ttk.Label(tab, text="Manage the live attendance session and review recent entries.", anchor="w").grid(
            row=0, column=0, sticky="ew", padx=12, pady=(12, 8)
        )

        tab.columnconfigure(0, weight=1)
        attendance_list = tk.Listbox(tab, height=16)
        attendance_list.grid(row=1, column=0, sticky="nsew", padx=12)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=attendance_list.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=12)
        attendance_list.configure(yscrollcommand=scrollbar.set)
        self.attendance_list = attendance_list

        controls = ttk.Frame(tab)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 12))
        controls.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(controls, text="Start Attendance Session", command=self.start_attendance).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(controls, text="Clear List", command=lambda: attendance_list.delete(0, tk.END)).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(controls, text="Export Excel", command=self.export_attendance_excel).grid(
            row=0, column=2, sticky="ew", padx=(6, 0)
        )

    def _build_registration_tab(self, tab: ttk.Frame) -> None:
        ttk.Label(tab, text="Collect face samples and manage the facebank.", anchor="w").grid(
            row=0, column=0, sticky="ew", padx=12, pady=(12, 8)
        )

        tab.columnconfigure(0, weight=1)
        facebank_list = tk.Listbox(tab, height=16)
        facebank_list.grid(row=1, column=0, sticky="nsew", padx=12)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=facebank_list.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=12)
        facebank_list.configure(yscrollcommand=scrollbar.set)
        self.facebank_list = facebank_list

        controls = ttk.Frame(tab)
        controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 12))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Identity name").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.name_var).grid(row=0, column=1, sticky="ew", padx=(4, 12))
        ttk.Button(controls, text="Register Identity", command=self.start_registration).grid(row=0, column=2, sticky="ew")
        ttk.Button(controls, text="Refresh Facebank", command=self.refresh_facebank).grid(row=0, column=3, sticky="ew", padx=(8, 0))

    def _build_settings_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)

        camera_frame = ttk.LabelFrame(tab, text="Camera")
        camera_frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        camera_frame.columnconfigure(3, weight=1)

        ttk.Label(camera_frame, text="Source").grid(row=0, column=0, sticky="w")
        ttk.Entry(camera_frame, textvariable=self.source_var).grid(row=0, column=1, sticky="ew", padx=(4, 12))

        ttk.Label(camera_frame, text="Resolution preset").grid(row=0, column=2, sticky="w")
        preset_combo = ttk.Combobox(
            camera_frame,
            textvariable=self.resolution_preset_var,
            values=[label for label, _ in RESOLUTION_PRESETS],
            state="readonly",
        )
        preset_combo.grid(row=0, column=3, sticky="ew")
        preset_combo.bind("<<ComboboxSelected>>", lambda _: self._update_resolution_label())

        ttk.Label(camera_frame, text="FPS").grid(row=1, column=0, sticky="w", pady=(8, 0))
        fps_combo = ttk.Combobox(camera_frame, textvariable=self.fps_var, values=("24", "30", "60"), state="readonly")
        fps_combo.grid(row=1, column=1, sticky="w", padx=(4, 12), pady=(8, 0))

        ttk.Label(camera_frame, text="Device").grid(row=1, column=2, sticky="w", pady=(8, 0))
        device_combo = ttk.Combobox(camera_frame, textvariable=self.device_var, values=("cpu", "cuda", "cuda:0", "mps"))
        device_combo.grid(row=1, column=3, sticky="ew", pady=(8, 0))

        thresholds = ttk.LabelFrame(tab, text="Thresholds")
        thresholds.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        thresholds.columnconfigure((1, 3), weight=1)

        ttk.Label(thresholds, text="Identity threshold").grid(row=0, column=0, sticky="w")
        ttk.Entry(thresholds, textvariable=self.identity_thr_var).grid(row=0, column=1, sticky="ew", padx=(4, 12))
        ttk.Label(thresholds, text="Spoof threshold").grid(row=0, column=2, sticky="w")
        ttk.Entry(thresholds, textvariable=self.spoof_thr_var).grid(row=0, column=3, sticky="ew", padx=(4, 12))

        ttk.Label(thresholds, text="Guidance box size").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(thresholds, textvariable=self.guidance_box_var).grid(row=1, column=1, sticky="ew", padx=(4, 12), pady=(8, 0))
        ttk.Label(thresholds, text="Guidance hold frames").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(thresholds, textvariable=self.guidance_hold_var).grid(row=1, column=3, sticky="ew", padx=(4, 12), pady=(8, 0))

        power_frame = ttk.LabelFrame(tab, text="Power logging")
        power_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        power_frame.columnconfigure(1, weight=1)

        power_available = jetson_power_available()
        self.power_enabled_var.set(self.power_enabled_var.get() and power_available)

        power_check = ttk.Checkbutton(
            power_frame,
            text="Enable Jetson power logging (requires jetson-stats)",
            variable=self.power_enabled_var,
        )
        power_check.grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(power_frame, text="Log path").grid(row=1, column=0, sticky="w", pady=(6, 0))
        power_path_entry = ttk.Entry(power_frame, textvariable=self.power_path_var)
        power_path_entry.grid(row=1, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(power_frame, text="Sample interval (s)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        power_interval_entry = ttk.Entry(power_frame, textvariable=self.power_interval_var)
        power_interval_entry.grid(row=2, column=1, sticky="ew", pady=(6, 8))

        if not power_available:
            power_check.state(["disabled"])
            power_path_entry.configure(state="disabled")
            power_interval_entry.configure(state="disabled")
            self.power_enabled_var.set(False)
            ttk.Label(
                power_frame,
                text="Power logging unavailable on this device.",
                foreground="#aa0000",
            ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Button(tab, text="Save Settings", command=self.save_settings).grid(row=3, column=0, sticky="w", padx=12, pady=(0, 12))

    def _update_resolution_label(self) -> None:
        selected = self.resolution_preset_var.get()
        for label, (width, height) in RESOLUTION_PRESETS:
            if label == selected:
                if width <= 0 or height <= 0:
                    self.status_var.set("Resolution: Camera default")
                else:
                    self.status_var.set(f"Resolution: {width} x {height}")
                break

    def _current_resolution(self) -> tuple[Optional[int], Optional[int]]:
        selected = self.resolution_preset_var.get()
        for label, (width, height) in RESOLUTION_PRESETS:
            if label == selected:
                return (width if width > 0 else None, height if height > 0 else None)
        return None, None

    def build_config(self) -> Optional[AppConfig]:
        try:
            width, height = self._current_resolution()
            fps = _safe_float(self.fps_var.get())
            identity_thr = float(self.identity_thr_var.get())
            spoof_thr = float(self.spoof_thr_var.get())
            power_enabled = bool(self.power_enabled_var.get())
            power_interval = float(self.power_interval_var.get() or "1.0")
            guidance_box_size = int(self.guidance_box_var.get() or "0")
            guidance_hold_frames = max(1, int(self.guidance_hold_var.get() or str(GUIDANCE_HOLD_FRAMES)))
        except ValueError as exc:
            messagebox.showerror("Invalid configuration", f"Unable to parse numeric value: {exc}", parent=self.root)
            return None

        power_path = Path(self.power_path_var.get().strip() or DEFAULT_POWER_LOG)
        if power_enabled and power_path.is_dir():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            power_path = power_path / f"jetson_power_{timestamp}.json"

        return AppConfig(
            source=_resolve_source(self.source_var.get()),
            width=width,
            height=height,
            fps=fps,
            device_name=self.device_var.get().strip() or "cpu",
            identity_threshold=identity_thr,
            spoof_threshold=spoof_thr,
            enable_power_logging=power_enabled and jetson_power_available(),
            power_log_path=power_path,
            power_interval=max(power_interval, 0.1),
            guidance_box_size=max(0, guidance_box_size),
            guidance_hold_frames=guidance_hold_frames,
        )

    def save_settings(self) -> None:
        config = self.build_config()
        if config is None:
            return
        payload = {
            "source": self.source_var.get(),
            "fps": self.fps_var.get(),
            "device": self.device_var.get(),
            "resolution_preset": self.resolution_preset_var.get(),
            "identity_threshold": config.identity_threshold,
            "spoof_threshold": config.spoof_threshold,
            "power_enabled": bool(self.power_enabled_var.get()),
            "power_path": self.power_path_var.get(),
            "power_interval": self.power_interval_var.get(),
            "guidance_box_size": config.guidance_box_size,
            "guidance_hold_frames": config.guidance_hold_frames,
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            messagebox.showinfo("Settings saved", f"Settings stored in {SETTINGS_PATH.resolve()}", parent=self.root)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Unable to save settings: {exc}", parent=self.root)

    def refresh_facebank(self) -> None:
        if self.facebank_list is None:
            return
        names: List[str] = []
        if DEFAULT_FACEBANK.exists():
            for entry in sorted(DEFAULT_FACEBANK.iterdir()):
                if entry.is_dir() and not entry.name.startswith("."):
                    names.append(entry.name)
        self.facebank_list.delete(0, tk.END)
        for name in names:
            self.facebank_list.insert(tk.END, name)
        self.status_var.set(f"Loaded {len(names)} identities from {DEFAULT_FACEBANK}.")

    def record_attendance_entry(self, entry: Dict[str, object]) -> None:
        if self.attendance_list is None:
            return
        timestamp = entry.get('timestamp', '')
        identity = entry.get('identity', 'Unknown')
        identity_score = float(entry.get('identity_score', 0.0))
        spoof_score = float(entry.get('spoof_score', 0.0))
        display = f"{timestamp} - {identity} (identity={identity_score:.2f}, spoof={spoof_score:.2f})"
        self.attendance_list.insert(0, display)
        if self.attendance_list.size() > 200:
            self.attendance_list.delete(tk.END)

    def export_attendance_excel(self) -> None:
        config = self.build_config()
        if config is None:
            return
        log_path = Path(self._latest_session_log) if self._latest_session_log else Path(config.attendance_log)
        if not log_path.exists():
            messagebox.showwarning("No log file", f"Attendance log not found at {log_path}", parent=self.root)
            return
        try:
            df = pd.read_json(log_path, lines=True)
        except ValueError as exc:
            messagebox.showerror("Export failed", f"Unable to parse log: {exc}", parent=self.root)
            return
        if df.empty:
            messagebox.showinfo("Export attendance", "Attendance log is empty.", parent=self.root)
            return
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
            except Exception:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = log_path.with_name(f"{log_path.stem}_{timestamp}.xlsx")
        try:
            df.to_excel(excel_path, index=False)
            messagebox.showinfo("Export attendance", f"Excel report written to {excel_path}", parent=self.root)
        except Exception as exc:
            messagebox.showerror("Export failed", f"Unable to write Excel file: {exc}", parent=self.root)

    def start_registration(self) -> None:
        if self.notebook is not None and self.notebook.index("current") != 1:
            self.notebook.select(1)
        config = self.build_config()
        if config is None:
            return
        identity_name = self.name_var.get().strip()
        if not identity_name:
            messagebox.showwarning("Missing name", "Enter an identity name before registration.", parent=self.root)
            return
        sanitized = _sanitize_identity_name(identity_name)
        face_dir = config.facebank_dir / sanitized
        if face_dir.exists():
            proceed = messagebox.askyesno(
                "Identity exists",
                f"The identity '{sanitized}' already has samples. Append new samples?",
                parent=self.root,
            )
            if not proceed:
                return

        self.name_var.set("")
        self.status_var.set(f"Registering '{sanitized}'...")
        RegisterWindow(self, identity_name=sanitized, config=config)

    def start_attendance(self) -> None:
        if self.notebook is not None and self.notebook.index("current") != 0:
            self.notebook.select(0)
        config = self.build_config()
        if config is None:
            return
        session_log = self._create_session_log_path()
        session_log.parent.mkdir(parents=True, exist_ok=True)
        config.attendance_log = session_log
        self._latest_session_log = session_log
        self.status_var.set(f"Starting attendance session... Logging to {session_log.name}")
        AttendanceWindow(self, config=config)

    def _create_session_log_path(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logs_dir = DEFAULT_ATTENDANCE_LOG.parent
        base = logs_dir / f"attendance_session_{timestamp}.jsonl"
        suffix = 1
        path = base
        while path.exists():
            path = logs_dir / f"attendance_session_{timestamp}_{suffix:02d}.jsonl"
            suffix += 1
        return path

    def on_session_finished(self, log_path: Path) -> None:
        self._latest_session_log = Path(log_path)
        self.status_var.set(f"Session complete. Log saved to {log_path.name}")


# ------------------------------------------------------------------------------ Registration window
class RegisterWindow(_FrameDisplayMixin, tk.Toplevel):
    ALIGN_CENTER_TOL = 0.18
    ALIGN_SIZE_TOL = 0.20
    ALIGN_ROTATION_DEG = 8.0

    def __init__(self, app: AttendanceApp, *, identity_name: str, config: AppConfig) -> None:
        super().__init__(app.root)
        self.app = app
        self.config = config
        self.identity_name = identity_name
        self.device = None
        self.detector: Optional[BlazeFaceService] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.samples: List[np.ndarray] = []
        self.capturing = False
        self._closed = False
        self._last_capture_time = 0.0
        self._init_display_loop()

        self.status_var = tk.StringVar(value="Initialising camera...")
        self.progress_var = tk.StringVar(value=f"0 / {REGISTRATION_SAMPLE_COUNT} samples")

        self.title(f"Register Identity – {identity_name}")
        self.geometry("780x560")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=12, pady=12)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(container, anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(container)
        controls.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        controls.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(controls, text="Start Capture", command=self._start_capture).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(controls, text="Save & Finish", command=self._finish_capture).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Cancel", command=self._on_close).grid(row=0, column=2, sticky="ew", padx=(6, 0))

        ttk.Label(controls, textvariable=self.progress_var).grid(row=1, column=0, columnspan=3, pady=(8, 0))
        ttk.Label(controls, textvariable=self.status_var).grid(row=2, column=0, columnspan=3, pady=(4, 0))

        threading.Thread(target=self._initialize_services, daemon=True).start()

    def _initialize_services(self) -> None:
        try:
            self.device = self.config.resolve_device()
            self.detector = BlazeFaceService(device=self.device)
            self.capture = open_video_source(
                self.config.source,
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
            )
            if not self.capture or not self.capture.isOpened():
                raise RuntimeError("Unable to open camera source.")
        except Exception as exc:  # pragma: no cover - UI failure path
            details = _format_exception(exc)
            self.after(0, lambda: self._show_initialization_error(details))
            return

        self.after(0, self._update_frame)
        self.after(0, lambda: self.status_var.set("Press 'Start Capture' when you are ready."))

    def _show_initialization_error(self, details: str) -> None:
        self.status_var.set("Failed to initialise services. See error dialog.")
        self.app.status_var.set("Registration failed; see error dialog.")
        messagebox.showerror("Registration Error", details, parent=self)
        self.after(0, self._on_close)

    def _adjust_window_size(self, width: int, height: int) -> None:
        self.minsize(width + 80, height + 220)

    def _run_guided_alignment(self, cancel_check: Optional[Callable[[], bool]] = None) -> bool:
        if self.capture is None or self.detector is None:
            return True
        self._start_display_loop()
        args = SimpleNamespace(
            guidance_box_size=self.config.guidance_box_size,
            guidance_center_tolerance=self.ALIGN_CENTER_TOL,
            guidance_size_tolerance=self.ALIGN_SIZE_TOL,
            guidance_rotation_thr=self.ALIGN_ROTATION_DEG,
            guidance_hold_frames=self.config.guidance_hold_frames,
        )
        window_name = "Registration Guidance"
        try:
            return run_guidance_phase(
                self.capture,
                self.detector,
                args,
                window_name,
                allow_resize=True,
                min_side=GUIDANCE_MIN_SIDE,
                box_scale=GUIDANCE_BOX_SCALE,
                window_limits=(960, 720),
                frame_transform=None,
                display_callback=self._queue_display_frame,
                poll_cancel=lambda: self._closed or (cancel_check and cancel_check()),
            )
        finally:
            self._stop_display_loop()
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

    def _start_capture(self) -> None:
        if self.detector is None:
            return
        self.samples.clear()
        self.capturing = True
        self._last_capture_time = 0.0
        self.status_var.set("Capturing samples...")
        self.progress_var.set(f"0 / {REGISTRATION_SAMPLE_COUNT} samples")

    def _finish_capture(self) -> None:
        if not self.samples:
            self.status_var.set("Capture at least one sample before saving.")
            return
        self.capturing = False
        self.status_var.set("Saving captured samples...")
        threading.Thread(target=self._save_samples, daemon=True).start()

    def _save_samples(self) -> None:
        if self.detector is None or self.device is None:
            return
        facebank_dir = self.config.facebank_dir / self.identity_name
        facebank_dir.mkdir(parents=True, exist_ok=True)

        next_index = _next_facebank_index(facebank_dir)
        saved = 0
        for sample in self.samples:
            target = facebank_dir / f"facebank_{next_index:03d}.png"
            if cv2.imwrite(str(target), sample):
                saved += 1
                next_index += 1

        try:
            MobileFaceNetService(
                weights_path=self.config.weights_path,
                facebank_dir=self.config.facebank_dir,
                detector=self.detector.detector,
                device=self.device,
                recognition_threshold=self.config.identity_threshold,
                refresh_facebank=True,
            )
        except Exception as exc:  # pragma: no cover - runtime failure
            err_text = _format_exception(exc)
            self.after(0, lambda: messagebox.showerror("Facebank refresh failed", err_text, parent=self))
            self.after(0, lambda: self.status_var.set("Facebank refresh failed; see error dialog."))
            return

        self.after(0, self.app.refresh_facebank)
        self.after(0, lambda: messagebox.showinfo("Registration complete", f"Saved {saved} samples for {self.identity_name}.", parent=self))
        self.after(0, self._on_close)

    def _update_frame(self) -> None:
        if self._closed:
            return
        if self.capture is None or self.detector is None:
            self.after(FRAME_INTERVAL_MS, self._update_frame)
            return

        ok, frame = self.capture.read()
        if not ok:
            self.status_var.set("Camera frame unavailable.")
            self.after(200, self._update_frame)
            return

        height, width = frame.shape[:2]
        detections = self.detector.detect(frame)
        detection = max(detections, key=lambda det: det.score * max(det.area(), 1.0)) if detections else None
        overlay = frame.copy()
        if detection is not None:
            x1, y1, x2, y2 = detection.bbox
            x1 = max(0, min(width - 1, int(x1)))
            y1 = max(0, min(height - 1, int(y1)))
            x2 = max(0, min(width - 1, int(x2)))
            y2 = max(0, min(height - 1, int(y2)))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if self.capturing and detection is not None:
            now = time.time()
            if now - self._last_capture_time >= 0.25:
                face = self.detector.detector.align_face(frame, detection)
                if face is not None:
                    self.samples.append(face)
                    self._last_capture_time = now
                    count = len(self.samples)
                    self.progress_var.set(f"{count} / {REGISTRATION_SAMPLE_COUNT} samples")
                    if count >= REGISTRATION_SAMPLE_COUNT:
                        self.capturing = False
                        self.status_var.set("Captured enough samples. Saving...")
                        threading.Thread(target=self._save_samples, daemon=True).start()
        elif self.capturing:
            self.status_var.set("Face not detected. Keep your face within view.")

        self._render_frame(overlay)
        self.after(FRAME_INTERVAL_MS, self._update_frame)

    def _on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_display_loop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.destroy()


# ------------------------------------------------------------------------------ Attendance window
class AttendanceWindow(_FrameDisplayMixin, tk.Toplevel):
    def __init__(self, app: AttendanceApp, *, config: AppConfig) -> None:
        super().__init__(app.root)
        self.app = app
        self.config = config
        self.device = None
        self.detector: Optional[BlazeFaceService] = None
        self.recogniser: Optional[MobileFaceNetService] = None
        self.spoof_service: Optional[DeePixBiSService] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.power_logger: Optional[JetsonPowerLogger] = None
        self._closed = False
        self._state: str = "initialising"
        self._awaiting_next = False
        self._last_logged: Dict[str, float] = {}
        self._logged_identities: set[str] = set()

        self._init_display_loop()

        self.status_var = tk.StringVar(value="Preparing services...")

        self.title("Attendance Session")
        self.geometry("1000x760")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=12, pady=12)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(container, anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.bind("<space>", self._on_space_key)
        self.focus_set()

        status_frame = ttk.Frame(container)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(status_frame, text="Stop Session", command=self._on_close).grid(row=0, column=1, sticky="e", padx=(12, 0))

        threading.Thread(target=self._initialize_services, daemon=True).start()

    def _initialize_services(self) -> None:
        power_logger: Optional[JetsonPowerLogger] = None
        try:
            self.device = self.config.resolve_device()
            self.detector = BlazeFaceService(device=self.device)
            self.recogniser = MobileFaceNetService(
                weights_path=self.config.weights_path,
                facebank_dir=self.config.facebank_dir,
                detector=self.detector.detector,
                device=self.device,
                recognition_threshold=self.config.identity_threshold,
                refresh_facebank=True,
            )
            if self.config.spoof_weights_path and Path(self.config.spoof_weights_path).exists():
                self.spoof_service = DeePixBiSService(
                    weights_path=self.config.spoof_weights_path,
                    device=self.device,
                    threshold=self.config.spoof_threshold,
                )
            elif self.config.spoof_weights_path:
                self.app.status_var.set(f"DeePixBiS weights not found at {self.config.spoof_weights_path}; liveness disabled.")

            power_logger = JetsonPowerLogger(
                enabled=self.config.enable_power_logging,
                log_path=self.config.power_log_path,
                sample_interval=self.config.power_interval,
                verbose=False,
            )
            power_logger.start()
            if power_logger.enabled:
                power_logger.set_activity("initializing")

            self.capture = open_video_source(
                self.config.source,
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
            )
            if not self.capture or not self.capture.isOpened():
                raise RuntimeError("Unable to open camera source.")

            if power_logger.enabled:
                raw_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                raw_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                resolved_width = raw_width if raw_width > 0 else int(self.config.width or 0)
                resolved_height = raw_height if raw_height > 0 else int(self.config.height or 0)
                fps_value = self.capture.get(cv2.CAP_PROP_FPS)
                actual_fps = float(fps_value) if fps_value and fps_value > 0 else None
                if resolved_width > 0 and resolved_height > 0:
                    power_logger.set_resolution(
                        resolved_width,
                        resolved_height,
                        fps=actual_fps,
                        source=str(self.config.source),
                    )
                requested_meta: Dict[str, int] = {}
                if self.config.width:
                    requested_meta["width"] = int(self.config.width)
                if self.config.height:
                    requested_meta["height"] = int(self.config.height)
                if requested_meta:
                    power_logger.update_metadata(requested_resolution=requested_meta)
                if self.config.fps:
                    try:
                        power_logger.update_metadata(requested_fps=float(self.config.fps))
                    except (TypeError, ValueError):
                        pass

            self.power_logger = power_logger if power_logger.enabled else None
            if self.power_logger is not None:
                self.power_logger.set_activity("ready")

            self.after(0, lambda: self.status_var.set("Align your face to begin verification."))
            self.after(0, self._start_cycle)
        except Exception as exc:  # pragma: no cover - UI failure path
            if power_logger is not None:
                power_logger.stop()
            details = _format_exception(exc)
            self.after(0, lambda: self._show_initialization_error(details))

    def _adjust_window_size(self, width: int, height: int) -> None:
        self.minsize(width + 100, height + 280)

    def _show_initialization_error(self, details: str) -> None:
        self.status_var.set("Failed to initialise services. See error dialog.")
        self.app.status_var.set("Attendance session failed to start; see error dialog.")
        messagebox.showerror("Attendance Session Error", details, parent=self)
        self.after(0, self._on_close)

    def _start_cycle(self) -> None:
        if self._closed or self.capture is None:
            return
        self._awaiting_next = False
        self._state = "guidance"
        self.status_var.set("Align your face inside the guide box.")
        if self.power_logger is not None:
            self.power_logger.set_activity("guidance")
        threading.Thread(target=self._guidance_worker, daemon=True).start()

    def _guidance_worker(self) -> None:
        success = self._run_guided_alignment(cancel_check=lambda: self._closed or self._state != "guidance")
        self.after(0, lambda: self._on_guidance_complete(success))

    def _run_guided_alignment(self, cancel_check: Optional[Callable[[], bool]] = None) -> bool:
        if self.capture is None or self.detector is None:
            return True
        self._start_display_loop()
        args = SimpleNamespace(
            guidance_box_size=self.config.guidance_box_size,
            guidance_center_tolerance=RegisterWindow.ALIGN_CENTER_TOL,
            guidance_size_tolerance=RegisterWindow.ALIGN_SIZE_TOL,
            guidance_rotation_thr=RegisterWindow.ALIGN_ROTATION_DEG,
            guidance_hold_frames=self.config.guidance_hold_frames,
        )
        window_name = "Attendance Guidance"
        try:
            return run_guidance_phase(
                self.capture,
                self.detector,
                args,
                window_name,
                allow_resize=True,
                min_side=GUIDANCE_MIN_SIDE,
                box_scale=GUIDANCE_BOX_SCALE,
                window_limits=(960, 720),
                frame_transform=None,
                display_callback=self._queue_display_frame,
                poll_cancel=lambda: self._closed or (cancel_check and cancel_check()),
            )
        finally:
            self._stop_display_loop()
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

    def _on_guidance_complete(self, success: bool) -> None:
        if self._closed or self._state != "guidance":
            return
        if not success:
            self.status_var.set("Guidance cancelled. Press 'Stop Session' to exit.")
            if self.power_logger is not None:
                self.power_logger.set_activity("waiting")
            return
        self._start_verification()

    def _start_verification(self) -> None:
        if self._closed or self.capture is None:
            return
        self._state = "verification"
        self.status_var.set("Capturing verification frames...")
        if self.power_logger is not None:
            self.power_logger.set_activity("verification")
        self._start_display_loop()
        threading.Thread(target=self._verification_worker, daemon=True).start()

    def _verification_worker(self) -> None:
        observations, last_frame, duration = run_verification_phase(
            self.capture,
            self.detector,
            self.recogniser,
            self.spoof_service,
            self.config.spoof_threshold,
            window_name="Attendance Verification",
            duration_limit=0.95,
            mode="time",
            frame_limit=45,
            display_callback=self._queue_display_frame,
            poll_cancel=lambda: self._closed or self._state != "verification",
        )
        self.after(0, lambda: self._on_verification_complete(observations, last_frame, duration))

    def _on_verification_complete(
        self,
        observations: List[FaceObservation],
        last_frame: Optional[np.ndarray],
        duration: float,
    ) -> None:
        if self._closed or self._state != "verification":
            return
        self._stop_display_loop()

        if self.power_logger is not None:
            self.power_logger.set_activity("processing")

        if not observations:
            self.status_var.set("No verification captured. Restarting...")
            self.app.status_var.set("No verification captured. Restarting...")
            if self.power_logger is not None:
                self.power_logger.set_activity("ready")
            self.after(800, self._start_cycle)
            return

        summary = aggregate_observations(observations, spoof_threshold=self.config.spoof_threshold)
        summary["capture_duration"] = duration

        if summary.get("is_real") is False:
            self.status_var.set("Liveness check failed. Please try again.")
            self.app.status_var.set("Liveness check failed. Please try again.")
            if self.power_logger is not None:
                self.power_logger.set_activity("ready")
            self.after(1200, self._start_cycle)
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        display_frame = last_frame if last_frame is not None else np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        final_display = compose_final_display(
            display_frame,
            summary,
            show_spoof_score=self.spoof_service is not None,
        )
        self._render_frame(final_display)

        identity = summary.get("identity") or "Unknown"
        status_text = "Access granted" if summary.get("accepted") else "Access denied"
        self.status_var.set(f"{status_text}: {identity}. Press space for next attendee.")
        self.app.status_var.set(f"{status_text} for {identity}.")
        if self.power_logger is not None:
            self.power_logger.set_activity("waiting")

        self._state = "results"
        self._awaiting_next = True
        self._log_summary(summary, timestamp)

    def _log_summary(self, summary: Dict[str, object], timestamp: str) -> None:
        identity = summary.get("identity") or "Unknown"
        avg_identity = float(summary.get("avg_identity_score") or 0.0)
        avg_spoof = float(summary.get("avg_spoof_score") or 0.0)

        log_entry = {
            "timestamp": timestamp,
            "source": str(self.config.source),
            "recognized": summary.get("recognized"),
            "identity": identity,
            "avg_identity_score": avg_identity,
            "avg_spoof_score": avg_spoof,
            "is_real": summary.get("is_real"),
            "accepted": summary.get("accepted"),
            "frames_with_detections": summary.get("frames_with_detections"),
            "capture_duration": summary.get("capture_duration"),
        }
        append_attendance_log(self.config.attendance_log, log_entry)

        if summary.get("accepted"):
            now = time.time()
            last = self._last_logged.get(identity, 0.0)
            if now - last >= LOG_THROTTLE_SECONDS:
                self._last_logged[identity] = now
                self._logged_identities.add(identity)
                ui_entry = {
                    "timestamp": timestamp,
                    "identity": identity,
                    "identity_score": avg_identity,
                    "spoof_score": avg_spoof,
                    "accepted": True,
                    "source": str(self.config.source),
                }
                self.after(0, lambda entry=ui_entry: self.app.record_attendance_entry(entry))

    def _on_space_key(self, _event) -> Optional[str]:
        if self._closed or not self._awaiting_next:
            return None
        self._awaiting_next = False
        if self.power_logger is not None:
            self.power_logger.set_activity("ready")
        self.status_var.set("Starting next attendee...")
        self.app.status_var.set("Ready for next attendee.")
        self._start_cycle()
        return "break"

    def _show_initialization_error(self, details: str) -> None:
        self.status_var.set("Failed to initialise services. See error dialog.")
        self.app.status_var.set("Attendance session failed to start; see error dialog.")
        messagebox.showerror("Attendance Session Error", details, parent=self)
        self.after(0, self._on_close)

    def _on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_display_loop()
        self._state = "closed"
        self._awaiting_next = False
        self.unbind("<space>")
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.power_logger is not None:
            self.power_logger.set_activity("terminating")
            self.power_logger.stop()
            self.power_logger = None
        self.app.on_session_finished(Path(self.config.attendance_log))
        self.destroy()


def main() -> None:
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
