from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import messagebox, ttk

from pipelines.attendance import DEFAULT_POWER_LOG
from dashboard.configuration import (
    CLI_DEFAULTS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_DEVICE,
    DEFAULT_IDENTITY_THR,
    DEFAULT_SOURCE,
    DEFAULT_SPOOF_THR,
    DemoConfig,
    POWER_AVAILABLE,
)
from dashboard.settings import AppSettings, SettingsStore
from dashboard.utils import (
    FPS_PRESETS,
    RESOLUTION_PRESETS,
    _fps_label_for_value,
    _fps_value,
    _resolution_label_for_dims,
    _resolution_value,
    _resolve_source,
)


def _default_error_dialog(title: str, message: str, *, parent: Optional[tk.Misc] = None) -> None:
    messagebox.showerror(title, message, parent=parent)


class ControlPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        settings_store: SettingsStore,
        initial_settings: AppSettings,
        on_overlay_change: Optional[Callable[[], None]] = None,
        on_settings_saved: Optional[Callable[[AppSettings], None]] = None,
        show_error_dialog: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        super().__init__(master)
        self.on_overlay_change = on_overlay_change or (lambda: None)
        self._on_settings_saved = on_settings_saved or (lambda settings: None)
        self._show_error_dialog = show_error_dialog or (lambda title, message, parent=None: _default_error_dialog(title, message, parent=parent))
        self._settings_store = settings_store
        self._current_settings = initial_settings
        self._power_available = POWER_AVAILABLE

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
        self._apply_settings(initial_settings)
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
        ).grid(row=5, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(capture_frame, text="FPS").grid(row=6, column=0, sticky="w")
        ttk.Combobox(
            capture_frame,
            textvariable=self.fps_var,
            values=[label for label, _ in FPS_PRESETS],
            state="readonly",
        ).grid(row=7, column=0, sticky="ew")

        overlay_frame = ttk.LabelFrame(right, text="Overlay")
        overlay_frame.grid(row=0, column=0, sticky="ew")
        overlay_frame.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            overlay_frame,
            text="Show identity scores",
            variable=self.display_scores_var,
            command=self.on_overlay_change,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            overlay_frame,
            text="Show metrics overlay",
            variable=self.show_metrics_var,
            command=self.on_overlay_change,
        ).grid(row=1, column=0, sticky="w")

        thresholds_frame = ttk.LabelFrame(right, text="Thresholds")
        thresholds_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        thresholds_frame.columnconfigure(0, weight=1)
        ttk.Label(thresholds_frame, text="Identity threshold").grid(row=0, column=0, sticky="w")
        ttk.Entry(thresholds_frame, textvariable=self.identity_thr_var).grid(row=1, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(thresholds_frame, text="Spoof threshold").grid(row=2, column=0, sticky="w")
        ttk.Entry(thresholds_frame, textvariable=self.spoof_thr_var).grid(row=3, column=0, sticky="ew", pady=(0, 4))
        ttk.Checkbutton(thresholds_frame, text="Enable spoof detection", variable=self.enable_spoof_var).grid(
            row=4, column=0, sticky="w"
        )

        guidance_frame = ttk.LabelFrame(right, text="Guidance")
        guidance_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
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
        if not self._power_available:
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

    def _apply_settings(self, settings: AppSettings) -> None:
        self._current_settings = settings
        self.source_var.set(settings.source or self.source_var.get())
        device = settings.device or self.device_var.get()
        if device == "cuda:0":
            device = "cuda"
        self.device_var.set(device)
        self.identity_thr_var.set(settings.identity_threshold or self.identity_thr_var.get())
        self.spoof_thr_var.set(settings.spoof_threshold or self.spoof_thr_var.get())
        self.enable_spoof_var.set(bool(settings.enable_spoof))
        self.display_scores_var.set(bool(settings.display_scores))
        self.show_metrics_var.set(bool(settings.show_metrics))
        self._apply_resolution_setting(settings.resolution_preset)
        self._apply_fps_setting(settings.fps)
        self.eval_mode_var.set((settings.evaluation_mode or self.eval_mode_var.get()).strip())
        self.eval_duration_var.set(settings.evaluation_duration or self.eval_duration_var.get())
        self.eval_frames_var.set(settings.evaluation_frames or self.eval_frames_var.get())
        self.power_enabled_var.set(bool(settings.power_enabled) and self._power_available)
        self.power_path_var.set(settings.power_path or self.power_path_var.get())
        self.power_interval_var.set(settings.power_interval or self.power_interval_var.get())
        self.guidance_box_size_var.set(settings.guidance_box_size or self.guidance_box_size_var.get())
        self.guidance_center_tol_var.set(settings.guidance_center_tolerance or self.guidance_center_tol_var.get())
        self.guidance_size_tol_var.set(settings.guidance_size_tolerance or self.guidance_size_tol_var.get())
        self.guidance_padding_var.set(settings.guidance_crop_padding or self.guidance_padding_var.get())

    def _apply_resolution_setting(self, preset: Optional[str]) -> None:
        if not preset:
            return
        resolution_labels = [label for label, _ in RESOLUTION_PRESETS]
        if preset in resolution_labels:
            self.resolution_var.set(preset)
            return
        if "x" in preset.lower():
            try:
                width_str, height_str = preset.lower().split("x", 1)
                width = int(width_str.strip())
                height = int(height_str.strip())
                self.resolution_var.set(_resolution_label_for_dims(width, height))
            except ValueError:
                pass

    def _apply_fps_setting(self, preset: Optional[str]) -> None:
        if not preset:
            return
        labels = [label for label, _ in FPS_PRESETS]
        if preset in labels:
            self.fps_var.set(preset)
            return
        try:
            numeric = float(preset)
            self.fps_var.set(_fps_label_for_value(numeric))
        except (TypeError, ValueError):
            pass

    def _handle_save_settings(self) -> None:
        payload = self._collect_app_settings()
        try:
            self._settings_store.save(payload)
            self._current_settings = payload
            self._on_settings_saved(payload)
            messagebox.showinfo("Settings saved", f"Settings stored in {self._settings_store.path.resolve()}", parent=self)
        except OSError as exc:
            self._show_error_dialog("Save failed", f"Unable to save settings: {exc}", parent=self)

    def _collect_app_settings(self) -> AppSettings:
        base = self._current_settings
        return AppSettings(
            source=self.source_var.get(),
            device=self.device_var.get(),
            resolution_preset=self.resolution_var.get(),
            fps=self.fps_var.get(),
            identity_threshold=self.identity_thr_var.get(),
            spoof_threshold=self.spoof_thr_var.get(),
            enable_spoof=bool(self.enable_spoof_var.get()),
            display_scores=bool(self.display_scores_var.get()),
            show_metrics=bool(self.show_metrics_var.get()),
            evaluation_mode=self.eval_mode_var.get(),
            evaluation_duration=self.eval_duration_var.get(),
            evaluation_frames=self.eval_frames_var.get(),
            power_enabled=bool(self.power_enabled_var.get()),
            power_path=self.power_path_var.get(),
            power_interval=self.power_interval_var.get(),
            guidance_box_size=self.guidance_box_size_var.get(),
            guidance_center_tolerance=self.guidance_center_tol_var.get(),
            guidance_size_tolerance=self.guidance_size_tol_var.get(),
            guidance_crop_padding=self.guidance_padding_var.get(),
            window_geometry=base.window_geometry,
            window_state=base.window_state,
        )

    def update_window_settings(self, *, geometry: Optional[str], state: str) -> None:
        """Keep cached settings aligned with the app's actual window placement."""
        if self._current_settings is None:
            return
        self._current_settings.window_geometry = geometry
        self._current_settings.window_state = state

    def build_demo_config(self) -> Optional[DemoConfig]:
        settings = self._collect_app_settings()
        try:
            identity_thr = float(settings.identity_threshold or "0.9")
            spoof_thr = float(settings.spoof_threshold or "0.9")
            power_interval = float(settings.power_interval or "1.0")
            box_size = int(float(settings.guidance_box_size or "0"))
            center_tol = float(settings.guidance_center_tolerance or "0.3")
            size_tol = float(settings.guidance_size_tolerance or "0.3")
            crop_padding = float(settings.guidance_crop_padding or "0.5")
            eval_duration = float(settings.evaluation_duration or "0.9")
            eval_frames = int(float(settings.evaluation_frames or "30"))
        except ValueError as exc:
            self._show_error_dialog("Invalid settings", f"Numeric format error: {exc}", parent=self)
            return None

        box_size = max(0, box_size)
        center_tol = max(0.0, center_tol)
        size_tol = max(0.0, size_tol)
        crop_padding = max(0.0, crop_padding)
        eval_duration = max(0.1, eval_duration)
        eval_frames = max(1, eval_frames)
        eval_mode = settings.evaluation_mode.strip().lower()
        if eval_mode not in ("time", "frames"):
            eval_mode = "time"
        width, height = _resolution_value(self.resolution_var.get())
        fps_value = _fps_value(self.fps_var.get())
        return DemoConfig(
            source=_resolve_source(settings.source),
            device=settings.device,
            identity_threshold=identity_thr,
            spoof_threshold=spoof_thr,
            enable_spoof=settings.enable_spoof and self.enable_spoof_var.get(),
            display_scores=settings.display_scores and self.display_scores_var.get(),
            show_metrics=settings.show_metrics and self.show_metrics_var.get(),
            width=width,
            height=height,
            fps=fps_value,
            enable_power_logging=settings.power_enabled and self._power_available,
            power_log_path=Path(settings.power_path.strip() or DEFAULT_POWER_LOG),
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
