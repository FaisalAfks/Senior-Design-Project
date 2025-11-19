from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from pipelines.attendance import (
    DEFAULT_ATTENDANCE_LOG,
    DEFAULT_FACEBANK,
    DEFAULT_POWER_LOG,
    DEFAULT_SPOOF_WEIGHTS,
    DEFAULT_WEIGHTS,
)
from utils.cli import parse_main_args
from utils.power import jetson_power_available

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

POWER_AVAILABLE = jetson_power_available()


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


__all__ = [
    "DemoConfig",
    "CLI_DEFAULTS",
    "DEFAULT_CAMERA_WIDTH",
    "DEFAULT_CAMERA_HEIGHT",
    "DEFAULT_SOURCE",
    "DEFAULT_DEVICE",
    "DEFAULT_IDENTITY_THR",
    "DEFAULT_SPOOF_THR",
    "DEFAULT_DETECTOR_THR",
    "POWER_AVAILABLE",
]
