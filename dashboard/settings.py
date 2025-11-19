from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class AppSettings:
    source: str = "0"
    device: str = "cpu"
    resolution_preset: str = "Camera default"
    fps: str = "Camera default"
    identity_threshold: str = "0.90"
    spoof_threshold: str = "0.90"
    enable_spoof: bool = True
    display_scores: bool = True
    show_metrics: bool = True
    evaluation_mode: str = "time"
    evaluation_duration: str = "1.0"
    evaluation_frames: str = "30"
    power_enabled: bool = False
    power_path: str = ""
    power_interval: str = "1.0"
    guidance_box_size: str = "224"
    guidance_center_tolerance: str = "0.3"
    guidance_size_tolerance: str = "0.3"
    guidance_crop_padding: str = "0.5"
    window_geometry: Optional[str] = None
    window_state: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppSettings":
        defaults = cls()
        values: dict[str, Any] = {}
        for field in fields(cls):
            values[field.name] = payload.get(field.name, getattr(defaults, field.name))
        return cls(**values)


class SettingsStore:
    """Lightweight JSON persistence for dashboard settings."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> AppSettings:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Invalid settings structure")
            return AppSettings.from_dict(payload)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        payload = settings.to_dict()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_name(self.path.name + ".tmp")
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(self.path)
        except OSError:
            pass
