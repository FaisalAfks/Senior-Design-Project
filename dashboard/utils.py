from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.overlay import PanelStyle


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


def _resolve_source(spec: str) -> int | str:
    text = (spec or "").strip()
    if not text:
        return 0
    return int(text) if text.isdigit() else text


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


def _sanitize_identity_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name.strip())
    return safe or "person"


__all__ = [
    "_resolve_source",
    "_format_display_timestamp",
    "_next_facebank_index",
    "_resolution_label_for_dims",
    "_resolution_value",
    "_fps_label_for_value",
    "_fps_value",
    "_sanitize_identity_name",
    "RESOLUTION_PRESETS",
    "FPS_PRESETS",
    "LIVE_FEED_SIZE",
    "UI_PAD",
    "METRICS_PANEL_STYLE",
]
