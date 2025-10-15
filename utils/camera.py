"""Runtime helpers for selecting devices and opening video sources."""
from __future__ import annotations

import platform
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.parse import parse_qs

import cv2

CSI_PREFIX = "csi://"
DEFAULT_FPS = 30
DEFAULT_FLIP = 0



SourceType = Union[int, str, Path]

def gstreamer_pipeline(
    capture_width: int,
    capture_height: int,
    display_width: Optional[int] = None,
    display_height: Optional[int] = None,
    framerate: int = DEFAULT_FPS,
    flip_method: int = DEFAULT_FLIP,
) -> str:
    """Compose a GStreamer pipeline string for Jetson CSI cameras."""
    display_width = display_width or capture_width
    display_height = display_height or capture_height
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
    )


def open_video_source(
    source: SourceType,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
) -> cv2.VideoCapture:
    """Open a video source using the caller-provided camera arguments."""
    width = int(width) if width and width > 0 else None
    height = int(height) if height and height > 0 else None
    fps = float(fps) if fps and fps > 0 else None

    if _is_linux():
        capture = _try_open_csi_source(source, width, height, fps)
        if capture is not None:
            return capture

    capture_source: SourceType = str(source) if isinstance(source, Path) else source
    capture = cv2.VideoCapture(capture_source)
    if capture.isOpened():
        _apply_capture_properties(capture, width, height, fps)
    return capture


def _is_linux() -> bool:
    return platform.system().lower() == "linux"


def _try_open_csi_source(
    source: SourceType,
    width: Optional[int],
    height: Optional[int],
    fps: Optional[float],
) -> Optional[cv2.VideoCapture]:
    if isinstance(source, str):
        if source.startswith(CSI_PREFIX):
            sensor_id, params = _parse_csi_source(source)
            return _open_csi_capture(sensor_id=sensor_id, params=params, width=width, height=height, fps=fps)
        if source.isdigit():
            return _open_csi_capture(sensor_id=int(source), params={}, width=width, height=height, fps=fps)
    if isinstance(source, int):
        return _open_csi_capture(sensor_id=source, params={}, width=width, height=height, fps=fps)
    return None


def _parse_csi_source(source: str) -> Tuple[Optional[int], Dict[str, str]]:
    spec = source[len(CSI_PREFIX) :]
    if "?" in spec:
        sensor_part, query_part = spec.split("?", 1)
        query = parse_qs(query_part, keep_blank_values=True)
        params = {key: values[-1] for key, values in query.items()}
    else:
        sensor_part = spec
        params = {}

    sensor_id = None
    sensor_part = sensor_part.strip("/")
    if sensor_part:
        try:
            sensor_id = int(sensor_part)
        except ValueError:
            params.setdefault("sensor-id", sensor_part)
    return sensor_id, params


def _int_param(params: Dict[str, str], *names: str, default: int) -> int:
    for name in names:
        if name in params and params[name]:
            try:
                return int(params[name])
            except ValueError:
                continue
    return default


def _apply_sensor_properties(pipeline: str, sensor_id: Optional[int], params: Dict[str, str]) -> str:
    modifiers = []
    if sensor_id is not None:
        modifiers.append(f"sensor-id={sensor_id}")
    sensor_mode = params.get("sensor_mode") or params.get("sensor-mode")
    if sensor_mode:
        modifiers.append(f"sensor-mode={sensor_mode}")
    exposure_timerange = params.get("exposuretimerange")
    if exposure_timerange:
        modifiers.append(f"exposuretimerange={exposure_timerange}")
    gainrange = params.get("gainrange")
    if gainrange:
        modifiers.append(f"gainrange={gainrange}")
    if modifiers:
        return pipeline.replace("nvarguscamerasrc", f"nvarguscamerasrc {' '.join(modifiers)}", 1)
    return pipeline


def _open_csi_capture(
    *,
    sensor_id: Optional[int],
    params: Dict[str, str],
    width: Optional[int],
    height: Optional[int],
    fps: Optional[float],
) -> Optional[cv2.VideoCapture]:
    if width is None or height is None:
        return None

    capture_width = _int_param(params, "capture_width", "width", default=width)
    capture_height = _int_param(params, "capture_height", "height", default=height)
    display_width = _int_param(params, "display_width", default=width)
    display_height = _int_param(params, "display_height", default=height)
    framerate = _int_param(params, "framerate", "fps", default=int(fps) if fps else DEFAULT_FPS)
    flip_method = _int_param(params, "flip_method", "flip", default=DEFAULT_FLIP)

    pipeline = gstreamer_pipeline(
        capture_width=capture_width,
        capture_height=capture_height,
        display_width=display_width,
        display_height=display_height,
        framerate=framerate,
        flip_method=flip_method,
    )
    pipeline = _apply_sensor_properties(pipeline, sensor_id, params)
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if capture.isOpened():
        return capture
    capture.release()
    return None


def _apply_capture_properties(
    capture: cv2.VideoCapture,
    width: Optional[int],
    height: Optional[int],
    fps: Optional[float],
) -> None:
    if width and width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height and height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps and fps > 0:
        capture.set(cv2.CAP_PROP_FPS, float(fps))
