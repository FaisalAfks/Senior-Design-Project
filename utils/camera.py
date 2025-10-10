"""Runtime helpers for selecting devices and opening video sources."""
from __future__ import annotations

import platform
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs

import cv2

CSI_PREFIX = "csi://"
DEFAULT_CAPTURE_WIDTH = 1920
DEFAULT_CAPTURE_HEIGHT = 1080
DEFAULT_DISPLAY_WIDTH = 960
DEFAULT_DISPLAY_HEIGHT = 540
DEFAULT_FPS = 30
DEFAULT_FLIP = 0


def gstreamer_pipeline(
    capture_width: int = DEFAULT_CAPTURE_WIDTH,
    capture_height: int = DEFAULT_CAPTURE_HEIGHT,
    display_width: int = DEFAULT_DISPLAY_WIDTH,
    display_height: int = DEFAULT_DISPLAY_HEIGHT,
    framerate: int = DEFAULT_FPS,
    flip_method: int = DEFAULT_FLIP,
) -> str:
    """Compose a GStreamer pipeline string for Jetson CSI cameras."""
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def _is_linux() -> bool:
    return platform.system().lower() == "linux"


def _parse_csi_source(source: str) -> Tuple[Optional[int], Dict[str, str]]:
    """Parse a csi:// URI into optional sensor id and query parameters."""
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
    frame_size: Optional[Tuple[int, int]],
    fps: Optional[float],
) -> cv2.VideoCapture:
    capture_width = _int_param(
        params,
        "capture_width",
        "width",
        default=frame_size[0] if frame_size else DEFAULT_CAPTURE_WIDTH,
    )
    capture_height = _int_param(
        params,
        "capture_height",
        "height",
        default=frame_size[1] if frame_size else DEFAULT_CAPTURE_HEIGHT,
    )
    display_width = _int_param(
        params,
        "display_width",
        default=frame_size[0] if frame_size else DEFAULT_DISPLAY_WIDTH,
    )
    display_height = _int_param(
        params,
        "display_height",
        default=frame_size[1] if frame_size else DEFAULT_DISPLAY_HEIGHT,
    )
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


def open_video_source(
    source,
    *,
    frame_size: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None,
) -> cv2.VideoCapture:
    """Open a video source with support for Jetson CSI cameras."""
    if _is_linux():
        if isinstance(source, str) and source.startswith(CSI_PREFIX):
            sensor_id, params = _parse_csi_source(source)
            capture = _open_csi_capture(sensor_id=sensor_id, params=params, frame_size=frame_size, fps=fps)
            if capture is not None:
                return capture
        else:
            try:
                sensor_index = int(source) if isinstance(source, str) else int(source)
            except (TypeError, ValueError):
                sensor_index = None
            if sensor_index is not None:
                capture = _open_csi_capture(sensor_id=sensor_index, params={}, frame_size=frame_size, fps=fps)
                if capture is not None:
                    return capture

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return capture

    if frame_size is not None:
        width, height = frame_size
        if width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps:
        capture.set(cv2.CAP_PROP_FPS, float(fps))
    return capture
