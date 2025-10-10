"""Camera capture helpers with Jetson-friendly fallbacks."""
from __future__ import annotations

import platform
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import cv2

CameraSource = Union[int, str]


@dataclass(frozen=True)
class CameraConfig:
    """Encapsulate preferred capture defaults for Jetson CSI devices."""

    backend: str = "auto"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    flip_method: int = 0
    sensor_mode: Optional[int] = 4  # IMX219 mode supporting 1280x720
    gstreamer_pipeline: Optional[str] = None


DEFAULT_CAMERA_CONFIG = CameraConfig()


def detect_jetson() -> bool:
    """Return True when running on an NVIDIA Jetson platform."""
    return platform.system() == "Linux" and Path("/etc/nv_tegra_release").exists()


def _fps_fraction(fps: float) -> Tuple[int, int]:
    """Convert an FPS value to a rational pair suitable for GStreamer."""
    if fps <= 0:
        return 30, 1
    numerator = max(1, int(round(fps * 1000)))
    denominator = 1000
    while denominator > 1 and numerator % 2 == 0 and denominator % 2 == 0:
        numerator //= 2
        denominator //= 2
    return numerator, denominator


def build_nvargus_pipeline(
    sensor_id: int,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    flip_method: int = 0,
    sensor_mode: Optional[int] = None,
) -> str:
    """Create a GStreamer pipeline string for Jetson CSI cameras."""
    width = width or 1280
    height = height or 720
    fps_num, fps_den = _fps_fraction(fps)
    mode_arg = f" sensor-mode={sensor_mode}" if sensor_mode is not None else ""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id}{mode_arg} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, "
        f"framerate={fps_num}/{fps_den} ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1"
    )


def _apply_capture_settings(capture: cv2.VideoCapture, width: int, height: int, fps: float) -> None:
    if capture is None:
        return
    if width and width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height and height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps and fps > 0:
        capture.set(cv2.CAP_PROP_FPS, float(fps))


def open_capture(
    source: CameraSource,
    *,
    backend: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    flip_method: Optional[int] = None,
    gstreamer_pipeline: Optional[str] = None,
    sensor_mode: Optional[int] = None,
    config: Optional[CameraConfig] = None,
) -> cv2.VideoCapture:
    """Open a cv2.VideoCapture with sensible defaults for Jetson CSI cameras."""
    if not isinstance(source, int):
        return cv2.VideoCapture(source)

    cfg = config or DEFAULT_CAMERA_CONFIG
    backend = cfg.backend if backend is None else backend
    width = cfg.width if width is None else width
    height = cfg.height if height is None else height
    fps = cfg.fps if fps is None else fps
    flip_method = cfg.flip_method if flip_method is None else flip_method
    sensor_mode = cfg.sensor_mode if sensor_mode is None else sensor_mode
    pipeline_override = gstreamer_pipeline or cfg.gstreamer_pipeline

    backend_choice = (backend or "auto").lower()
    if backend_choice not in {"auto", "opencv", "gstreamer"}:
        backend_choice = "auto"

    prefer_directshow = sys.platform.startswith("win")

    attempts: list[Tuple[str, Callable[[], cv2.VideoCapture]]] = []
    if backend_choice == "gstreamer":
        attempts.append(
            (
                "gstreamer",
                lambda: cv2.VideoCapture(
                    pipeline_override
                    or build_nvargus_pipeline(
                        source,
                        width=width,
                        height=height,
                        fps=fps,
                        flip_method=flip_method,
                        sensor_mode=sensor_mode,
                    ),
                    cv2.CAP_GSTREAMER,
                ),
            )
        )
    elif backend_choice == "opencv":
        if prefer_directshow:
            attempts.append(("directshow", lambda: cv2.VideoCapture(source, cv2.CAP_DSHOW)))
        attempts.append(("opencv", lambda: cv2.VideoCapture(source)))
    else:  # auto
        if detect_jetson():
            attempts.append(
                (
                    "gstreamer",
                    lambda: cv2.VideoCapture(
                        pipeline_override
                        or build_nvargus_pipeline(
                            source,
                            width=width,
                            height=height,
                            fps=fps,
                            flip_method=flip_method,
                            sensor_mode=sensor_mode,
                        ),
                        cv2.CAP_GSTREAMER,
                    ),
                )
            )
            attempts.append(("opencv", lambda: cv2.VideoCapture(source)))
        else:
            if prefer_directshow:
                attempts.append(("directshow", lambda: cv2.VideoCapture(source, cv2.CAP_DSHOW)))
            attempts.append(("opencv", lambda: cv2.VideoCapture(source)))

    for name, factory in attempts:
        capture = factory()
        if capture.isOpened():
            if name in {"opencv", "directshow"}:
                _apply_capture_settings(capture, width, height, fps)
            return capture
        capture.release()

    # Final attempt using default backend so the caller can still inspect .isOpened().
    fallback = cv2.VideoCapture(source)
    if fallback.isOpened():
        _apply_capture_settings(fallback, width, height, fps)
    return fallback
