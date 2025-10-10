"""Runtime helpers for selecting devices and opening video sources."""
from __future__ import annotations

import platform
from typing import Optional, Tuple

import cv2


def gstreamer_pipeline(
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 960,
    display_height: int = 540,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
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

def open_video_source(source, *, frame_size: Optional[Tuple[int, int]] = None, fps: Optional[float] = None,) -> cv2.VideoCapture:
    """Open a cv2.VideoCapture for USB, file, or Jetson CSI camera sources."""
    if platform.system().lower() == "linux":
        return cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    else:
        capture = cv2.VideoCapture(source)
    
    if not capture.isOpened():
        return capture

    if frame_size is not None:
        width, height = frame_size
        if width: capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height: capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps:
        capture.set(cv2.CAP_PROP_FPS, float(fps))
    return capture
