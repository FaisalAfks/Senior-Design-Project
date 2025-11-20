"""Session controller utilities for guided verification cycles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from BlazeFace import BlazeFaceService
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService

from .guidance import GuidanceBox, run_guidance_phase
from .verification import FaceObservation, run_verification_phase

_WINDOW_READY: set[str] = set()


@dataclass
class SessionCycle:
    observations: List[FaceObservation]
    last_frame: Optional[np.ndarray]
    duration: float
    metrics: Dict[str, float]


class SessionRunner:
    """Manage alternating guidance/verification cycles for a capture device."""

    def __init__(
        self,
        capture: cv2.VideoCapture,
        detector: BlazeFaceService,
        recogniser: MobileFaceNetService,
        spoof_service: Optional[DeePixBiSService],
        args,
        *,
        window_name: str,
        window_limits: Tuple[int, int],
        guidance_min_side: int,
        guidance_box_scale: float,
        frame_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        guidance_display_callback: Optional[Callable[[np.ndarray], None]] = None,
        verification_display_callback: Optional[Callable[[np.ndarray], None]] = None,
        poll_cancel_callback: Optional[Callable[[], bool]] = None,
        wait_for_next_callback: Optional[Callable[[], bool]] = None,
        blocked_identity_checker: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.capture = capture
        self.detector = detector
        self.recogniser = recogniser
        self.spoof_service = spoof_service
        self.args = args
        self.window_name = window_name
        self.window_limits = window_limits
        self.guidance_min_side = guidance_min_side
        self.guidance_box_scale = guidance_box_scale
        self.window_adjusted = False
        self.frame_transform = frame_transform
        self._guidance_shape: Optional[Tuple[int, int]] = None
        self._guidance_display_callback = guidance_display_callback
        self._verification_display_callback = verification_display_callback
        self._poll_cancel_callback = poll_cancel_callback
        self._wait_for_next_callback = wait_for_next_callback
        self._guidance_box: Optional[GuidanceBox] = None
        self._guidance_padding = float(getattr(args, "guidance_crop_padding", 0.2))
        self._blocked_identity_checker = blocked_identity_checker

    def run_cycle(
        self,
        *,
        require_guidance: bool,
        on_activity_change: Optional[Callable[[str], None]] = None,
    ) -> Optional[SessionCycle]:
        if require_guidance:
            self._guidance_box = None
            if on_activity_change is not None:
                on_activity_change("guidance")
            proceed, guidance_box = run_guidance_phase(
                self.capture,
                self.detector,
                self.args,
                min_side=self.guidance_min_side,
                box_scale=self.guidance_box_scale,
                frame_transform=self.frame_transform,
                display_callback=self._display_guidance_frame,
                poll_cancel=self._poll_guidance_cancel,
            )
            if not proceed:
                return None
            self._guidance_box = guidance_box
        else:
            if not self.window_adjusted:
                adjust_window_to_capture(self.capture, self.window_name, self.window_limits)
                self.window_adjusted = True
        if on_activity_change is not None:
            on_activity_change("verification")

        observations, last_frame, duration, metrics = run_verification_phase(
            self.capture,
            self.detector,
            self.recogniser,
            self.spoof_service,
            self.args.spoof_thr,
            self.window_name,
            self.args.evaluation_duration,
            mode=self.args.evaluation_mode,
            frame_limit=self.args.evaluation_frames,
            frame_transform=self.frame_transform,
            display_callback=self._display_verification_frame,
            poll_cancel=self._poll_guidance_cancel,
            collect_timings=True,
            guidance_box=self._guidance_box,
            guidance_padding=self._guidance_padding,
            blocked_identity_checker=self._blocked_identity_checker,
        )
        return SessionCycle(observations=observations, last_frame=last_frame, duration=duration, metrics=metrics)

    def wait_for_next_person(self) -> bool:
        """Return True to continue, False to exit."""
        if self._wait_for_next_callback is not None:
            return bool(self._wait_for_next_callback())
        while True:
            if self._window_closed():
                return False
            key = cv2.waitKey(0) & 0xFF
            if key == ord(" "):
                return True
            if key in (27, ord("q")):
                return False

    def _display_guidance_frame(self, frame: np.ndarray) -> None:
        """Render a guidance frame inside the shared OpenCV window."""
        if self._guidance_display_callback is not None:
            self._guidance_display_callback(frame.copy())
            return
        height, width = frame.shape[:2]
        shape = (height, width)
        if self._guidance_shape != shape:
            target_w = min(width, self.window_limits[0])
            target_h = min(height, self.window_limits[1])
            cv2.resizeWindow(self.window_name, target_w, target_h)
            self.window_adjusted = True
            self._guidance_shape = shape
        cv2.imshow(self.window_name, frame)

    def _display_verification_frame(self, frame: np.ndarray) -> None:
        if self._verification_display_callback is not None:
            self._verification_display_callback(frame.copy())
            return
        height, width = frame.shape[:2]
        if not self.window_adjusted:
            adjust_window_to_capture(self.capture, self.window_name, self.window_limits)
            self.window_adjusted = True
        cv2.imshow(self.window_name, frame)

    def _poll_guidance_cancel(self) -> bool:
        """Return True when the operator cancels guidance from the OpenCV window."""
        if self._poll_cancel_callback is not None:
            return bool(self._poll_cancel_callback())
        if self._window_closed():
            return True
        key = cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))

    def _window_closed(self) -> bool:
        try:
            visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            return self.window_name in _WINDOW_READY
        if visible < 0:
            return False
        _WINDOW_READY.add(self.window_name)
        return visible < 1


def adjust_window_to_capture(
    capture: cv2.VideoCapture,
    window_name: str,
    window_limits: Tuple[int, int],
) -> None:
    width_limit, height_limit = window_limits
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width_limit
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height_limit
    target_w = min(frame_width, width_limit)
    target_h = min(frame_height, height_limit)
    cv2.resizeWindow(window_name, target_w, target_h)
