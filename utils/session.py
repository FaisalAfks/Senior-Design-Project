"""Session controller utilities for guided verification cycles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from BlazeFace import BlazeFaceService
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService

from .guidance import run_guidance_phase
from .verification import FaceObservation, run_verification_phase


@dataclass
class SessionCycle:
    observations: List[FaceObservation]
    last_frame: Optional[np.ndarray]
    duration: float


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

    def run_cycle(
        self,
        *,
        require_guidance: bool,
        on_activity_change: Optional[Callable[[str], None]] = None,
    ) -> Optional[SessionCycle]:
        if require_guidance:
            if on_activity_change is not None:
                on_activity_change("guidance")
            proceed = run_guidance_phase(
                self.capture,
                self.detector,
                self.args,
                self.window_name,
                allow_resize=not self.window_adjusted,
                min_side=self.guidance_min_side,
                box_scale=self.guidance_box_scale,
                window_limits=self.window_limits,
                frame_transform=self.frame_transform,
            )
            if not proceed:
                return None
            self.window_adjusted = True
        else:
            if not self.window_adjusted:
                adjust_window_to_capture(self.capture, self.window_name, self.window_limits)
                self.window_adjusted = True
        if on_activity_change is not None:
            on_activity_change("verification")

        observations, last_frame, duration = run_verification_phase(
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
        )
        return SessionCycle(observations=observations, last_frame=last_frame, duration=duration)

    def wait_for_next_person(self) -> bool:
        """Return True to continue, False to exit."""
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(" "):
                return True
            if key in (27, ord("q")):
                return False


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
