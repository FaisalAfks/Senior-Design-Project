"""Guidance helpers for face alignment overlay and confirmation loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from BlazeFace import BlazeFaceService, Detection
from utils.overlay import BannerStyle, draw_center_banner
from .orientation import compute_orientation


@dataclass
class AlignmentAssessment:
    centered: bool
    sized: bool
    leveled: bool
    messages: List[str]
    angle_deg: float

    @property
    def is_aligned(self) -> bool:
        return self.centered and self.sized and self.leveled


def evaluate_alignment(
    detection: Detection,
    box_center: Tuple[float, float],
    half_side: float,
    *,
    center_tolerance_ratio: float,
    size_tolerance_ratio: float,
    rotation_threshold: float,
) -> AlignmentAssessment:
    x1, y1, x2, y2 = detection.bbox
    face_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    box_center = np.array(box_center, dtype=np.float32)
    delta = face_center - box_center
    tolerance_px = center_tolerance_ratio * half_side

    messages: List[str] = []
    centered = True
    if delta[0] > tolerance_px:
        messages.append("Move slightly left")
        centered = False
    elif delta[0] < -tolerance_px:
        messages.append("Move slightly right")
        centered = False
    if delta[1] > tolerance_px:
        messages.append("Raise your head")
        centered = False
    elif delta[1] < -tolerance_px:
        messages.append("Lower your head")
        centered = False

    face_size = max((x2 - x1), (y2 - y1))
    target_size = half_side * 2.0
    size_tolerance = size_tolerance_ratio * target_size
    sized = True
    if face_size < target_size - size_tolerance:
        messages.append("Move closer to the camera")
        sized = False
    elif face_size > target_size + size_tolerance:
        messages.append("Move slightly back")
        sized = False

    orientation = compute_orientation(detection)
    angle_deg = orientation.roll_deg
    leveled = True
    if abs(angle_deg) > rotation_threshold:
        direction = "counter-clockwise" if angle_deg < 0 else "clockwise"
        messages.append(f"Tilt your head {direction}")
        leveled = False

    if not messages and not (centered and sized and leveled):
        messages.append("Adjust your position")

    return AlignmentAssessment(
        centered=centered,
        sized=sized,
        leveled=leveled,
        messages=messages,
        angle_deg=angle_deg,
    )


GUIDANCE_TOP_STYLE = BannerStyle(bg_color=(20, 20, 20), text_color=(255, 255, 255), alpha=0.7, font_scale=0.7, margin=24)
GUIDANCE_BOTTOM_STYLE = BannerStyle(bg_color=(0, 0, 0), text_color=(255, 255, 255), alpha=0.75, font_scale=0.8, margin=32)


def draw_guidance_overlay(
    frame: np.ndarray,
    *,
    box_center: Tuple[int, int],
    half_side: int,
    detection: Optional[Detection],
    assessment: Optional[AlignmentAssessment],
    consecutive_good: int,
    hold_frames: int,
) -> np.ndarray:
    display = frame.copy()
    height, width = display.shape[:2]

    cx, cy = box_center
    half = max(1, half_side)
    left = max(0, cx - half)
    right = min(width - 1, cx + half)
    top = max(0, cy - half)
    bottom = min(height - 1, cy + half)

    cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 4, cv2.LINE_AA)

    if detection is None:
        display = draw_center_banner(
            display,
            "No face detected",
            position="top",
            style=GUIDANCE_TOP_STYLE,
        )
        return display

    if assessment is None:
        return display

    instruction = _instruction_message(
        detection,
        assessment,
        consecutive_good,
        hold_frames,
    )

    if instruction:
        display = draw_center_banner(
            display,
            instruction,
            position="bottom",
            style=GUIDANCE_BOTTOM_STYLE,
        )

    return display


def _instruction_message(
    detection: Optional[Detection],
    assessment: Optional[AlignmentAssessment],
    consecutive_good: int,
    hold_frames: int,
) -> str:
    if detection is None or assessment is None:
        return ""
    if assessment.is_aligned:
        progress = f"{min(consecutive_good, hold_frames)}/{hold_frames}"
        return f"Hold steady... ({progress})"
    message = (assessment.messages or ["Adjust your position"])[0]
    return message


def run_guidance_phase(
    capture: cv2.VideoCapture,
    detector: BlazeFaceService,
    args,
    *,
    min_side: int,
    box_scale: float,
    frame_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    display_callback: Optional[Callable[[np.ndarray], None]] = None,
    poll_cancel: Optional[Callable[[], bool]] = None,
) -> bool:
    """Drive the alignment logic independent of any specific UI."""
    consecutive_good = 0
    hold_frames = max(1, getattr(args, "guidance_hold_frames", 1))

    while True:
        if poll_cancel is not None and poll_cancel():
            return False
        ok, frame = capture.read()
        if not ok:
            return False
        if frame_transform is not None:
            frame = frame_transform(frame)

        height, width = frame.shape[:2]
        min_dim = min(height, width)
        min_required_side = min_side
        auto_side = max(min_required_side, int(min_dim * box_scale))
        if args.guidance_box_size > 0:
            side = max(min_required_side, args.guidance_box_size)
        else:
            side = auto_side
        max_side = max(min_required_side, min_dim - 20)
        side = min(side, max_side)
        if side % 2 != 0:
            side = max(min_required_side, side - 1)
        half_side = max(20, side // 2)
        box_center = (width // 2, height // 2)

        detections = detector.detect(frame)
        detection = max(detections, key=lambda det: det.score * max(det.area(), 1.0)) if detections else None
        assessment = None
        if detection is not None:
            assessment = evaluate_alignment(
                detection,
                box_center,
                half_side,
                center_tolerance_ratio=args.guidance_center_tolerance,
                size_tolerance_ratio=args.guidance_size_tolerance,
                rotation_threshold=args.guidance_rotation_thr,
            )
            consecutive_good = consecutive_good + 1 if assessment.is_aligned else 0
        else:
            consecutive_good = 0

        display = draw_guidance_overlay(
            frame,
            box_center=box_center,
            half_side=half_side,
            detection=detection,
            assessment=assessment,
            consecutive_good=consecutive_good,
            hold_frames=hold_frames,
        )

        if display_callback is not None:
            display_callback(display)

        if poll_cancel is not None and poll_cancel():
            return False

        if consecutive_good >= hold_frames:
            return True
