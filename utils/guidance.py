"""Guidance helpers for face alignment overlay and confirmation loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from BlazeFace import BlazeFaceService, Detection
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


def draw_guidance_overlay(
    frame: np.ndarray,
    *,
    box_center: Tuple[int, int],
    half_side: int,
    detection: Optional[Detection],
    assessment: Optional[AlignmentAssessment],
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

    if detection is None or assessment is None:
        cv2.putText(display, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return display

    status_color = (0, 255, 0) if assessment.is_aligned else (0, 0, 255)
    cv2.putText(
        display,
        f"Tilt: {assessment.angle_deg:.1f} deg",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
    )

    messages = assessment.messages or ["Hold steady..."]
    y_offset = 70
    for message in messages:
        cv2.putText(
            display,
            message,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        y_offset += 28

    return display


def run_guidance_phase(
    capture: cv2.VideoCapture,
    detector: BlazeFaceService,
    args,
    window_name: str,
    *,
    allow_resize: bool,
    min_side: int,
    box_scale: float,
    window_limits: Tuple[int, int],
) -> bool:
    consecutive_good = 0
    window_resized = False
    width_limit, height_limit = window_limits

    while True:
        ok, frame = capture.read()
        if not ok:
            return False

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

        if allow_resize and not window_resized:
            target_w = min(width, width_limit)
            target_h = min(height, height_limit)
            cv2.resizeWindow(window_name, target_w, target_h)
            window_resized = True

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
        )
        instruction_lines: List[Tuple[str, Tuple[int, int, int], float]] = []
        instruction_lines.append(("Align your face within the square", (255, 255, 255), 0.8))

        if detection is None:
            instruction_lines.append(("Face not detected, center yourself", (0, 0, 255), 0.7))
            instruction_lines.append(("Look toward the camera", (0, 0, 255), 0.7))
        elif assessment is not None:
            if assessment.is_aligned:
                instruction_lines.append(("Hold steady to confirm", (0, 255, 0), 0.75))
                instruction_lines.append((f"Progress: {consecutive_good}/{args.guidance_hold_frames}", (0, 255, 0), 0.7))
            else:
                messages = assessment.messages or ["Adjust your position"]
                for msg in messages:
                    instruction_lines.append((msg, (0, 165, 255), 0.7))
                instruction_lines.append((f"Progress: {consecutive_good}/{args.guidance_hold_frames}", (0, 165, 255), 0.7))
        else:
            instruction_lines.append(("Detecting face...", (0, 165, 255), 0.7))

        base_y = 40
        line_spacing = 28
        for text, color, scale in instruction_lines:
            cv2.putText(display, text, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            base_y += int(line_spacing * scale / 0.7)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False

        if consecutive_good >= args.guidance_hold_frames:
            return True

