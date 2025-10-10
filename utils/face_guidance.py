"""Reusable helpers for face alignment guidance overlays."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from BlazeFace import BlazeFaceService, Detection
from .camera import open_capture


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

    angle_deg = 0.0
    leveled = True
    if detection.keypoints.shape[0] >= 2:
        right_eye = detection.keypoints[0]
        left_eye = detection.keypoints[1]
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        if dx != 0 or dy != 0:
            angle_deg = math.degrees(math.atan2(dy, dx))
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


def select_best_detection(detections: Sequence[Detection]) -> Detection:
    return max(detections, key=lambda d: d.score * max(1.0, d.area()))


def draw_guidance_overlay(
    frame: np.ndarray,
    *,
    box_center: Tuple[int, int],
    half_side: int,
    detection: Optional[Detection],
    assessment: Optional[AlignmentAssessment],
    show_messages: bool = True,
) -> np.ndarray:
    display = frame.copy()
    height, width = display.shape[:2]

    cx, cy = box_center
    half = max(1, half_side)
    left = max(0, cx - half)
    right = min(width - 1, cx + half)
    top = max(0, cy - half)
    bottom = min(height - 1, cy + half)

    box_color = (0, 255, 0)
    thickness = 4
    cv2.rectangle(display, (left, top), (right, bottom), box_color, thickness, cv2.LINE_AA)

    if detection is None or assessment is None:
        if show_messages:
            cv2.putText(display, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return display

    status_color = (0, 255, 0) if assessment.is_aligned else (0, 0, 255)
    cv2.putText(display, f"Tilt: {assessment.angle_deg:.1f} deg",
                (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    if show_messages:
        messages = assessment.messages or ["Hold steady..."]
        y_offset = 60
        for message in messages:
            cv2.putText(display, message, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            y_offset += 30

    return display


def run_guidance_session(
    source,
    *,
    device,
    box_size: int = 0,
    center_tolerance: float = 0.25,
    size_tolerance: float = 0.15,
    rotation_thr: float = 7.0,
    hold_frames: int = 15,
    window_name: str = "Face Guidance",
    camera_backend: str = "auto",
    camera_width: int = 1280,
    camera_height: int = 720,
    camera_fps: float = 30.0,
    camera_flip: int = 0,
    gstreamer_pipeline: Optional[str] = None,
    sensor_mode: Optional[int] = None,
    min_box_size: int = 224,
    box_scale: float = 0.40,
) -> bool:
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    capture = open_capture(
        source,
        backend=camera_backend,
        width=camera_width,
        height=camera_height,
        fps=camera_fps,
        flip_method=camera_flip,
        gstreamer_pipeline=gstreamer_pipeline,
        sensor_mode=sensor_mode,
    )
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    detector_service = BlazeFaceService(device=device)
    consecutive_good = 0
    success = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            min_dim = min(height, width)
            min_required_side = max(20, min_box_size)
            auto_side = max(min_required_side, int(min_dim * box_scale))
            if box_size > 0:
                side = max(min_required_side, box_size)
            else:
                side = auto_side
            max_side = max(min_required_side, min_dim - 20)
            side = min(side, max_side)
            if side % 2 != 0:
                side -= 1
            half_side = max(20, side // 2)
            box_center = (width // 2, height // 2)

            detections = detector_service.detect(frame)
            detection = select_best_detection(detections) if detections else None

            assessment = None
            if detection is not None:
                assessment = evaluate_alignment(
                    detection,
                    box_center,
                    half_side,
                    center_tolerance_ratio=center_tolerance,
                    size_tolerance_ratio=size_tolerance,
                    rotation_threshold=rotation_thr,
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

            if assessment and assessment.is_aligned and consecutive_good >= hold_frames:
                cv2.putText(
                    display,
                    "Alignment OK! Press SPACE to continue.",
                    (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32 and consecutive_good >= hold_frames:
                success = True
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return success
