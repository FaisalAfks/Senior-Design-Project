"""Shared orientation helpers for BlazeFace detections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Set

import numpy as np

from BlazeFace.detector import Detection

# Default ordering for canonical orientation buckets.
DEFAULT_ORIENTATION_CATEGORIES: Sequence[str] = ("yaw_left", "yaw_right", "pitch_up", "pitch_down")


@dataclass(frozen=True)
class OrientationFeatures:
    roll_deg: float
    yaw_offset: float
    pitch_offset: float


def compute_orientation(detection: Detection) -> OrientationFeatures:
    """Estimate roll/yaw/pitch proxies from BlazeFace keypoints."""
    keypoints = detection.keypoints.astype(np.float32)
    if keypoints.shape[0] < 4:
        return OrientationFeatures(roll_deg=0.0, yaw_offset=0.0, pitch_offset=0.0)

    right_eye = keypoints[0]
    left_eye = keypoints[1]
    nose_tip = keypoints[2]
    mouth_center = keypoints[3]

    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    roll_deg = float(np.degrees(np.arctan2(dy, dx))) if dx != 0 or dy != 0 else 0.0

    inter_eye = float(np.linalg.norm(left_eye - right_eye))
    if inter_eye <= 1e-6:
        inter_eye = 1.0
    midpoint = (left_eye + right_eye) * 0.5
    yaw_offset = float((nose_tip[0] - midpoint[0]) / inter_eye)

    x1, y1, x2, y2 = detection.bbox
    bbox_height = max(1.0, y2 - y1)
    bbox_center_y = y1 + bbox_height * 0.5
    pitch_offset = float((nose_tip[1] - bbox_center_y) / bbox_height)
    mouth_delta = float((mouth_center[1] - bbox_center_y) / bbox_height)
    pitch_offset = float(0.7 * pitch_offset + 0.3 * mouth_delta)

    return OrientationFeatures(roll_deg=roll_deg, yaw_offset=yaw_offset, pitch_offset=pitch_offset)


def orientation_distance(a: OrientationFeatures, b: OrientationFeatures) -> float:
    """Heuristic distance between two orientations."""
    yaw_diff = abs(a.yaw_offset - b.yaw_offset)
    pitch_diff = abs(a.pitch_offset - b.pitch_offset)
    roll_diff = abs(a.roll_deg - b.roll_deg) / 180.0
    return yaw_diff * 1.5 + pitch_diff * 1.2 + roll_diff * 0.3


def categorise_orientation(
    orientation: OrientationFeatures,
    *,
    yaw_thr: float,
    pitch_thr: float,
) -> Set[str]:
    categories: Set[str] = set()
    if orientation.yaw_offset >= yaw_thr:
        categories.add("yaw_left")
    elif orientation.yaw_offset <= -yaw_thr:
        categories.add("yaw_right")
    if orientation.pitch_offset <= -pitch_thr:
        categories.add("pitch_up")
    elif orientation.pitch_offset >= pitch_thr:
        categories.add("pitch_down")
    return categories


def describe_orientation(
    orientation: OrientationFeatures,
    *,
    yaw_thr: float,
    pitch_thr: float,
) -> str:
    yaw_val = orientation.yaw_offset
    if yaw_val >= yaw_thr:
        yaw_label = f"left({yaw_val:+.2f})"
    elif yaw_val <= -yaw_thr:
        yaw_label = f"right({yaw_val:+.2f})"
    else:
        yaw_label = f"front({yaw_val:+.2f})"

    pitch_val = orientation.pitch_offset
    if pitch_val <= -pitch_thr:
        pitch_label = f"up({pitch_val:+.2f})"
    elif pitch_val >= pitch_thr:
        pitch_label = f"down({pitch_val:+.2f})"
    else:
        pitch_label = f"level({pitch_val:+.2f})"

    return f"yaw={yaw_label}, pitch={pitch_label}, roll={orientation.roll_deg:+.1f}"


def orientation_category_significance(category: str, orientation: OrientationFeatures) -> float:
    """Return a positive value representing how strongly this orientation supports the category."""
    if category == "yaw_left":
        return max(0.0, orientation.yaw_offset)
    if category == "yaw_right":
        return max(0.0, -orientation.yaw_offset)
    if category == "pitch_up":
        return max(0.0, -orientation.pitch_offset)
    if category == "pitch_down":
        return max(0.0, orientation.pitch_offset)
    return 0.0
