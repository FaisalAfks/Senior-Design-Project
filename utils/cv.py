"""Shared OpenCV helpers for drawing face annotations."""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from BlazeFace import Detection



def draw_detection_labels(
    frame: np.ndarray,
    observations: Sequence["FaceObservationLike"],
    *,
    show_landmarks: bool = True,
    show_identity_score: bool = True,
    show_spoof_score: bool = True,
) -> np.ndarray:
    annotated = frame.copy()
    for obs in observations:
        x1, y1, x2, y2 = obs.detection.as_int_bbox()
        color = (0, 255, 0) if getattr(obs, "is_real", True) in (True, None) else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        labels: List[str] = []
        identity = getattr(obs, "identity", None)
        identity_score = getattr(obs, "identity_score", None)
        is_recognized = getattr(obs, "is_recognized", None)
        spoof_score = getattr(obs, "spoof_score", None)
        is_real = getattr(obs, "is_real", None)
        if identity is not None:
            if show_identity_score and identity_score is not None:
                labels.append(f"{identity} ({identity_score:.0f})")
            else:
                labels.append(identity if is_recognized else "Unknown")
        if spoof_score is not None:
            status = "Real" if is_real else "Fake"
            labels.append(f"{status}: {spoof_score:.2f}" if show_spoof_score else status)
        for i, text in enumerate(labels):
            cv2.putText(
                annotated,
                text,
                (x1, max(0, y1 - 10 - 18 * i)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        if show_landmarks:
            for px, py in obs.detection.keypoints:
                cv2.circle(annotated, (int(px), int(py)), 2, color, -1)
    return annotated



class FaceObservationLike:
    detection: Detection
    identity: Optional[str]
    identity_score: Optional[float]
    is_recognized: Optional[bool]
    spoof_score: Optional[float]
    is_real: Optional[bool]

