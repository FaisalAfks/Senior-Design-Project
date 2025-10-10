#!/usr/bin/env python3
"""Convenience API for BlazeFace face detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from BlazeFace.detector import BlazeFaceDetector, Detection


@dataclass
class FaceCrop:
    detection: Detection
    aligned: Optional[np.ndarray]
    spoof_crop: Optional[np.ndarray]


class BlazeFaceService:
    """High-level interface around `BlazeFaceDetector`."""

    def __init__(
        self,
        *,
        score_threshold: float = 0.6,
        spoof_crop_expand: float = 0.15,
        spoof_crop_size: Tuple[int, int] = (224, 224),
        device: Optional[torch.device] = None,
    ) -> None:
        self.detector = BlazeFaceDetector(score_threshold=score_threshold, device=device)
        self.spoof_expand = spoof_crop_expand
        self.spoof_size = spoof_crop_size

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        return self.detector.detect(frame_bgr)

    def prepare_crops(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
        *,
        include_aligned: bool = True,
        include_spoof: bool = True,
    ) -> List[FaceCrop]:
        crops: List[FaceCrop] = []
        for det in detections:
            aligned = None
            spoof_crop = None
            if include_aligned:
                aligned = self.detector.align_face(frame_bgr, det)
            if include_spoof:
                spoof_crop = self.detector.crop_face(
                    frame_bgr,
                    det,
                    expand=self.spoof_expand,
                    output_size=self.spoof_size,
                )
            crops.append(FaceCrop(detection=det, aligned=aligned, spoof_crop=spoof_crop))
        return crops
