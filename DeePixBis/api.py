#!/usr/bin/env python3
"""High-level DeePixBiS inference utilities."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
 

from .core import DeePixBiS, load_checkpoint


class DeePixBiSService:
    """Convenience wrapper around DeePixBiS for batched inference."""

    def __init__(
        self,
        weights_path: Optional[Path | str] = None,
        *,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeePixBiS().to(self.device).eval()
        if weights_path:
            load_checkpoint(self.model, Path(weights_path), device=self.device)
        self.threshold = float(threshold)
        self._mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

    def to(self, device: torch.device) -> "DeePixBiSService":
        self.device = device
        self.model.to(device)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        return self

    def preprocess(self, faces_bgr: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert a sequence of BGR faces into a normalised tensor."""
        rgb_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces_bgr]
        batch = torch.from_numpy(np.stack(rgb_faces, axis=0)).permute(0, 3, 1, 2).float()
        batch = batch.to(self.device, non_blocking=True) / 255.0
        batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
        return (batch - self._mean) / self._std

    @torch.no_grad()
    def predict_scores(self, faces_bgr: Sequence[np.ndarray]) -> List[float]:
        """Return DeePixBiS spoof scores for the provided faces."""
        if not faces_bgr:
            return []
        inputs = self.preprocess(faces_bgr)
        masks, scores = self.model(inputs)
        spoof_scores = masks.mean(dim=(1, 2, 3))
        return [float(score.item()) for score in spoof_scores]

    def classify(self, faces_bgr: Sequence[np.ndarray]) -> List[bool]:
        """Label each face as real (True) or fake (False)."""
        return [score >= self.threshold for score in self.predict_scores(faces_bgr)]

