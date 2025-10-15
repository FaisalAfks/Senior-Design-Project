#!/usr/bin/env python3
"""High-level MobileFaceNet embeddings + facebank lookup service."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

from BlazeFace.detector import BlazeFaceDetector
from MobileFaceNet.models.mobilefacenet import (
    MobileFaceNet,
    l2_norm,
    load_facebank,
    prepare_facebank,
)


@dataclass
class RecognitionResult:
    name: str
    confidence: float
    is_recognized: bool


class MobileFaceNetService:
    """Handles embedding extraction and identity lookup via a facebank."""

    def __init__(
        self,
        weights_path: Path | str,
        facebank_dir: Path | str,
        *,
        detector: Optional[BlazeFaceDetector] = None,
        device: Optional[torch.device] = None,
        recognition_threshold: float = 0.7,
        tta: bool = False,
        refresh_facebank: bool = False,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        weights_path = Path(weights_path)
        facebank_dir = Path(facebank_dir)
        if not weights_path.exists():
            raise FileNotFoundError(f"MobileFaceNet weights not found: {weights_path}")
        if not facebank_dir.exists():
            raise FileNotFoundError(f"Facebank directory not found: {facebank_dir}")

        self.model = MobileFaceNet(512).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        self.detector = detector or BlazeFaceDetector()
        self.tta = tta
        if recognition_threshold > 1.0:
            recognition_threshold = recognition_threshold / 100.0
        recognition_threshold = float(np.clip(recognition_threshold, 0.0, 1.0))
        self.recognition_threshold = recognition_threshold

        self._rec_mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self._rec_std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

        if refresh_facebank:
            embeddings, names = prepare_facebank(
                self.model,
                path=facebank_dir,
                tta=self.tta,
                detector=self.detector,
                device=self.device,
            )
        else:
            embeddings, names = load_facebank(facebank_dir)

        self.facebank_embeddings = embeddings.to(self.device)
        self.facebank_names = list(names)

    def _normalise(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, non_blocking=True) / 255.0
        return (batch - self._rec_mean) / self._rec_std

    def recognise_faces(self, faces_bgr: Sequence[np.ndarray]) -> List[RecognitionResult]:
        if not faces_bgr:
            return []

        with torch.no_grad():
            rgb_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces_bgr]
            batch = torch.from_numpy(np.stack(rgb_faces, axis=0)).permute(0, 3, 1, 2).float()
            inputs = self._normalise(batch)

            if self.tta:
                flipped = torch.flip(inputs, dims=[3])
                embeds = self.model(torch.cat([inputs, flipped], dim=0))
                emb_orig, emb_mirror = embeds.chunk(2, dim=0)
                embeddings = l2_norm(emb_orig + emb_mirror)
            else:
                embeddings = self.model(inputs)

            facebank = self.facebank_embeddings
            dist = (
                torch.sum(embeddings ** 2, dim=1, keepdim=True)
                + torch.sum(facebank ** 2, dim=1)
                - 2.0 * torch.matmul(embeddings, facebank.t())
            )
            dist = torch.clamp(dist, min=0.0)

            min_vals, min_idx = torch.min(dist, dim=1)
            scores = torch.clamp(-0.8 * min_vals + 1.56, 0.0, 1.0)

        results: List[RecognitionResult] = []
        for value, idx, score in zip(min_vals, min_idx, scores):
            recognized = bool(score >= self.recognition_threshold)
            name = "Unknown"
            if recognized and idx + 1 < len(self.facebank_names):
                name = self.facebank_names[idx + 1]
            results.append(RecognitionResult(name=name, confidence=float(score.item()), is_recognized=recognized))
        return results

    def detect_and_align(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """Helper that leverages BlazeFace to detect & align faces."""
        faces: List[np.ndarray] = []
        for detection in self.detector.detect(frame_bgr):
            face = self.detector.align_face(frame_bgr, detection)
            if face is not None:
                faces.append(face)
        return faces

    def rebuild_facebank(self, facebank_dir: Path | str) -> None:
        embeddings, names = prepare_facebank(
            self.model,
            path=Path(facebank_dir),
            tta=self.tta,
            detector=self.detector,
            device=self.device,
        )
        self.facebank_embeddings = embeddings.to(self.device)
        self.facebank_names = list(names)
