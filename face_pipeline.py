"""High-level pipeline orchestrating BlazeFace, MobileFaceNet, and DeePixBiS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from antispoof.antispoof_model import DeePixBiS
from detection.blazeface_detector import BlazeFaceDetector, Detection
from recognition import MobileFaceNet, l2_norm, load_facebank, prepare_facebank


@dataclass
class FaceObservation:
    """Unified per-face output from the guard pipeline."""

    detection: Detection
    identity: Optional[str]
    identity_score: Optional[float]
    is_recognized: Optional[bool]
    spoof_score: Optional[float]
    is_real: Optional[bool]


class FacePipeline:
    """Runs detection, recognition, and anti-spoofing in a single pass."""

    def __init__(
        self,
        *,
        recognition_weights: Path,
        facebank_dir: Path,
        spoof_weights: Optional[Path] = None,
        detection_threshold: float = 0.6,
        recognition_threshold: float = 60.0,
        spoof_threshold: float = 0.7,
        tta: bool = False,
        refresh_facebank: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.detector = BlazeFaceDetector(score_threshold=detection_threshold)
        self.recognition_threshold = recognition_threshold
        self.spoof_threshold = spoof_threshold
        self.use_tta = tta

        self._init_recogniser(Path(recognition_weights), Path(facebank_dir), refresh_facebank)
        self._init_spoof_model(Path(spoof_weights) if spoof_weights is not None else None)

    # ------------------------------------------------------------------
    def _init_recogniser(self, weights: Path, facebank_dir: Path, refresh_facebank: bool) -> None:
        if not weights.exists():
            raise FileNotFoundError(f"MobileFaceNet weights not found: {weights}")
        if not facebank_dir.exists():
            raise FileNotFoundError(f"Facebank directory not found: {facebank_dir}")

        self.recogniser = MobileFaceNet(512).to(self.device)
        self.recogniser.load_state_dict(torch.load(weights, map_location=self.device))
        self.recogniser.eval()

        self._rec_mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self._rec_std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

        if refresh_facebank:
            embeddings, names = prepare_facebank(
                self.recogniser,
                path=facebank_dir,
                tta=self.use_tta,
                detector=self.detector,
                device=self.device,
            )
        else:
            embeddings, names = load_facebank(path=facebank_dir)

        self.facebank_embeddings = embeddings.to(self.device)
        self.facebank_names = list(names)

    # ------------------------------------------------------------------
    def _init_spoof_model(self, weights: Optional[Path]) -> None:
        self.spoof_model: Optional[DeePixBiS] = None
        if weights is None:
            return
        if not weights.exists():
            raise FileNotFoundError(f"DeePixBiS weights not found: {weights}")

        self.spoof_model = DeePixBiS().to(self.device)
        self.spoof_model.load_state_dict(torch.load(weights, map_location=self.device))
        self.spoof_model.eval()
        self._spoof_mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        self._spoof_std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> List[FaceObservation]:
        detections = self.detector.detect(frame_bgr)
        if not detections:
            return []

        rec_faces: List[np.ndarray] = []
        rec_indices: List[int] = []
        spoof_faces: List[np.ndarray] = []
        spoof_indices: List[int] = []

        for idx, det in enumerate(detections):
            aligned = self.detector.align_face(frame_bgr, det)
            if aligned is not None:
                rec_faces.append(aligned)
                rec_indices.append(idx)

            if self.spoof_model is not None:
                crop = self.detector.crop_face(frame_bgr, det, expand=0.15, output_size=(224, 224))
                if crop is not None:
                    spoof_faces.append(crop)
                    spoof_indices.append(idx)

        recognised = self._recognise_batch(rec_faces) if rec_faces else []
        spoof_scores = self._spoof_batch(spoof_faces) if spoof_faces else []

        rec_lookup = {idx: data for idx, data in zip(rec_indices, recognised)}
        spoof_lookup = {idx: score for idx, score in zip(spoof_indices, spoof_scores)}

        observations: List[FaceObservation] = []
        for idx, det in enumerate(detections):
            identity = identity_score = spoof_score = None
            is_recognized = is_real = None

            if idx in rec_lookup:
                identity, identity_score, is_recognized = rec_lookup[idx]

            if idx in spoof_lookup:
                spoof_score = spoof_lookup[idx]
                is_real = spoof_score >= self.spoof_threshold

            observations.append(
                FaceObservation(
                    detection=det,
                    identity=identity,
                    identity_score=identity_score,
                    is_recognized=is_recognized,
                    spoof_score=spoof_score,
                    is_real=is_real,
                )
            )
        return observations

    # ------------------------------------------------------------------
    def _recognise_batch(self, faces_bgr: List[np.ndarray]) -> List[Tuple[str, float, bool]]:
        with torch.no_grad():
            rgb_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces_bgr]
            batch = torch.from_numpy(np.stack(rgb_faces, axis=0)).permute(0, 3, 1, 2).float()
            batch = batch.to(self.device, non_blocking=True) / 255.0
            batch = (batch - self._rec_mean) / self._rec_std

            if self.use_tta:
                flipped = torch.flip(batch, dims=[3])
                inputs = torch.cat([batch, flipped], dim=0)
                embeddings = self.recogniser(inputs)
                emb_orig, emb_mirror = embeddings.chunk(2, dim=0)
                source_embs = l2_norm(emb_orig + emb_mirror)
            else:
                source_embs = self.recogniser(batch)

            facebank = self.facebank_embeddings
            dist = (
                torch.sum(source_embs ** 2, dim=1, keepdim=True)
                + torch.sum(facebank ** 2, dim=1)
                - 2.0 * torch.matmul(source_embs, facebank.t())
            )
            dist = torch.clamp(dist, min=0.0)

            min_vals, min_idx = torch.min(dist, dim=1)
            scores = torch.clamp(min_vals * -80 + 156, 0, 100)
            threshold = (self.recognition_threshold - 156) / (-80)

        results: List[Tuple[str, float, bool]] = []
        for value, idx, score in zip(min_vals, min_idx, scores):
            recognized = bool(value <= threshold)
            name = "Unknown"
            if recognized and idx + 1 < len(self.facebank_names):
                name = self.facebank_names[idx + 1]
            results.append((name, float(score.item()), recognized))
        return results

    # ------------------------------------------------------------------
    def _spoof_batch(self, faces_bgr: List[np.ndarray]) -> List[float]:
        if self.spoof_model is None:
            return []

        with torch.no_grad():
            rgb_faces = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in faces_bgr]
            batch = torch.from_numpy(np.stack(rgb_faces, axis=0)).permute(0, 3, 1, 2).float()
            batch = batch.to(self.device, non_blocking=True) / 255.0
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = (batch - self._spoof_mean) / self._spoof_std
            masks, _ = self.spoof_model(batch)
            scores = masks.mean(dim=(1, 2, 3))
        return [float(score.item()) for score in scores]

    # ------------------------------------------------------------------
    @staticmethod
    def annotate(
        frame_bgr: np.ndarray,
        observations: List[FaceObservation],
        *,
        show_landmarks: bool = True,
        show_identity_score: bool = True,
        show_spoof_score: bool = True,
    ) -> np.ndarray:
        annotated = frame_bgr.copy()
        for obs in observations:
            x1, y1, x2, y2 = obs.detection.as_int_bbox()
            color = (0, 255, 0) if obs.is_real in (True, None) else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            labels: List[str] = []
            if obs.identity is not None:
                if show_identity_score and obs.identity_score is not None:
                    labels.append(f"{obs.identity} ({obs.identity_score:.0f})")
                else:
                    labels.append(obs.identity if obs.is_recognized else "Unknown")

            if obs.spoof_score is not None:
                status = "Real" if obs.is_real else "Fake"
                labels.append(f"{status}: {obs.spoof_score:.2f}" if show_spoof_score else status)

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


__all__ = ["FacePipeline", "FaceObservation"]


