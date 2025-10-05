"""BlazeFace detector wrapper with eye/nose/mouth alignment helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from detection.blazeface import BlazeFace

_CANONICAL_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],  # right eye
        [73.5318, 51.5014],  # left eye
        [56.0252, 71.7366],  # nose tip
        [56.0000, 92.2041],  # mouth centre
    ],
    dtype=np.float32,
)


@dataclass
class Detection:
    """Face detection result in pixel coordinates."""

    bbox: Tuple[float, float, float, float]
    keypoints: np.ndarray  # shape (6, 2)
    score: float

    def as_int_bbox(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class BlazeFaceDetector:
    """Convenience wrapper around the converted MediaPipe BlazeFace model."""

    def __init__(
        self,
        *,
        back_model: bool = False,
        device: Optional[torch.device] = None,
        weights_path: Optional[Path] = None,
        anchors_path: Optional[Path] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        asset_dir = Path(__file__).resolve().parent / "blazeface_assets"
        weights_path = weights_path or asset_dir / ("blazeface_back.pth" if back_model else "blazeface.pth")
        anchors_path = anchors_path or asset_dir / ("anchors_back.npy" if back_model else "anchors.npy")

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not anchors_path.exists():
            raise FileNotFoundError(f"Anchors file not found: {anchors_path}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BlazeFace(back_model=back_model)
        self.model.load_weights(str(weights_path))
        self.model.load_anchors(str(anchors_path))
        self.model.to(self.device).eval()

        self.input_size = 256 if back_model else 128
        default_thr = self.model.min_score_thresh
        self.score_threshold = float(score_threshold if score_threshold is not None else default_thr)

    # ------------------------------------------------------------------
    def detect(self, image_bgr: np.ndarray, *, score_threshold: Optional[float] = None) -> List[Detection]:
        """Return detections in pixel coordinates for the provided BGR frame."""
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("detect expects a HxWx3 BGR image")

        square, (left, top, side) = _center_square_crop(image_bgr)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)

        raw_det = self.model.predict_on_image(resized)
        detections = _to_numpy(raw_det)
        if detections.size == 0:
            return []

        threshold = float(score_threshold if score_threshold is not None else self.score_threshold)
        return self._convert_detections(detections, left, top, side, image_bgr.shape[:2], threshold)

    # ------------------------------------------------------------------
    def align_face(
        self,
        image_bgr: np.ndarray,
        detection: Detection,
        *,
        output_size: Tuple[int, int] = (112, 112),
    ) -> Optional[np.ndarray]:
        """Return an aligned face crop using BlazeFace landmarks."""
        keypoints = detection.keypoints.astype(np.float32)
        if keypoints.shape[0] < _CANONICAL_LANDMARKS.shape[0]:
            return None

        # Limit the affine fit to eyes/nose/mouth to avoid ear-driven drift.
        src = keypoints[: _CANONICAL_LANDMARKS.shape[0]]
        dst = _CANONICAL_LANDMARKS[: src.shape[0]]
        matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if matrix is None:
            return self.crop_face(image_bgr, detection, output_size=output_size)
        return cv2.warpAffine(image_bgr, matrix, output_size, flags=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    def crop_face(
        self,
        image_bgr: np.ndarray,
        detection: Detection,
        *,
        expand: float = 0.1,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None

        expand_x = width * expand
        expand_y = height * expand
        xmin = max(0, int(np.floor(x1 - expand_x)))
        ymin = max(0, int(np.floor(y1 - expand_y)))
        xmax = min(image_bgr.shape[1], int(np.ceil(x2 + expand_x)))
        ymax = min(image_bgr.shape[0], int(np.ceil(y2 + expand_y)))
        if xmax <= xmin or ymax <= ymin:
            return None

        crop = image_bgr[ymin:ymax, xmin:xmax]
        if output_size is not None and crop.size > 0:
            crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
        return crop if crop.size > 0 else None

    # ------------------------------------------------------------------
    def detect_and_align(
        self,
        image_bgr: np.ndarray,
        *,
        score_threshold: Optional[float] = None,
        output_size: Tuple[int, int] = (112, 112),
    ) -> List[np.ndarray]:
        faces: List[np.ndarray] = []
        for det in self.detect(image_bgr, score_threshold=score_threshold):
            face = self.align_face(image_bgr, det, output_size=output_size)
            if face is not None:
                faces.append(face)
        return faces

    # ------------------------------------------------------------------
    def _convert_detections(
        self,
        detections: np.ndarray,
        left: int,
        top: int,
        side: int,
        image_hw: Sequence[int],
        score_thr: float,
    ) -> List[Detection]:
        image_h, image_w = image_hw
        results: List[Detection] = []

        for det in detections.reshape(-1, detections.shape[-1]):
            if det.shape[0] != 17:
                continue
            score = float(det[16])
            if score < score_thr:
                continue

            ymin, xmin, ymax, xmax = det[0:4]
            x1 = xmin * side + left
            x2 = xmax * side + left
            y1 = ymin * side + top
            y2 = ymax * side + top

            keypoints = det[4:16].reshape(6, 2)
            keypoints[:, 0] = np.clip(keypoints[:, 0] * side + left, 0, image_w - 1)
            keypoints[:, 1] = np.clip(keypoints[:, 1] * side + top, 0, image_h - 1)

            bbox = (
                float(np.clip(x1, 0, image_w - 1)),
                float(np.clip(y1, 0, image_h - 1)),
                float(np.clip(x2, 0, image_w - 1)),
                float(np.clip(y2, 0, image_h - 1)),
            )
            results.append(Detection(bbox=bbox, keypoints=keypoints.astype(np.float32), score=score))
        return results


def _center_square_crop(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    height, width = frame.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame[top : top + side, left : left + side], (left, top, side)


def _to_numpy(raw_det: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(raw_det, torch.Tensor):
        return raw_det.detach().cpu().numpy()
    return np.asarray(raw_det)


__all__ = ["BlazeFaceDetector", "Detection"]
