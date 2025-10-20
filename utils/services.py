"""Factories for BlazeFace, MobileFaceNet, and DeePixBiS services."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from BlazeFace import BlazeFaceService
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService


def create_services(
    args,
    device,
) -> Tuple[BlazeFaceService, MobileFaceNetService, Optional[DeePixBiSService]]:
    detector = BlazeFaceService(score_threshold=args.detector_thr, device=device)
    recogniser = MobileFaceNetService(
        weights_path=Path(args.weights),
        facebank_dir=Path(args.facebank),
        detector=detector.detector,
        device=device,
        recognition_threshold=args.identity_thr,
        tta=args.tta,
        refresh_facebank=args.update_facebank,
    )
    spoiler: Optional[DeePixBiSService] = None
    if not args.disable_spoof:
        spoiler = DeePixBiSService(weights_path=Path(args.spoof_weights), device=device)
    return detector, recogniser, spoiler


def warmup_services(
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_service: Optional[DeePixBiSService],
    *,
    frame_size: Tuple[int, int] = (640, 480),
    iters: int = 2,
) -> None:
    """Run a few dry-forward passes to initialize CUDA/cuDNN.

    This avoids an initial stall during the first verification cycle by
    triggering model initializations and algorithm selection outside of
    the timed capture window.
    """
    h, w = frame_size[1], frame_size[0]
    # Use simple constant images of the right sizes to exercise models.
    dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
    dummy_face_112 = np.zeros((112, 112, 3), dtype=np.uint8)
    dummy_face_224 = np.zeros((224, 224, 3), dtype=np.uint8)

    for _ in range(max(1, iters)):
        try:
            _ = detector.detect(dummy_frame)
        except Exception:
            pass
        try:
            _ = recogniser.recognise_faces([dummy_face_112])
        except Exception:
            pass
        if spoof_service is not None:
            try:
                _ = spoof_service.predict_scores([dummy_face_224])
            except Exception:
                pass

    # Ensure all queued CUDA kernels complete before proceeding.
    try:
        if hasattr(recogniser, "device") and isinstance(recogniser.device, torch.device) and recogniser.device.type == "cuda":
            torch.cuda.synchronize(recogniser.device)
    except Exception:
        pass

