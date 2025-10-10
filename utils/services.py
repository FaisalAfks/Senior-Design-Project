"""Factories for BlazeFace, MobileFaceNet, and DeePixBiS services."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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

