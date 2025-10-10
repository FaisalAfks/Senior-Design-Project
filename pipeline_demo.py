#!/usr/bin/env python3
"""Unified CLI that orchestrates BlazeFace, MobileFaceNet, and DeePixBiS services."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BlazeFace import BlazeFaceService, Detection
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService, RecognitionResult
from utils import (
    draw_detection_labels,
    resolve_device,
    run_guidance_session,
)

DEFAULT_WEIGHTS = ROOT / "MobileFaceNet" / "Weights" / "MobileFace_Net"
DEFAULT_FACEBANK = ROOT / "facebank"
DEFAULT_SPOOF_WEIGHTS = ROOT / "DeePixBis" / "Weights" / "DeePixBiS.pth"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


@dataclass
class FaceObservation:
    detection: Detection
    identity: Optional[str]
    identity_score: Optional[float]
    is_recognized: Optional[bool]
    spoof_score: Optional[float]
    is_real: Optional[bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face Guardian unified demo")
    parser.add_argument("--source", default="0",
                        help="camera index (e.g. 0) or path to an image/video")
    parser.add_argument("-th", "--threshold", default=70.0, type=float,
                        help="identity acceptance threshold on the 0-100 score scale")
    parser.add_argument("--spoof-thr", default=0.9, type=float,
                        help="minimum DeePixBiS score to label a face as real")
    parser.add_argument("--detector-thr", default=0.7, type=float,
                        help="minimum BlazeFace confidence required to keep detections")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="path to MobileFaceNet weights")
    parser.add_argument("--facebank", default=str(DEFAULT_FACEBANK),
                        help="path to facebank directory")
    parser.add_argument("--spoof-weights", default=str(DEFAULT_SPOOF_WEIGHTS),
                        help="path to DeePixBiS weights")
    parser.add_argument("-u", "--update", action="store_true",
                        help="rebuild the facebank before running")
    parser.add_argument("-tta", "--tta", action="store_true",
                        help="enable flip test-time augmentation for recognition embeddings")
    parser.add_argument("--device", default="cpu",
                        help="torch device to run on (e.g. cpu, cuda, cuda:0)")
    parser.add_argument("--no-score", action="store_true",
                        help="hide identity confidence values in the overlay")
    parser.add_argument("--no-spoof-score", action="store_true",
                        help="hide DeePixBiS scores in the overlay")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="hide landmark dots in the overlay")
    parser.add_argument("--output", default=None,
                        help="optional path to save annotated output (image or video)")
    parser.add_argument("--disable-spoof", action="store_true",
                        help="skip DeePixBiS anti-spoofing")
    parser.add_argument("--guidance", action="store_true",
                        help="run the face alignment guide before processing")
    parser.add_argument("--guidance-circle-radius", type=int, default=0,
                        help="radius in pixels for the guidance circle (0 = auto)")
    parser.add_argument("--guidance-center-tolerance", type=float, default=0.25,
                        help="fraction of the radius tolerated for centering")
    parser.add_argument("--guidance-size-tolerance", type=float, default=0.15,
                        help="fractional tolerance for face size vs circle")
    parser.add_argument("--guidance-rotation-thr", type=float, default=7.0,
                        help="maximum allowed head tilt in degrees during guidance")
    parser.add_argument("--guidance-hold-frames", type=int, default=15,
                        help="number of consecutive aligned frames before guidance succeeds")
    return parser.parse_args()


def classify_source(src: str) -> Tuple[str, object]:
    path = Path(src)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            return "image", path
        if suffix in VIDEO_EXTS:
            return "video", path
    if src.isdigit():
        return "camera", int(src)
    return "video", src


def build_services(args: argparse.Namespace, device: torch.device):
    detector = BlazeFaceService(score_threshold=args.detector_thr, device=device)
    recogniser = MobileFaceNetService(
        weights_path=Path(args.weights),
        facebank_dir=Path(args.facebank),
        detector=detector.detector,
        device=device,
        recognition_threshold=args.threshold,
        tta=args.tta,
        refresh_facebank=args.update,
    )
    spoiler = None
    if not args.disable_spoof:
        spoiler = DeePixBiSService(weights_path=Path(args.spoof_weights), device=device)
    return detector, recogniser, spoiler


def process_frame(
    frame_bgr: np.ndarray,
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_model: Optional[DeePixBiSService],
    *,
    spoof_threshold: float,
) -> List[FaceObservation]:
    detections = detector.detect(frame_bgr)
    if not detections:
        return []

    crops = detector.prepare_crops(frame_bgr, detections, include_aligned=True, include_spoof=spoof_model is not None)

    aligned_faces: List[np.ndarray] = []
    alignment_indices: List[int] = []

    spoof_faces: List[np.ndarray] = []
    spoof_indices: List[int] = []

    for idx, crop in enumerate(crops):
        if crop.aligned is not None:
            aligned_faces.append(crop.aligned)
            alignment_indices.append(idx)
        if spoof_model is not None and crop.spoof_crop is not None:
            spoof_faces.append(crop.spoof_crop)
            spoof_indices.append(idx)

    rec_lookup = {}
    if aligned_faces:
        results = recogniser.recognise_faces(aligned_faces)
        rec_lookup = {idx: res for idx, res in zip(alignment_indices, results)}

    spoof_lookup = {}
    if spoof_model is not None and spoof_faces:
        scores = spoof_model.predict_scores(spoof_faces)
        spoof_lookup = {idx: score for idx, score in zip(spoof_indices, scores)}

    observations: List[FaceObservation] = []
    for idx, crop in enumerate(crops):
        detection = crop.detection

        identity = None
        identity_score = None
        is_recognized = None
        if idx in rec_lookup:
            rec: RecognitionResult = rec_lookup[idx]
            identity = rec.name
            identity_score = rec.confidence
            is_recognized = rec.is_recognized

        spoof_score = None
        is_real = None
        if idx in spoof_lookup:
            spoof_score = spoof_lookup[idx]
            is_real = spoof_score >= spoof_threshold

        observations.append(
            FaceObservation(
                detection=detection,
                identity=identity,
                identity_score=identity_score,
                is_recognized=is_recognized,
                spoof_score=spoof_score,
                is_real=is_real,
            )
        )
    return observations


def log_results(observations: List[FaceObservation], prefix: str = "") -> None:
    if not observations:
        print(prefix + "No faces detected")
        return
    for idx, obs in enumerate(observations, 1):
        identity = obs.identity or "Unknown"
        identity_score = f"{obs.identity_score:.0f}" if obs.identity_score is not None else "--"
        spoof_score = f"{obs.spoof_score:.2f}" if obs.spoof_score is not None else "--"
        verdict = "Real" if obs.is_real else "Fake" if obs.is_real is not None else "Unknown"
        print(f"{prefix}Face #{idx}: {identity} (score {identity_score}) | {verdict} ({spoof_score})")


def annotate(
    frame_bgr: np.ndarray,
    observations: List[FaceObservation],
    *,
    show_landmarks: bool = True,
    show_identity_score: bool = True,
    show_spoof_score: bool = True,
) -> np.ndarray:
    return draw_detection_labels(
        frame_bgr,
        observations,
        show_landmarks=show_landmarks,
        show_identity_score=show_identity_score,
        show_spoof_score=show_spoof_score,
    )


def run_image_mode(
    services: Tuple[BlazeFaceService, MobileFaceNetService, Optional[DeePixBiSService]],
    path: Path,
    args: argparse.Namespace,
) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    detector, recogniser, spoiler = services
    observations = process_frame(image, detector, recogniser, spoiler, spoof_threshold=args.spoof_thr)
    log_results(observations)
    annotated = annotate(
        image,
        observations,
        show_landmarks=not args.no_landmarks,
        show_identity_score=not args.no_score,
        show_spoof_score=not args.no_spoof_score,
    )
    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"Annotated image saved to {args.output}")
    cv2.imshow("Face Recognition System", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_mode(
    services: Tuple[BlazeFaceService, MobileFaceNetService, Optional[DeePixBiSService]],
    source,
    args: argparse.Namespace,
) -> None:
    detector, recogniser, spoiler = services
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read from the provided source.")
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
            True,
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        start = time.time()
        observations = process_frame(frame, detector, recogniser, spoiler, spoof_threshold=args.spoof_thr)
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        log_results(observations, prefix=f"[{frame_index}] ")

        annotated = annotate(
            frame,
            observations,
            show_landmarks=not args.no_landmarks,
            show_identity_score=not args.no_score,
            show_spoof_score=not args.no_spoof_score,
        )
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(annotated)
        cv2.imshow("Face Recognition System", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    mode, source = classify_source(args.source)
    device = resolve_device(args.device)
    if args.guidance:
        if mode != "camera":
            print("Face guidance is only available for live camera sources; skipping guidance step.")
        else:
            guidance_success = run_guidance_session(
                args.source,
                device=device,
                circle_radius=args.guidance_circle_radius,
                center_tolerance=args.guidance_center_tolerance,
                size_tolerance=args.guidance_size_tolerance,
                rotation_thr=args.guidance_rotation_thr,
                hold_frames=args.guidance_hold_frames,
            )
            if not guidance_success:
                print("Face alignment was not confirmed. Exiting without running the pipeline.")
                return

    services = build_services(args, device)

    if mode == "image":
        run_image_mode(services, Path(source), args)
    else:
        video_source = str(source) if mode == "video" else source
        run_video_mode(services, video_source, args)


if __name__ == "__main__":
    main()
