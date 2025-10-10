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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BlazeFace import BlazeFaceService, Detection
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService, RecognitionResult
from utils.camera import DEFAULT_CAPTURE_HEIGHT, DEFAULT_CAPTURE_WIDTH, open_video_source
from utils.device import select_device
from utils.guidance import run_guidance_phase
from utils.services import create_services

DEFAULT_WEIGHTS = ROOT / "MobileFaceNet" / "Weights" / "MobileFace_Net"
DEFAULT_FACEBANK = ROOT / "facebank"
DEFAULT_SPOOF_WEIGHTS = ROOT / "DeePixBis" / "Weights" / "DeePixBiS.pth"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

WINDOW_WIDTH_LIMIT = 1920
WINDOW_HEIGHT_LIMIT = 1080
GUIDANCE_BOX_SCALE = 0.40
GUIDANCE_MIN_SIDE = 224


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
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to an image/video file.",
    )
    parser.add_argument(
        "-th",
        "--identity-thr",
        type=float,
        default=70.0,
        dest="identity_thr",
        help="Identity acceptance threshold on the 0-100 score scale.",
    )
    parser.add_argument(
        "--spoof-thr",
        type=float,
        default=0.9,
        help="Minimum DeePixBiS score to label a face as real.",
    )
    parser.add_argument(
        "--detector-thr",
        type=float,
        default=0.7,
        help="Minimum BlazeFace confidence required to keep detections.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=0,
        help="Preferred camera capture width in pixels (0 = leave camera default).",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=0,
        help="Preferred camera capture height in pixels (0 = leave camera default).",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to MobileFaceNet weights.",
    )
    parser.add_argument(
        "--facebank",
        default=str(DEFAULT_FACEBANK),
        help="Path to facebank directory.",
    )
    parser.add_argument(
        "--spoof-weights",
        default=str(DEFAULT_SPOOF_WEIGHTS),
        help="Path to DeePixBiS weights.",
    )
    parser.add_argument(
        "-u",
        "--update-facebank",
        "--update",
        action="store_true",
        dest="update_facebank",
        help="Rebuild the facebank before running.",
    )
    parser.add_argument(
        "-tta",
        "--tta",
        action="store_true",
        help="Enable flip test-time augmentation for recognition embeddings.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Hide identity confidence values in the overlay.",
    )
    parser.add_argument(
        "--no-spoof-score",
        action="store_true",
        help="Hide DeePixBiS scores in the overlay.",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Hide landmark dots in the overlay.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save annotated output (image or video).",
    )
    parser.add_argument(
        "--disable-spoof",
        action="store_true",
        help="Skip DeePixBiS anti-spoofing.",
    )
    parser.add_argument(
        "--guidance",
        action="store_true",
        help="Run the face alignment guide before processing (camera sources only).",
    )
    parser.add_argument(
        "--guidance-box-size",
        type=int,
        default=0,
        help="Edge length in pixels for the guidance square (0 = auto).",
    )
    parser.add_argument(
        "--guidance-center-tolerance",
        type=float,
        default=0.25,
        help="Fraction of the square half-side tolerated for centering during guidance.",
    )
    parser.add_argument(
        "--guidance-size-tolerance",
        type=float,
        default=0.15,
        help="Fractional tolerance for face size vs square during guidance.",
    )
    parser.add_argument(
        "--guidance-rotation-thr",
        type=float,
        default=7.0,
        help="Maximum allowed head tilt in degrees during guidance.",
    )
    parser.add_argument(
        "--guidance-hold-frames",
        type=int,
        default=15,
        help="Number of consecutive aligned frames before guidance succeeds.",
    )
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


def build_services(args: argparse.Namespace, device):
    return create_services(args, device)


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
            identity_score = float(rec.confidence)
            is_recognized = bool(rec.is_recognized)

        spoof_score = None
        is_real = None
        if idx in spoof_lookup:
            spoof_score = float(spoof_lookup[idx])
            is_real = bool(spoof_score >= spoof_threshold)

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
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for obs in observations:
        detection = obs.detection
        x1, y1, x2, y2 = detection.as_int_bbox()
        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        y2 = int(np.clip(y2, 0, height - 1))

        if obs.is_real is False:
            color = (0, 0, 255)
        elif obs.is_real is True:
            color = (0, 200, 0)
        elif obs.is_recognized:
            color = (0, 215, 255)
        else:
            color = (255, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if show_landmarks and detection.keypoints is not None:
            for point in detection.keypoints:
                px, py = int(round(point[0])), int(round(point[1]))
                if 0 <= px < width and 0 <= py < height:
                    cv2.circle(annotated, (px, py), 2, color, -1, lineType=cv2.LINE_AA)

        lines: List[str] = []
        identity_label = obs.identity or "Unknown"
        if show_identity_score and obs.identity_score is not None:
            identity_label += f" ({obs.identity_score:.0f})"
        lines.append(identity_label)

        if show_spoof_score and obs.spoof_score is not None:
            if obs.is_real is True:
                verdict = "Real"
            elif obs.is_real is False:
                verdict = "Fake"
            else:
                verdict = "Unknown"
            lines.append(f"{verdict}: {obs.spoof_score:.2f}")

        text_x = max(0, x1 + 4)
        text_y = max(20, y1 + 18)
        for line in lines:
            (text_w, text_h), baseline = cv2.getTextSize(line, font, 0.55, 2)
            box_top = max(0, text_y - text_h - baseline)
            box_bottom = min(height, text_y + baseline)
            box_right = min(width, text_x + text_w + 4)
            box_left = max(0, text_x - 2)
            cv2.rectangle(annotated, (box_left, box_top), (box_right, box_bottom), (0, 0, 0), -1)
            cv2.putText(annotated, line, (text_x, text_y), font, 0.55, color, 2, lineType=cv2.LINE_AA)
            text_y = min(height - 5, text_y + text_h + 6)

    return annotated


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
    *,
    mode: str,
) -> None:
    detector, recogniser, spoiler = services
    frame_size = None
    if mode == "camera" and args.camera_width > 0 and args.camera_height > 0:
        frame_size = (args.camera_width, args.camera_height)

    cap = open_video_source(source, frame_size=frame_size)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    window_name = "Face Recognition System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if args.guidance:
        if mode != "camera":
            print("Face guidance is only available for live camera sources; skipping guidance step.")
        else:
            guidance_success = run_guidance_phase(
                cap,
                detector,
                args,
                window_name,
                allow_resize=True,
                min_side=GUIDANCE_MIN_SIDE,
                box_scale=GUIDANCE_BOX_SCALE,
                window_limits=(WINDOW_WIDTH_LIMIT, WINDOW_HEIGHT_LIMIT),
            )
            if not guidance_success:
                cap.release()
                cv2.destroyWindow(window_name)
                print("Face alignment was not confirmed. Exiting without running the pipeline.")
                return

    writer = None
    writer_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not writer_fps or writer_fps <= 0:
        writer_fps = 30.0

    display_width = WINDOW_WIDTH_LIMIT
    display_height = WINDOW_HEIGHT_LIMIT
    geometry_reported = False

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if not geometry_reported:
            frame_height, frame_width = frame.shape[:2]
            resolved_width = frame_width if frame_width > 0 else (frame_size[0] if frame_size else DEFAULT_CAPTURE_WIDTH)
            resolved_height = frame_height if frame_height > 0 else (frame_size[1] if frame_size else DEFAULT_CAPTURE_HEIGHT)
            fps_prop = cap.get(cv2.CAP_PROP_FPS)
            actual_fps = float(fps_prop) if fps_prop and fps_prop > 0 else 0.0
            if actual_fps > 0:
                print(f"Camera opened at {resolved_width}x{resolved_height} @ {actual_fps:.2f} FPS")
            else:
                print(f"Camera opened at {resolved_width}x{resolved_height}")
            width_scale = WINDOW_WIDTH_LIMIT / max(1, resolved_width)
            height_scale = WINDOW_HEIGHT_LIMIT / max(1, resolved_height)
            scale = min(width_scale, height_scale, 1.0)
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            display_width = max(1, int(round(resolved_width * scale)))
            display_height = max(1, int(round(resolved_height * scale)))
            cv2.resizeWindow(window_name, display_width, display_height)
            geometry_reported = True

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

        if args.output:
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    args.output,
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    writer_fps,
                    (width, height),
                    True,
                )
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    print(f"Warning: unable to open video writer for {args.output}")
            if writer is not None:
                writer.write(annotated)

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    mode, source = classify_source(args.source)
    device = select_device(args.device)
    services = build_services(args, device)

    if mode == "image":
        run_image_mode(services, Path(source), args)
    else:
        video_source = str(source) if mode == "video" else source
        run_video_mode(services, video_source, args, mode=mode)


if __name__ == "__main__":
    main()
