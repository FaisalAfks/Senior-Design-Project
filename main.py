#!/usr/bin/env python3
"""Unified face pipeline (detection + recognition + anti-spoofing)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_pipeline import FacePipeline

DEFAULT_WEIGHTS = ROOT / "recognition" / "weights" / "MobileFace_Net"
DEFAULT_FACEBANK = ROOT / "dataset" / "facebank"
DEFAULT_SPOOF_WEIGHTS = ROOT / "antispoof" / "weights" / "DeePixBiS.pth"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face Guardian unified demo")
    parser.add_argument("--source", default="0",
                        help="camera index (e.g. 0) or path to an image/video")
    parser.add_argument("-th", "--threshold", default=60.0, type=float,
                        help="identity acceptance threshold on the 0-100 score scale")
    parser.add_argument("--spoof-thr", default=0.7, type=float,
                        help="minimum DeePixBiS score to label a face as real")
    parser.add_argument("--detector-thr", default=0.6, type=float,
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


def build_pipeline(args: argparse.Namespace) -> FacePipeline:
    spoof_weights = None if args.disable_spoof else Path(args.spoof_weights)
    return FacePipeline(
        recognition_weights=Path(args.weights),
        facebank_dir=Path(args.facebank),
        spoof_weights=spoof_weights,
        detection_threshold=args.detector_thr,
        recognition_threshold=args.threshold,
        spoof_threshold=args.spoof_thr,
        tta=args.tta,
        refresh_facebank=args.update,
    )


def log_results(observations, prefix: str = "") -> None:
    if not observations:
        print(prefix + "No faces detected")
        return
    for idx, obs in enumerate(observations, 1):
        identity = obs.identity or "Unknown"
        identity_score = f"{obs.identity_score:.0f}" if obs.identity_score is not None else "--"
        spoof_score = f"{obs.spoof_score:.2f}" if obs.spoof_score is not None else "--"
        verdict = "Real" if obs.is_real else "Fake" if obs.is_real is not None else "Unknown"
        print(f"{prefix}Face #{idx}: {identity} (score {identity_score}) | {verdict} ({spoof_score})")


def run_image_mode(pipeline: FacePipeline, path: Path, args: argparse.Namespace) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    observations = pipeline.process(image)
    log_results(observations)
    annotated = pipeline.annotate(
        image,
        observations,
        show_landmarks=not args.no_landmarks,
        show_identity_score=not args.no_score,
        show_spoof_score=not args.no_spoof_score,
    )
    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"Annotated image saved to {args.output}")
    cv2.imshow("Face Guardian", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_mode(pipeline: FacePipeline, source, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
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
        observations = pipeline.process(frame)
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        log_results(observations, prefix=f"[{frame_index}] ")

        annotated = pipeline.annotate(
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
        cv2.imshow("Face Guardian", annotated)
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
    pipeline = build_pipeline(args)

    if mode == "image":
        run_image_mode(pipeline, Path(source), args)
    else:
        video_source = str(source) if mode == "video" else source
        run_video_mode(pipeline, video_source, args)


if __name__ == "__main__":
    main()


