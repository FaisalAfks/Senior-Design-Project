#!/usr/bin/env python3
"""Capture a single frame through detector/recogniser/spoofer and save debug crops."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from BlazeFace import BlazeFaceService, Detection
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService
from main import DEFAULT_FACEBANK, DEFAULT_SPOOF_WEIGHTS, DEFAULT_WEIGHTS
from utils.device import select_device

ANNOTATION_COLOR = (0, 255, 0)
KEYPOINT_COLOR = (255, 0, 0)
SPOOF_BOX_COLOR = (0, 165, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pipeline on a single frame and export intermediate crops.")
    parser.add_argument("frame", type=Path, help="Image to process (BGR readable by OpenCV).")
    parser.add_argument("--output-dir", type=Path, default=Path("debug_snapshots"), help="Directory to store exported visuals.")
    parser.add_argument("--device", default="cpu", help="Torch device string (cpu, cuda, cuda:0, ...).")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to MobileFaceNet weights.")
    parser.add_argument("--facebank", type=Path, default=DEFAULT_FACEBANK, help="Path to facebank directory.")
    parser.add_argument("--spoof-weights", type=Path, default=DEFAULT_SPOOF_WEIGHTS, help="Path to DeePixBiS weights.")
    parser.add_argument("--identity-thr", type=float, default=0.7, help="Identity acceptance threshold (0-1).")
    parser.add_argument("--spoof-thr", type=float, default=0.9, help="Spoof acceptance threshold (0-1).")
    parser.add_argument("--detector-thr", type=float, default=0.7, help="Minimum BlazeFace confidence to keep a detection.")
    parser.add_argument("--disable-spoof", action="store_true", help="Skip anti-spoof evaluation.")
    return parser.parse_args()


def annotate_detection(image: np.ndarray, detection: Detection) -> np.ndarray:
    annotated = image.copy()
    x1, y1, x2, y2 = detection.as_int_bbox()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), ANNOTATION_COLOR, 2)
    cv2.putText(
        annotated,
        f"score={detection.score:.2f}",
        (x1, max(10, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        ANNOTATION_COLOR,
        2,
    )
    for (px, py) in detection.keypoints.astype(int):
        cv2.circle(annotated, (int(px), int(py)), 3, KEYPOINT_COLOR, -1)
    return annotated


def annotate_spoof_crop(image: np.ndarray, detection: Detection, *, expand: float = 0.15) -> np.ndarray:
    annotated = image.copy()
    x1, y1, x2, y2 = detection.bbox
    width = x2 - x1
    height = y2 - y1
    expand_x = width * expand
    expand_y = height * expand
    xmin = max(0, int(np.floor(x1 - expand_x)))
    ymin = max(0, int(np.floor(y1 - expand_y)))
    xmax = min(image.shape[1], int(np.ceil(x2 + expand_x)))
    ymax = min(image.shape[0], int(np.ceil(y2 + expand_y)))
    cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), SPOOF_BOX_COLOR, 2)
    return annotated


def main() -> None:
    args = parse_args()
    if not args.frame.exists():
        raise FileNotFoundError(f"Frame not found: {args.frame}")

    frame = cv2.imread(str(args.frame))
    if frame is None or frame.size == 0:
        raise RuntimeError(f"Unable to read image: {args.frame}")

    device = select_device(args.device)
    detector = BlazeFaceService(score_threshold=args.detector_thr, device=device)
    recogniser = MobileFaceNetService(
        weights_path=args.weights,
        facebank_dir=args.facebank,
        detector=detector.detector,
        device=device,
        recognition_threshold=args.identity_thr,
        tta=False,
        refresh_facebank=False,
    )
    spoof_service: Optional[DeePixBiSService] = None
    if not args.disable_spoof:
        spoof_service = DeePixBiSService(weights_path=args.spoof_weights, device=device)

    detections = detector.detect(frame)
    if not detections:
        print("No detections found in the provided frame.")
        return

    best_det = max(detections, key=lambda det: det.score * max(det.area(), 1.0))
    aligned = detector.detector.align_face(frame, best_det)
    spoof_crop = detector.detector.crop_face(frame, best_det, expand=0.15, output_size=(224, 224))

    recognition_score = None
    identity_name = "Unknown"
    is_recognized = False
    if aligned is not None:
        rec_results = recogniser.recognise_faces([aligned])
        if rec_results:
            rec = rec_results[0]
            recognition_score = float(rec.confidence)
            identity_name = rec.name
            is_recognized = rec.is_recognized

    spoof_score = None
    if spoof_service is not None and spoof_crop is not None:
        spoof_score = float(spoof_service.predict_scores([spoof_crop])[0])

    accepted = is_recognized and (spoof_score is None or spoof_score >= args.spoof_thr)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / args.frame.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_detection = annotate_detection(frame, best_det)
    annotated_spoof = annotate_spoof_crop(annotated_detection, best_det)

    cv2.imwrite(str(output_dir / f"{timestamp}_raw.png"), frame)
    cv2.imwrite(str(output_dir / f"{timestamp}_detector.png"), annotated_detection)
    cv2.imwrite(str(output_dir / f"{timestamp}_spoof_region.png"), annotated_spoof)
    if aligned is not None:
        cv2.imwrite(str(output_dir / f"{timestamp}_recogniser_input.png"), aligned)
    if spoof_crop is not None:
        cv2.imwrite(str(output_dir / f"{timestamp}_spoof_input.png"), spoof_crop)

    summary: Dict[str, object] = {
        "timestamp": timestamp,
        "frame": str(args.frame),
        "detector_score": best_det.score,
        "identity": identity_name,
        "identity_score": recognition_score,
        "spoof_score": spoof_score,
        "recognized": is_recognized,
        "accepted": accepted,
        "identity_threshold": args.identity_thr,
        "spoof_threshold": args.spoof_thr,
        "detections": len(detections),
    }
    with (output_dir / f"{timestamp}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved debug artefacts to {output_dir}")
    if recognition_score is not None:
        print(f"Identity: {identity_name} | score={recognition_score * 100.0:.1f}% | recognized={is_recognized}")
    else:
        print("Identity: no embedding produced.")
    if spoof_score is not None:
        verdict = "REAL" if spoof_score >= args.spoof_thr else "SPOOF"
        print(f"Spoof score: {spoof_score * 100.0:.1f}% ({verdict})")
    elif spoof_service is None:
        print("Spoofing disabled by flag.")
    else:
        print("Spoof score: no crop available.")
    print(f"Final access decision: {'ACCEPTED' if accepted else 'REJECTED'}")


if __name__ == "__main__":
    main()

