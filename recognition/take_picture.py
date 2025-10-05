#!/usr/bin/env python3
"""Capture aligned face images from webcam using BlazeFace."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.blazeface_detector import BlazeFaceDetector

parser = argparse.ArgumentParser(description="capture face images for the facebank")
parser.add_argument("--name", "-n", default="unknown", help="name of the recording person")
parser.add_argument("--detector-thr", type=float, default=0.6,
                    help="minimum BlazeFace confidence to keep a detection")
parser.add_argument("--camera", default=0, help="VideoCapture index or path")
args = parser.parse_args()

save_dir = ROOT / "dataset" / "facebank" / args.name
save_dir.mkdir(parents=True, exist_ok=True)

detector = BlazeFaceDetector(score_threshold=args.detector_thr)
cap_source = int(args.camera) if isinstance(args.camera, str) and args.camera.isdigit() else args.camera
cap = cv2.VideoCapture(cap_source)
if not cap.isOpened():
    raise RuntimeError(f"Unable to open source: {args.camera}")

print("Press 't' to capture a face crop, 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failure")
        break

    preview = frame.copy()
    for det in detector.detect(frame):
        x1, y1, x2, y2 = det.as_int_bbox()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(preview, "Press t to save, q to quit", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("BlazeFace Capture", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("t"):
        detections = detector.detect(frame)
        if not detections:
            print("No face captured")
            continue
        best_det = max(detections, key=lambda d: d.score)
        face = detector.align_face(frame, best_det)
        if face is None:
            print("Failed to align face")
            continue
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
        cv2.imwrite(str(save_dir / filename), face)
        print(f"Saved {save_dir / filename}")

cap.release()
cv2.destroyAllWindows()


