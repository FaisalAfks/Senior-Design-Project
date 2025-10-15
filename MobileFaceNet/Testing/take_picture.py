#!/usr/bin/env python3
"""Capture aligned face images from webcam using BlazeFace."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BlazeFace import BlazeFaceService

parser = argparse.ArgumentParser(description="capture face images for the facebank")
parser.add_argument("--name", "-n", default="unknown", help="name of the recording person")
parser.add_argument("--detector-thr", type=float, default=0.6, help="minimum BlazeFace confidence to keep a detection")
parser.add_argument("--camera", default=0, help="VideoCapture index or path")
args = parser.parse_args()

save_dir = PROJECT_ROOT / "Pictures" / args.name
save_dir.mkdir(parents=True, exist_ok=True)


def _next_facebank_index(directory: Path) -> int:
    prefix = "facebank_"
    max_index = 0
    for file in directory.glob(f"{prefix}*"):
        if not file.is_file():
            continue
        parts = file.stem.split("_")
        if not parts:
            continue
        try:
            idx = int(parts[-1])
        except ValueError:
            continue
        if idx > max_index:
            max_index = idx
    return max_index + 1

service = BlazeFaceService(score_threshold=args.detector_thr)
cap_source = int(args.camera) if isinstance(args.camera, str) and args.camera.isdigit() else args.camera
cap = cv2.VideoCapture(cap_source)
if not cap.isOpened():
    raise RuntimeError(f"Unable to open source: {args.camera}")

print("Press 't' to capture a face crop, 'q' to quit.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failure")
        break
    
    preview = frame.copy()
    for det in service.detect(frame):
        x1, y1, x2, y2 = det.as_int_bbox()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(preview, "Press t to save, q to quit", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("BlazeFace Capture", preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("t"):
        detections = service.detect(frame)
        if not detections:
            print("No face captured")
            continue
        best_det = max(detections, key=lambda d: d.score)
        face = service.detector.align_face(frame, best_det)
        if face is None:
            print("Failed to align face")
            continue
        next_idx = _next_facebank_index(save_dir)
        filename = f"facebank_{next_idx:02d}.jpg"
        cv2.imwrite(str(save_dir / filename), face)
        print(f"Saved {save_dir / filename}")

cap.release()
cv2.destroyAllWindows()
