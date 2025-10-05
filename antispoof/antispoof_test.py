#!/usr/bin/env python3
"""Webcam DeePixBiS demo using BlazeFace crops."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from antispoof.antispoof_model import DeePixBiS
from detection.blazeface_detector import BlazeFaceDetector

WEIGHTS_PATH = ROOT /"antispoof" / "weights" / "DeePixBiS.pth"
print(WEIGHTS_PATH)

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def main() -> None:
    model = DeePixBiS()
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"DeePixBiS weights not found at {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    detector = BlazeFaceDetector(score_threshold=0.6)
    camera = cv2.VideoCapture(0)
    smoothing = 0.9
    avg_latency = None
    avg_fps = None


    while True:
        frame_start = time.perf_counter()
        ok, frame = camera.read()
        if not ok:
            print("Camera read failure")
            break

        detections = detector.detect(frame)
        for det in detections:
            crop = detector.crop_face(frame, det, expand=0.15, output_size=(224, 224))
            if crop is None:
                continue

            tensor = _transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
            mask, _ = model.forward(tensor)
            score = float(mask.mean().item())

            x1, y1, x2, y2 = det.as_int_bbox()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = "Real" if score >= 0.7 else "Fake"
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y2 + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        elapsed = time.perf_counter() - frame_start
        inst_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        if avg_latency is None:
            avg_latency = elapsed
            avg_fps = inst_fps
        else:
            avg_latency = smoothing * avg_latency + (1 - smoothing) * elapsed
            avg_fps = smoothing * avg_fps + (1 - smoothing) * inst_fps

        cv2.putText(frame, f"Latency: {avg_latency * 1000:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        print(f"Latency: {elapsed * 1000:.1f} ms | FPS: {inst_fps:.1f}", end="\r")

        cv2.imshow("DeePixBiS Anti-Spoofing", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
