"""Webcam/video test for BlazeFace."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.blazeface import BlazeFace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BlazeFace camera test")
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parent / "blazeface_assets" / "blazeface.pth"),
                        help="Path to the BlazeFace weights (.pth)")
    parser.add_argument("--anchors", default=str(Path(__file__).resolve().parent / "blazeface_assets" / "anchors.npy"),
                        help="Path to the BlazeFace anchors (.npy)")
    parser.add_argument("--back_model", action="store_true",
                        help="Use the back-facing BlazeFace model (256x256)")
    parser.add_argument("--source", default="0",
                        help="Camera index or path to a video file")
    parser.add_argument("--score-thr", type=float, default=0.5,
                        help="score threshold for drawing detections")
    parser.add_argument("--flip", action="store_true",
                        help="Flip preview horizontally (selfie view)")
    parser.add_argument("--max-fps", type=float, default=0.0,
                        help="Limit preview FPS (0 = unlimited)")
    parser.add_argument("--window", default="BlazeFace Cam Test",
                        help="Window title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = BlazeFace(back_model=args.back_model)
    model.load_weights(args.weights)
    model.load_anchors(args.anchors)
    model.min_score_thresh = args.score_thr
    model.eval()

    side = 256 if args.back_model else 128
    _ = model(torch.randn((1, 3, side, side)))

    src = args.source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    wait_ms = 1 if args.max_fps <= 0 else max(1, int(1000.0 / args.max_fps))
    last = time.perf_counter()
    fps_ema = 0.0
    latency_ema = 0.0
    screenshot_id = 0

    while True:
        frame_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("End of stream or camera read failure.")
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        square, (left, top, side_px, _) = _center_square_crop(frame)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)

        faces = model.predict_on_image(resized)
        drawn = draw_detections(square, faces, args.score_thr)

        elapsed = time.perf_counter() - frame_start
        latency_ms = elapsed * 1000.0
        latency_ema = latency_ms if latency_ema == 0 else 0.9 * latency_ema + 0.1 * latency_ms
        now = frame_start + elapsed
        fps = 1.0 / (now - last) if now != last else 0.0
        fps_ema = fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * fps
        last = now

        cv2.putText(drawn, f"FPS: {fps_ema:.1f}  Latency: {latency_ema:.1f} ms", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
        cv2.imshow(args.window, drawn)

        key = cv2.waitKey(wait_ms) & 0xFFFF
        if key in (27, ord('q')):
            break
        if key in (ord('s'), ord('S')):
            name = Path(__file__).resolve().parent / f"snap_{screenshot_id:03d}.png"
            cv2.imwrite(str(name), drawn)
            print(f"Saved {name}")
            screenshot_id += 1

    cap.release()
    cv2.destroyAllWindows()


def _center_square_crop(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return frame[top:top + side, left:left + side], (left, top, side, side)


def draw_detections(frame_sq, faces, score_thr=0.5):
    if faces is None:
        return frame_sq
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    if faces.ndim == 1:
        faces = faces[None, :]

    img = frame_sq.copy()
    h, w = img.shape[:2]
    for det in faces:
        if det.shape[0] != 17 or det[16] < score_thr:
            continue
        ymin, xmin, ymax, xmax = det[:4]
        x1, y1, x2, y2 = map(int, [xmin * w, ymin * h, xmax * w, ymax * h])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det[16]:.2f}"
        cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        keypoints = det[4:16].reshape(6, 2)
        for kx, ky in keypoints:
            cv2.circle(img, (int(kx * w), int(ky * h)), 2, (255, 0, 0), -1)
    return img


if __name__ == "__main__":
    main()

