#!/usr/bin/env python3
"""Face verification workflow with alignment guidance and short evaluation."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BlazeFace import BlazeFaceService
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService
from utils import (
    FaceObservation,
    DEFAULT_CAMERA_CONFIG,
    CameraConfig,
    aggregate_observations,
    append_attendance_log,
    compose_final_display,
    draw_guidance_overlay,
    evaluate_alignment,
    evaluate_frame,
    resolve_device,
    select_best_detection,
    open_capture,
)

DEFAULT_WEIGHTS = ROOT / "MobileFaceNet" / "Weights" / "MobileFace_Net"
DEFAULT_FACEBANK = ROOT / "facebank"
DEFAULT_SPOOF_WEIGHTS = ROOT / "DeePixBis" / "Weights" / "DeePixBiS.pth"
DEFAULT_ATTENDANCE_LOG = ROOT / "attendance_results.jsonl"
GUIDANCE_BOX_SCALE = 0.40  # base fraction of the shortest frame side for typical close-up
WINDOW_WIDTH_LIMIT = 960
WINDOW_HEIGHT_LIMIT = 720
# Minimum effective face sizes (px) across the pipeline
BLAZEFACE_MIN_FACE = 64
MOBILEFACENET_INPUT = 112
DEEPIX_TARGET_SIZE = 224
GUIDANCE_MIN_SIDE = max(BLAZEFACE_MIN_FACE, MOBILEFACENET_INPUT, DEEPIX_TARGET_SIZE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align face, verify identity, and log attendance.")
    parser.add_argument("--source", default="0",
                        help="camera index (e.g. 0) or path to a video file")
    parser.add_argument("--device", default="auto",
                        help="torch device to run on (e.g. auto, cpu, cuda, cuda:0)")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="path to MobileFaceNet weights")
    parser.add_argument("--facebank", default=str(DEFAULT_FACEBANK),
                        help="path to facebank directory")
    parser.add_argument("--update-facebank", action="store_true",
                        help="rebuild the facebank before running verification")
    parser.add_argument("--tta", action="store_true",
                        help="enable flip test-time augmentation for recognition embeddings")
    parser.add_argument("--identity-thr", type=float, default=70.0,
                        help="identity acceptance threshold on the 0-100 score scale")
    parser.add_argument("--spoof-weights", default=str(DEFAULT_SPOOF_WEIGHTS),
                        help="path to DeePixBiS weights")
    parser.add_argument("--disable-spoof", action="store_true",
                        help="skip DeePixBiS anti-spoofing during verification")
    parser.add_argument("--detector-thr", type=float, default=0.7,
                        help="minimum BlazeFace confidence required to keep detections")
    parser.add_argument("--spoof-thr", type=float, default=0.9,
                        help="minimum DeePixBiS score to label a face as real")
    parser.add_argument("--evaluation-duration", type=float, default=1.0,
                        help="duration (seconds) to capture frames for verification")
    parser.add_argument("--attendance-log", default=str(DEFAULT_ATTENDANCE_LOG),
                        help="path to append attendance results (JSON lines)")

    parser.add_argument("--guidance-box-size", type=int, default=0,
                        help="edge length in pixels for the guidance square (0 = auto)")
    parser.add_argument("--guidance-center-tolerance", type=float, default=0.25,
                        help="fraction of the square half-side tolerated for centering during guidance")
    parser.add_argument("--guidance-size-tolerance", type=float, default=0.15,
                        help="fractional tolerance for face size vs square during guidance")
    parser.add_argument("--guidance-rotation-thr", type=float, default=7.0,
                        help="maximum allowed head tilt in degrees during guidance")
    parser.add_argument("--guidance-hold-frames", type=int, default=15,
                        help="number of consecutive aligned frames before verification begins")

    cam_defaults = DEFAULT_CAMERA_CONFIG
    parser.add_argument("--camera-backend", choices=["auto", "opencv", "gstreamer"],
                        default=cam_defaults.backend,
                        help="backend to use when opening a camera index")
    parser.add_argument("--camera-width", type=int, default=cam_defaults.width,
                        help="desired capture width for camera indices")
    parser.add_argument("--camera-height", type=int, default=cam_defaults.height,
                        help="desired capture height for camera indices")
    parser.add_argument("--camera-fps", type=float, default=cam_defaults.fps,
                        help="desired capture FPS for camera indices")
    parser.add_argument("--camera-flip", type=int, default=cam_defaults.flip_method,
                        help="nvargus flip-method to use with the Jetson GStreamer pipeline")
    parser.add_argument("--camera-sensor-mode", type=int,
                        default=cam_defaults.sensor_mode if cam_defaults.sensor_mode is not None else 0,
                        help="nvargus sensor-mode for Jetson CSI cameras (0 = auto)")
    parser.add_argument("--gstreamer-pipeline", default=cam_defaults.gstreamer_pipeline,
                        help="custom GStreamer pipeline string (overrides the auto-generated nvargus pipeline)")
    return parser.parse_args()


def resolve_execution_device(device_name: str) -> torch.device:
    name = (device_name or "auto").strip().lower()
    if not name or name == "auto":
        for candidate in ("cuda", "cuda:0", "cpu"):
            try:
                return resolve_device(candidate)
            except (RuntimeError, ValueError):
                continue
        raise RuntimeError("Unable to resolve an appropriate torch device.")

    try:
        return resolve_device(name)
    except (RuntimeError, ValueError) as ex:
        if name != "cpu":
            print(f"Warning: {ex}. Falling back to CPU.")
            return resolve_device("cpu")
        raise


def build_services(args: argparse.Namespace, device: torch.device):
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
    spoiler = None
    if not args.disable_spoof:
        spoiler = DeePixBiSService(weights_path=Path(args.spoof_weights), device=device)
    return detector, recogniser, spoiler


def run_guidance_phase(
    capture: cv2.VideoCapture,
    detector: BlazeFaceService,
    args: argparse.Namespace,
    window_name: str,
    *,
    allow_resize: bool,
) -> bool:
    consecutive_good = 0
    window_resized = False
    while True:
        ok, frame = capture.read()
        if not ok:
            return False

        height, width = frame.shape[:2]
        min_dim = min(height, width)
        min_required_side = GUIDANCE_MIN_SIDE
        auto_side = max(min_required_side, int(min_dim * GUIDANCE_BOX_SCALE))
        if args.guidance_box_size > 0:
            side = max(min_required_side, args.guidance_box_size)
        else:
            side = auto_side
        max_side = max(min_required_side, min_dim - 20)
        side = min(side, max_side)
        if side % 2 != 0:
            side = max(min_required_side, side - 1)
        half_side = max(20, side // 2)
        box_center = (width // 2, height // 2)

        if allow_resize and not window_resized:
            target_w = min(width, WINDOW_WIDTH_LIMIT)
            target_h = min(height, WINDOW_HEIGHT_LIMIT)
            cv2.resizeWindow(window_name, target_w, target_h)
            window_resized = True

        detections = detector.detect(frame)
        detection = select_best_detection(detections) if detections else None
        assessment = None
        if detection is not None:
            assessment = evaluate_alignment(
                detection,
                box_center,
                half_side,
                center_tolerance_ratio=args.guidance_center_tolerance,
                size_tolerance_ratio=args.guidance_size_tolerance,
                rotation_threshold=args.guidance_rotation_thr,
            )
            consecutive_good = consecutive_good + 1 if assessment.is_aligned else 0
        else:
            consecutive_good = 0

        display = draw_guidance_overlay(
            frame,
            box_center=box_center,
            half_side=half_side,
            detection=detection,
            assessment=assessment,
            show_messages=False,
        )
        instruction_lines: List[Tuple[str, Tuple[int, int, int], float]] = []
        instruction_lines.append(("Align your face within the square", (255, 255, 255), 0.8))

        if detection is None:
            instruction_lines.append(("Face not detected, center yourself", (0, 0, 255), 0.7))
            instruction_lines.append(("Look toward the camera", (0, 0, 255), 0.7))
        elif assessment is not None:
            if assessment.is_aligned:
                instruction_lines.append(("Hold steady to confirm", (0, 255, 0), 0.75))
                instruction_lines.append((f"Progress: {consecutive_good}/{args.guidance_hold_frames}", (0, 255, 0), 0.7))
            else:
                messages = assessment.messages or ["Adjust your position"]
                for msg in messages:
                    instruction_lines.append((msg, (0, 165, 255), 0.7))
                instruction_lines.append((f"Progress: {consecutive_good}/{args.guidance_hold_frames}", (0, 165, 255), 0.7))
        else:
            instruction_lines.append(("Detecting face...", (0, 165, 255), 0.7))

        base_y = 40
        line_spacing = 28
        for text, color, scale in instruction_lines:
            cv2.putText(display, text, (20, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            base_y += int(line_spacing * scale / 0.7)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False

        if consecutive_good >= args.guidance_hold_frames:
            return True


def run_verification_phase(
    capture: cv2.VideoCapture,
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_service: Optional[DeePixBiSService],
    args: argparse.Namespace,
    window_name: str,
) -> Tuple[List[FaceObservation], Optional[np.ndarray], float]:
    observations: List[FaceObservation] = []
    last_frame: Optional[np.ndarray] = None
    duration_limit = max(0.0, min(args.evaluation_duration, 5.0))
    start_time = time.perf_counter()

    while True:
        now = time.perf_counter()
        if now - start_time >= duration_limit:
            break
        ok, frame = capture.read()
        if not ok:
            break
        last_frame = frame.copy()

        observation = evaluate_frame(frame, detector, recogniser, spoof_service, args.spoof_thr)
        if observation is not None:
            observations.append(observation)

        overlay = frame.copy()
        elapsed = min(time.perf_counter() - start_time, duration_limit)
        identity_scores = [obs.identity_score for obs in observations if obs.identity_score is not None]
        avg_identity = sum(identity_scores) / len(identity_scores) if identity_scores else None
        spoof_scores = [obs.spoof_score for obs in observations if obs.spoof_score is not None]
        avg_spoof = sum(spoof_scores) / len(spoof_scores) if spoof_scores else None

        cv2.putText(overlay, "Capturing verification frames...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay, f"Elapsed: {elapsed:.2f}s / {duration_limit:.2f}s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(overlay, f"Number of frames: {len(observations)}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if avg_identity is not None:
            cv2.putText(overlay, f"Avg identity: {avg_identity:.1f}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        if avg_spoof is not None:
            cv2.putText(overlay, f"Avg spoof: {avg_spoof:.2f}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    total_time = min(time.perf_counter() - start_time, duration_limit)
    return observations, last_frame, total_time


def main() -> None:
    args = parse_args()
    raw_source = args.source
    source = int(raw_source) if isinstance(raw_source, str) and raw_source.isdigit() else raw_source

    device = resolve_execution_device(args.device)

    sensor_mode = None if args.camera_sensor_mode in (None, 0) else args.camera_sensor_mode
    camera_config = CameraConfig(
        backend=args.camera_backend,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        flip_method=args.camera_flip,
        sensor_mode=sensor_mode,
        gstreamer_pipeline=args.gstreamer_pipeline,
    )

    capture = open_capture(source, config=camera_config)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {raw_source}")

    detector, recogniser, spoof_service = build_services(args, device)

    window_name = "Face Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_w = min(WINDOW_WIDTH_LIMIT, max(320, args.camera_width))
    window_h = min(WINDOW_HEIGHT_LIMIT, max(240, args.camera_height))
    cv2.resizeWindow(window_name, window_w, window_h)

    attending = True
    auto_resize_applied = False
    while attending:
        if not run_guidance_phase(
            capture,
            detector,
            args,
            window_name,
            allow_resize=not auto_resize_applied,
        ):
            break
        auto_resize_applied = True

        observations, last_frame, duration = run_verification_phase(
            capture,
            detector,
            recogniser,
            spoof_service,
            args,
            window_name,
        )

        summary = aggregate_observations(observations, spoof_threshold=args.spoof_thr)
        summary["capture_duration"] = duration
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "source": raw_source,
            "recognized": summary["recognized"],
            "identity": summary["identity"],
            "avg_identity_score": summary["avg_identity_score"],
            "avg_spoof_score": summary["avg_spoof_score"],
            "is_real": summary["is_real"],
            "accepted": summary["accepted"],
            "frames_with_detections": summary["frames_with_detections"],
            "capture_duration": summary.get("capture_duration"),
        }
        append_attendance_log(Path(args.attendance_log), log_entry)

        if last_frame is None:
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.camera_height
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.camera_width
            last_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        last_observation = observations[-1] if observations else None
        final_display = compose_final_display(
            last_frame,
            last_observation,
            summary,
            show_spoof_score=spoof_service is not None,
        )

        cv2.imshow(window_name, final_display)
        status_text = "ACCESS GRANTED" if summary["accepted"] else "ACCESS DENIED"
        print(f"[{timestamp}] {status_text}: {summary['identity']}")
        print("Press SPACE to welcome the next person or ESC to close.")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(" "):
                break
            if key in (27, ord("q")):
                attending = False
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
