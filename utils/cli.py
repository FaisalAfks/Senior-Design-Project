"""Command-line argument builders for project entry points."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_main_args(
    *,
    default_weights: Path,
    default_facebank: Path,
    default_spoof_weights: Path,
    default_attendance_log: Path,
    default_power_log: Path,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face verification with optional alignment guidance.")
    parser.add_argument(
        "--mode",
        choices=["guided", "direct"],
        default="guided",
        help="Choose 'guided' to require alignment guidance before verification or 'direct' to capture immediately.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or path to a video file (csi://0) for jetson.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Preferred camera capture width in pixels (0 = leave camera default).",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Preferred camera capture height in pixels (0 = leave camera default).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--weights",
        default=str(default_weights),
        help="Path to MobileFaceNet weights.",
    )
    parser.add_argument(
        "--facebank",
        default=str(default_facebank),
        help="Path to facebank directory.",
    )
    parser.add_argument(
        "--update-facebank",
        action="store_true",
        help="Rebuild the facebank before running verification (disable with --no-update-facebank).",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable flip test-time augmentation for recognition embeddings.",
    )
    parser.add_argument(
        "--identity-thr",
        type=float,
        default=0.85,
        help="Identity acceptance threshold on the 0-1 score scale.",
    )
    parser.add_argument(
        "--detector-thr",
        type=float,
        default=0.7,
        help="Minimum BlazeFace confidence required to keep detections.",
    )
    parser.add_argument(
        "--spoof-weights",
        default=str(default_spoof_weights),
        help="Path to DeePixBiS weights.",
    )
    parser.add_argument(
        "--disable-spoof",
        action="store_true",
        help="Skip DeePixBiS anti-spoofing during verification.",
    )
    parser.add_argument(
        "--spoof-thr",
        type=float,
        default=0.85,
        help="Minimum DeePixBiS score to label a face as real.",
    )
    parser.add_argument(
        "--evaluation-duration",
        type=float,
        default=1.0,
        help="Duration (seconds) to capture frames for verification.",
    )
    parser.add_argument(
        "--attendance-log",
        default=str(default_attendance_log),
        help="Path to append attendance results (JSON lines).",
    )
    parser.add_argument(
        "--guidance-box-size",
        type=int,
        default=224,
        help="Edge length in pixels for the guidance square (0 = auto).",
    )
    parser.add_argument(
        "--guidance-center-tolerance",
        type=float,
        default=0.3,
        help="Fraction of the square half-side tolerated for centering during guidance.",
    )
    parser.add_argument(
        "--guidance-size-tolerance",
        type=float,
        default=0.3,
        help="Fractional tolerance for face size vs square during guidance.",
    )
    parser.add_argument(
        "--guidance-rotation-thr",
        type=float,
        default=15,
        help="Maximum allowed head tilt in degrees during guidance.",
    )
    parser.add_argument(
        "--guidance-hold-frames",
        type=int,
        default=10,
        help="Number of consecutive aligned frames before verification begins.",
    )
    parser.add_argument(
        "--frame-max-size",
        type=str,
        default="1280x720",
        help="Optional maximum resolution for captured frames as WIDTHxHEIGHT (e.g. 1280x720).",
    )
    parser.add_argument(
        "--power-log",
        default=str(default_power_log),
        help="Path to store Jetson power telemetry as JSON (ignored when power logging is disabled).",
    )
    parser.add_argument(
        "--power-interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds for Jetson power logging.",
    )
    parser.add_argument(
        "--enable-power-log",
        action="store_true",
        help="Force-enable Jetson power logging regardless of auto-detection.",
    )
    parser.add_argument(
        "--disable-power-log",
        action="store_true",
        help="Disable Jetson power logging even if auto-detected.",
    )
    parser.add_argument(
        "--quiet-power-log",
        action="store_true",
        help="Suppress per-sample power output while logging.",
    )
    return parser.parse_args()
