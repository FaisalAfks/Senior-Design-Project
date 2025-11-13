"""Command-line argument builders for project entry points."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


def parse_main_args(
    *,
    default_weights: Path,
    default_facebank: Path,
    default_spoof_weights: Path,
    default_attendance_log: Path,
    default_power_log: Path,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face verification with optional alignment guidance.")

    capture_group = parser.add_argument_group("Capture settings")
    capture_group.add_argument(
        "--mode",
        choices=["guided", "direct"],
        default="guided",
        help="Choose 'guided' to require alignment guidance before verification or 'direct' to capture immediately.",
    )
    capture_group.add_argument(
        "--source",
        default="csi://0",
        help="Camera index (e.g. 0) or path to a video file (csi://0) for jetson.",
    )
    capture_group.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Preferred camera capture width in pixels (0 = leave camera default).",
    )
    capture_group.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Preferred camera capture height in pixels (0 = leave camera default).",
    )
    capture_group.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run on (e.g. cpu, cuda, cuda:0).",
    )
    capture_group.add_argument(
        "--attendance-log",
        default=str(default_attendance_log),
        help="Path to append attendance results (JSON lines).",
    )
    capture_group.add_argument(
        "--frame-max-size",
        type=str,
        default="1280x720",
        help="Optional maximum resolution for captured frames as WIDTHxHEIGHT (e.g. 1280x720).",
    )

    model_group = parser.add_argument_group("Models and thresholds")
    model_group.add_argument(
        "--weights",
        default=str(default_weights),
        help="Path to MobileFaceNet weights.",
    )
    model_group.add_argument(
        "--facebank",
        default=str(default_facebank),
        help="Path to facebank directory.",
    )
    model_group.add_argument(
        "--update-facebank",
        action="store_true",
        help="Rebuild the facebank before running verification (disable with --no-update-facebank).",
    )
    model_group.add_argument(
        "--tta",
        action="store_true",
        help="Enable flip test-time augmentation for recognition embeddings.",
    )
    model_group.add_argument(
        "--identity-thr",
        type=float,
        default=0.9,
        help="Identity acceptance threshold on the 0-1 score scale.",
    )
    model_group.add_argument(
        "--detector-thr",
        type=float,
        default=0.8,
        help="Minimum BlazeFace confidence required to keep detections.",
    )
    model_group.add_argument(
        "--spoof-weights",
        default=str(default_spoof_weights),
        help="Path to DeePixBiS weights.",
    )
    model_group.add_argument(
        "--disable-spoof",
        action="store_true",
        help="Skip DeePixBiS anti-spoofing during verification.",
    )
    model_group.add_argument(
        "--spoof-thr",
        type=float,
        default=0.90,
        help="Minimum DeePixBiS score to label a face as real.",
    )

    verification_group = parser.add_argument_group("Verification loop")
    verification_group.add_argument(
        "--evaluation-duration",
        type=float,
        default=0.9,
        help="Duration (seconds) to capture frames for verification.",
    )
    verification_group.add_argument(
        "--evaluation-mode",
        choices=["time", "frames"],
        default="time",
        help="Capture by wall-time budget ('time') or fixed frame count ('frames').",
    )
    verification_group.add_argument(
        "--evaluation-frames",
        type=int,
        default=30,
        help="Number of frames to capture when --evaluation-mode=frames.",
    )

    guidance_group = parser.add_argument_group("Guidance flow")
    guidance_group.add_argument(
        "--guidance-box-size",
        type=int,
        default=224,
        help="Edge length in pixels for the guidance square (0 = auto).",
    )
    guidance_group.add_argument(
        "--guidance-center-tolerance",
        type=float,
        default=0.3,
        help="Fraction of the square half-side tolerated for centering during guidance.",
    )
    guidance_group.add_argument(
        "--guidance-size-tolerance",
        type=float,
        default=0.3,
        help="Fractional tolerance for face size vs square during guidance.",
    )
    guidance_group.add_argument(
        "--guidance-rotation-thr",
        type=float,
        default=15,
        help="Maximum allowed head tilt in degrees during guidance.",
    )
    guidance_group.add_argument(
        "--guidance-hold-frames",
        type=int,
        default=10,
        help="Number of consecutive aligned frames before verification begins.",
    )

    power_group = parser.add_argument_group("Jetson power logging")
    power_group.add_argument(
        "--power-log",
        default=str(default_power_log),
        help="Path to store Jetson power telemetry as JSON (ignored when power logging is disabled).",
    )
    power_group.add_argument(
        "--power-interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds for Jetson power logging.",
    )
    power_group.add_argument(
        "--enable-power-log",
        action="store_true",
        help="Force-enable Jetson power logging regardless of auto-detection.",
    )
    power_group.add_argument(
        "--disable-power-log",
        action="store_true",
        help="Disable Jetson power logging even if auto-detected.",
    )
    power_group.add_argument(
        "--quiet-power-log",
        action="store_true",
        help="Suppress per-sample power output while logging.",
    )

    display_group = parser.add_argument_group("Display / overlay")
    display_group.add_argument(
        "--minimal-overlay",
        action="store_true",
        help="Show a user-friendly overlay (hide scores/FPS) instead of detailed diagnostics.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)
