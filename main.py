#!/usr/bin/env python3
"""Face verification workflow with optional alignment guidance."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.camera import DEFAULT_CAPTURE_HEIGHT, DEFAULT_CAPTURE_WIDTH, open_video_source
from utils.cli import parse_main_args
from utils.device import select_device
from utils.logging import append_attendance_log
from utils.services import create_services
from utils.session import SessionRunner
from utils.verification import aggregate_observations, compose_final_display

DEFAULT_WEIGHTS = ROOT / "MobileFaceNet" / "Weights" / "MobileFace_Net"
DEFAULT_FACEBANK = ROOT / "facebank"
DEFAULT_SPOOF_WEIGHTS = ROOT / "DeePixBis" / "Weights" / "DeePixBiS.pth"
DEFAULT_ATTENDANCE_LOG = ROOT / "attendance_results.jsonl"

WINDOW_WIDTH_LIMIT = 1280
WINDOW_HEIGHT_LIMIT = 720
GUIDANCE_BOX_SCALE = 0.5
BLAZEFACE_MIN_FACE = 64
MOBILEFACENET_INPUT = 112
DEEPIX_TARGET_SIZE = 224
GUIDANCE_MIN_SIDE = max(BLAZEFACE_MIN_FACE, MOBILEFACENET_INPUT, DEEPIX_TARGET_SIZE)


def main() -> None:
    args = parse_main_args(
        default_weights=DEFAULT_WEIGHTS,
        default_facebank=DEFAULT_FACEBANK,
        default_spoof_weights=DEFAULT_SPOOF_WEIGHTS,
        default_attendance_log=DEFAULT_ATTENDANCE_LOG,
    )
    raw_source = args.source
    source = int(raw_source) if isinstance(raw_source, str) and raw_source.isdigit() else raw_source

    frame_size = None
    if args.camera_width > 0 and args.camera_height > 0:
        frame_size = (args.camera_width, args.camera_height)

    device = select_device(args.device)
    detector, recogniser, spoof_service = create_services(args, device)

    capture = open_video_source(source, frame_size=frame_size)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {raw_source}")

    raw_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolved_width = raw_width if raw_width > 0 else (frame_size[0] if frame_size else DEFAULT_CAPTURE_WIDTH)
    resolved_height = raw_height if raw_height > 0 else (frame_size[1] if frame_size else DEFAULT_CAPTURE_HEIGHT)
    resolved_width = max(1, resolved_width)
    resolved_height = max(1, resolved_height)

    fps_value = capture.get(cv2.CAP_PROP_FPS)
    actual_fps = float(fps_value) if fps_value and fps_value > 0 else 0.0
    if actual_fps > 0:
        print(f"Camera opened at {resolved_width}x{resolved_height} @ {actual_fps:.2f} FPS")
    else:
        print(f"Camera opened at {resolved_width}x{resolved_height}")

    if frame_size and (abs(resolved_width - frame_size[0]) > 1 or abs(resolved_height - frame_size[1]) > 1):
        print(f"Requested {frame_size[0]}x{frame_size[1]} but camera delivered {resolved_width}x{resolved_height}")

    width_scale = WINDOW_WIDTH_LIMIT / resolved_width
    height_scale = WINDOW_HEIGHT_LIMIT / resolved_height
    scale = min(width_scale, height_scale, 1.0)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    display_width = max(1, int(round(resolved_width * scale)))
    display_height = max(1, int(round(resolved_height * scale)))

    window_name = "Face Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)

    runner = SessionRunner(
        capture,
        detector,
        recogniser,
        spoof_service,
        args,
        window_name=window_name,
        window_limits=(display_width, display_height),
        guidance_min_side=GUIDANCE_MIN_SIDE,
        guidance_box_scale=GUIDANCE_BOX_SCALE,
    )

    require_guidance = args.mode == "guided"

    try:
        while True:
            cycle = runner.run_cycle(require_guidance=require_guidance)
            if cycle is None:
                break

            summary = aggregate_observations(cycle.observations, spoof_threshold=args.spoof_thr)
            summary["capture_duration"] = cycle.duration
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

            display_frame = cycle.last_frame
            if display_frame is None:
                frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or resolved_height
                frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or resolved_width
                display_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            final_display = compose_final_display(
                display_frame,
                summary,
                show_spoof_score=spoof_service is not None,
            )
            cv2.imshow(window_name, final_display)

            status_text = "ACCESS GRANTED" if summary["accepted"] else "ACCESS DENIED"
            print(f"[{timestamp}] {status_text}: {summary['identity']}")
            print("Press SPACE to start the next check or ESC to exit.")

            if not runner.wait_for_next_person():
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
