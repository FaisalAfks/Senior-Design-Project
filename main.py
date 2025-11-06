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

from utils.camera import open_video_source
from utils.cli import parse_main_args
from utils.device import select_device
from utils.logging import append_attendance_log
from utils.paths import logs_path
from utils.power import JetsonPowerLogger, jetson_power_available
from utils.resolution import build_resizer, parse_max_size
from utils.services import create_services, warmup_services
from utils.session import SessionRunner
from utils.verification import (
    aggregate_observations,
    compose_final_display,
    compose_direct_display,
)

DEFAULT_WEIGHTS = ROOT / "MobileFaceNet" / "Weights" / "MobileFace_Net"
DEFAULT_FACEBANK = ROOT / "Facebank"
DEFAULT_SPOOF_WEIGHTS = ROOT / "DeePixBis" / "Weights" / "DeePixBiS.pth"
DEFAULT_ATTENDANCE_LOG = logs_path("attendance_results.jsonl")
DEFAULT_POWER_LOG = logs_path("jetson_power_log.json")

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
        default_power_log=DEFAULT_POWER_LOG,
    )
    raw_source = args.source
    source = int(raw_source) if isinstance(raw_source, str) and raw_source.isdigit() else raw_source

    requested_width = args.camera_width if args.camera_width > 0 else None
    requested_height = args.camera_height if args.camera_height > 0 else None
    frame_size = (requested_width, requested_height) if requested_width and requested_height else None

    try:
        max_frame_width, max_frame_height = parse_max_size(args.frame_max_size)
    except ValueError as exc:
        raise SystemExit(f"Invalid --frame-max-size value: {exc}") from exc
    frame_resizer = build_resizer(max_frame_width, max_frame_height)
    if frame_resizer is not None:
        print("Frame downsampling enabled: "f"max_width={max_frame_width or '∞'}, max_height={max_frame_height or '∞'}")

    device = select_device(args.device)
    detector, recogniser, spoof_service = create_services(args, device)

    auto_detect = jetson_power_available()
    if args.enable_power_log:
        power_enabled = True
    elif args.disable_power_log:
        power_enabled = False
    else:
        power_enabled = auto_detect
    power_logger = JetsonPowerLogger(
        enabled=power_enabled,
        log_path=Path(args.power_log),
        sample_interval=max(float(args.power_interval), 0.1),
        verbose=not args.quiet_power_log,
    )

    with power_logger:
        power_logger.set_activity("camera-initialization")
        capture = open_video_source(source, width=requested_width, height=requested_height)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open source: {raw_source}")

        raw_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fallback_width = frame_size[0] if frame_size else (requested_width or 0)
        fallback_height = frame_size[1] if frame_size else (requested_height or 0)
        resolved_width = raw_width if raw_width > 0 else fallback_width
        resolved_height = raw_height if raw_height > 0 else fallback_height
        resolved_width = max(1, resolved_width)
        resolved_height = max(1, resolved_height)

        fps_value = capture.get(cv2.CAP_PROP_FPS)
        actual_fps = float(fps_value) if fps_value and fps_value > 0 else 0.0
        if actual_fps > 0:
            print(f"Camera opened at {resolved_width}x{resolved_height} @ {actual_fps:.2f} FPS")
        else:
            print(f"Camera opened at {resolved_width}x{resolved_height}")

        if power_logger.enabled:
            fps_for_log = actual_fps if actual_fps > 0 else None
            power_logger.set_resolution(
                resolved_width,
                resolved_height,
                fps=fps_for_log,
                source=str(raw_source),
            )
            requested_meta: dict[str, int] = {}
            if requested_width:
                requested_meta["width"] = int(requested_width)
            if requested_height:
                requested_meta["height"] = int(requested_height)
            if requested_meta:
                power_logger.update_metadata(requested_resolution=requested_meta)
            if frame_size:
                power_logger.update_metadata(target_resolution={"width": frame_size[0], "height": frame_size[1]})
            if args.frame_max_size:
                power_logger.update_metadata(frame_max_size=args.frame_max_size)

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

        # Proactively warm up CUDA/cuDNN by running a few dry forwards
        # so the first verification doesn't stall. This uses constant
        # dummy inputs matching the expected model input sizes.
        try:
            power_logger.set_activity("warmup")
            warmup_services(
                detector,
                recogniser,
                spoof_service,
                frame_size=(resolved_width, resolved_height),
                iters=2,
            )
        finally:
            power_logger.set_activity("ready")

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
            frame_transform=frame_resizer,
        )

        require_guidance = args.mode == "guided"
        power_logger.set_activity("ready")

        try:
            if args.mode == "direct":
                power_logger.set_activity("direct")
                import time
                prev_t = time.perf_counter()
                smoothed_fps = None
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        power_logger.set_activity("terminating")
                        break
                    if frame_resizer is not None:
                        frame = frame_resizer(frame)

                    from utils.verification import evaluate_frame_with_timing
                    observation, timings = evaluate_frame_with_timing(frame, detector, recogniser, spoof_service, args.spoof_thr)
                    annotated = compose_direct_display(
                        frame,
                        observation,
                        spoof_threshold=args.spoof_thr,
                    )

                    # Always show FPS (smoothed) and per-system timings at top-left
                    now_t = time.perf_counter()
                    dt = max(1e-6, now_t - prev_t)
                    inst_fps = 1.0 / dt
                    smoothed_fps = inst_fps if smoothed_fps is None else (0.85 * smoothed_fps + 0.15 * inst_fps)
                    prev_t = now_t
                    # FPS in cyan-like color (BGR: 255,255,0)
                    fps_text = f"FPS: {smoothed_fps:5.1f}"
                    x = 20
                    y = 30
                    cv2.putText(annotated, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # Timings in ms for detector / recogniser / spoof (yellow)
                    y += 28
                    cv2.putText(annotated, f"Detector: {timings.get('detector_ms', 0.0):.1f} ms", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    y += 28
                    cv2.putText(annotated, f"Recogniser: {timings.get('recognition_ms', 0.0):.1f} ms", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if spoof_service is not None:
                        y += 28
                        cv2.putText(annotated, f"Spoof: {timings.get('spoof_ms', 0.0):.1f} ms", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Hint text
                    instruction = "Press ESC or q to quit"
                    (itw, ith), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    ix = max(20, (annotated.shape[1] - itw) // 2)
                    iy = annotated.shape[0] - 20
                    cv2.putText(annotated, instruction, (ix, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.imshow(window_name, annotated)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        power_logger.set_activity("terminating")
                        break
            else:
                while True:
                    cycle = runner.run_cycle(
                        require_guidance=require_guidance,
                        on_activity_change=power_logger.set_activity,
                    )
                    if cycle is None:
                        power_logger.set_activity("terminating")
                        break

                    power_logger.set_activity("processing")
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
                    score_fragments = []
                    if summary.get("avg_identity_score") is not None:
                        score_fragments.append(f"identity={summary['avg_identity_score'] * 100.0:.1f}%")
                    if spoof_service is not None and summary.get("avg_spoof_score") is not None:
                        spoof_label = "real"
                        if summary.get("is_real") is False:
                            spoof_label = "spoof"
                        elif summary.get("is_real") is None:
                            spoof_label = "unknown"
                        score_fragments.append(f"spoof={summary['avg_spoof_score'] * 100.0:.1f}% ({spoof_label})")
                    if score_fragments:
                        print("Scores: " + ", ".join(score_fragments))
                    power_logger.set_activity("waiting")

                    if not runner.wait_for_next_person():
                        power_logger.set_activity("terminating")
                        break

                    power_logger.set_activity("ready")
        finally:
            capture.release()
            cv2.destroyAllWindows()
            power_logger.set_activity("cleanup")


if __name__ == "__main__":
    main()
