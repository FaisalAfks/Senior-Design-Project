"""Shared attendance pipeline logic for CLI and GUI entry points."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.camera import open_video_source
from utils.device import select_device
from utils.logging import append_attendance_log
from utils.power import JetsonPowerLogger, jetson_power_available
from utils.resolution import build_resizer, parse_max_size
from utils.services import create_services, warmup_services
from utils.session import SessionRunner
from utils.verification import FaceEvaluator, aggregate_observations, compose_direct_display, compose_final_display


DEFAULT_WEIGHTS = ROOT / "MobileFaceNet" / "Weights" / "MobileFace_Net"
DEFAULT_FACEBANK = ROOT / "Facebank"
DEFAULT_SPOOF_WEIGHTS = ROOT / "DeePixBis" / "Weights" / "DeePixBiS.pth"

from utils.paths import logs_path

DEFAULT_ATTENDANCE_LOG = logs_path("attendance_log.csv")
DEFAULT_POWER_LOG = logs_path("jetson_power_metrics.json")

WINDOW_WIDTH_LIMIT = 1280
WINDOW_HEIGHT_LIMIT = 720
GUIDANCE_BOX_SCALE = 0.5
BLAZEFACE_MIN_FACE = 64
MOBILEFACENET_INPUT = 112
DEEPIX_TARGET_SIZE = 224
GUIDANCE_MIN_SIDE = max(BLAZEFACE_MIN_FACE, MOBILEFACENET_INPUT, DEEPIX_TARGET_SIZE)


@dataclass
class SessionCallbacks:
    """Extensible hooks for UI layers to observe session progress."""

    on_guidance_frame: Optional[Callable[[np.ndarray], None]] = None
    on_verification_frame: Optional[Callable[[np.ndarray], None]] = None
    on_final_frame: Optional[Callable[[np.ndarray], None]] = None
    poll_cancel: Optional[Callable[[], bool]] = None
    wait_for_next_person: Optional[Callable[[], bool]] = None
    on_summary: Optional[Callable[[dict[str, object]], None]] = None
    on_status: Optional[Callable[[str], None]] = None
    on_stage_change: Optional[Callable[[str], None]] = None
    on_metrics: Optional[Callable[[dict[str, float]], None]] = None


@dataclass
class CaptureContext:
    capture: cv2.VideoCapture
    raw_source: str | int
    source: str | int
    requested_width: Optional[int]
    requested_height: Optional[int]
    resolved_width: int
    resolved_height: int
    actual_fps: float
    display_width: int
    display_height: int


class AttendancePipeline:
    """Owns BlazeFace/MobileFaceNet/PAD services plus capture/power orchestration."""

    def __init__(self, args) -> None:
        self.args = args
        self.show_summary_scores = bool(getattr(args, "show_summary_scores", True))
        self.minimal_overlay = bool(getattr(args, "minimal_overlay", False))
        self.frame_resizer = self._build_frame_resizer()
        self.device = select_device(args.device)
        self.detector, self.recogniser, self.spoof_service = create_services(args, self.device)
        self.frame_evaluator = FaceEvaluator(
            self.detector,
            self.recogniser,
            self.spoof_service,
            getattr(args, "spoof_thr", 0.5),
        )
        self.power_logger = self._create_power_logger()

    # ------------------------------------------------------------------ public helpers
    def build_session_runner(
        self,
        capture: cv2.VideoCapture,
        *,
        window_name: str,
        window_limits: tuple[int, int],
        callbacks: SessionCallbacks | None = None,
    ) -> SessionRunner:
        return SessionRunner(
            capture,
            self.detector,
            self.recogniser,
            self.spoof_service,
            self.args,
            window_name=window_name,
            window_limits=window_limits,
            guidance_min_side=GUIDANCE_MIN_SIDE,
            guidance_box_scale=GUIDANCE_BOX_SCALE,
            frame_transform=self.frame_resizer,
            guidance_display_callback=callbacks.on_guidance_frame if callbacks else None,
            verification_display_callback=callbacks.on_verification_frame if callbacks else None,
            poll_cancel_callback=callbacks.poll_cancel if callbacks else None,
            wait_for_next_callback=callbacks.wait_for_next_person if callbacks else None,
        )

    def warmup(self, *, width: int, height: int) -> None:
        warmup_services(
            self.detector,
            self.recogniser,
            self.spoof_service,
            frame_size=(width, height),
            iters=2,
        )

    # ------------------------------------------------------------------ capture lifecycle
    def open_capture(self) -> CaptureContext:
        args = self.args
        raw_source = args.source
        source = self._coerce_source(raw_source)
        requested_width = args.camera_width if getattr(args, "camera_width", 0) > 0 else None
        requested_height = args.camera_height if getattr(args, "camera_height", 0) > 0 else None

        capture = open_video_source(
            source,
            width=requested_width,
            height=requested_height,
            fps=getattr(args, "fps", None),
        )
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open source: {raw_source}")

        raw_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fallback_width = requested_width or 0
        fallback_height = requested_height or 0
        resolved_width = max(1, raw_width if raw_width > 0 else fallback_width)
        resolved_height = max(1, raw_height if raw_height > 0 else fallback_height)

        fps_value = capture.get(cv2.CAP_PROP_FPS)
        actual_fps = float(fps_value) if fps_value and fps_value > 0 else 0.0

        display_width, display_height = self._resolve_window_size(resolved_width, resolved_height)
        self._update_power_metadata(
            raw_source=raw_source,
            resolved_width=resolved_width,
            resolved_height=resolved_height,
            actual_fps=actual_fps if actual_fps > 0 else None,
            requested_width=requested_width,
            requested_height=requested_height,
            frame_size=(requested_width, requested_height) if requested_width and requested_height else None,
        )

        return CaptureContext(
            capture=capture,
            raw_source=raw_source,
            source=source,
            requested_width=requested_width,
            requested_height=requested_height,
            resolved_width=resolved_width,
            resolved_height=resolved_height,
            actual_fps=actual_fps,
            display_width=display_width,
            display_height=display_height,
        )

    def close_capture(self, context: CaptureContext | None) -> None:
        if context is None:
            return
        if context.capture is not None:
            context.capture.release()

    def _prepare_window(self, window_name: str, context: CaptureContext) -> None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, context.display_width, context.display_height)

    def start_power_logging(self) -> None:
        self.power_logger.start()

    def stop_power_logging(self) -> None:
        self.power_logger.stop()

    # ------------------------------------------------------------------ CLI execution
    def run_cli(self) -> None:
        context: CaptureContext | None = None
        try:
            with self.power_logger:
                self.power_logger.set_activity("camera-initialization")
                context = self.open_capture()
                if context.actual_fps > 0:
                    print(f"Camera opened at {context.resolved_width}x{context.resolved_height} @ {context.actual_fps:.2f} FPS")
                else:
                    print(f"Camera opened at {context.resolved_width}x{context.resolved_height}")

                if (
                    context.requested_width
                    and context.requested_height
                    and (
                        abs(context.resolved_width - context.requested_width) > 1
                        or abs(context.resolved_height - context.requested_height) > 1
                    )
                ):
                    print(
                        f"Requested {context.requested_width}x{context.requested_height} "
                        f"but camera delivered {context.resolved_width}x{context.resolved_height}"
                    )

                window_name = "Face Verification"
                self._prepare_window(window_name, context)

                try:
                    self.power_logger.set_activity("warmup")
                    self.warmup(width=context.resolved_width, height=context.resolved_height)
                finally:
                    self.power_logger.set_activity("ready")

                runner = self.build_session_runner(
                    context.capture,
                    window_name=window_name,
                    window_limits=(context.display_width, context.display_height),
                )

                if self.args.mode == "direct":
                    self.run_direct_session(context, window_name=window_name)
                else:
                    self.run_guided_session(
                        runner,
                        context=context,
                        window_name=window_name,
                    )
        finally:
            self.close_capture(context)
            cv2.destroyAllWindows()
            self.power_logger.set_activity("cleanup")

    # ------------------------------------------------------------------ internal helpers
    def _build_frame_resizer(self):
        try:
            max_frame_width, max_frame_height = parse_max_size(self.args.frame_max_size)
        except ValueError as exc:
            raise SystemExit(f"Invalid --frame-max-size value: {exc}") from exc
        frame_resizer = build_resizer(max_frame_width, max_frame_height)
        if frame_resizer is not None:
            max_w = max_frame_width or "∞"
            max_h = max_frame_height or "∞"
            print(f"Frame downsampling enabled: max_width={max_w}, max_height={max_h}")
        return frame_resizer

    def _create_power_logger(self) -> JetsonPowerLogger:
        args = self.args
        auto_detect = jetson_power_available()
        if args.enable_power_log:
            power_enabled = True
        elif args.disable_power_log:
            power_enabled = False
        else:
            power_enabled = auto_detect
        logger = JetsonPowerLogger(
            enabled=power_enabled,
            log_path=Path(args.power_log),
            sample_interval=max(float(args.power_interval), 0.1),
            verbose=not args.quiet_power_log,
        )
        if logger.enabled:
            script_name = Path(sys.argv[0]).name or "app.py"
            logger.set_process_target(os.getpid(), script_name)
        return logger

    @staticmethod
    def _coerce_source(raw_source):
        if isinstance(raw_source, str) and raw_source.isdigit():
            return int(raw_source)
        return raw_source

    def _update_power_metadata(
        self,
        *,
        raw_source,
        resolved_width: int,
        resolved_height: int,
        actual_fps: Optional[float],
        requested_width: Optional[int],
        requested_height: Optional[int],
        frame_size: Optional[tuple[int, int]],
    ) -> None:
        if not self.power_logger.enabled:
            return
        self.power_logger.set_resolution(
            resolved_width,
            resolved_height,
            fps=actual_fps,
            source=str(raw_source),
        )
        requested_meta: dict[str, int] = {}
        if requested_width:
            requested_meta["width"] = int(requested_width)
        if requested_height:
            requested_meta["height"] = int(requested_height)
        if requested_meta:
            self.power_logger.update_metadata(requested_resolution=requested_meta)
        if frame_size:
            self.power_logger.update_metadata(target_resolution={"width": frame_size[0], "height": frame_size[1]})
        if self.args.frame_max_size:
            self.power_logger.update_metadata(frame_max_size=self.args.frame_max_size)

    def _resolve_window_size(self, width: int, height: int) -> tuple[int, int]:
        width_scale = WINDOW_WIDTH_LIMIT / width
        height_scale = WINDOW_HEIGHT_LIMIT / height
        scale = min(width_scale, height_scale, 1.0)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        display_width = max(1, int(round(width * scale)))
        display_height = max(1, int(round(height * scale)))
        return display_width, display_height

    def run_direct_session(
        self,
        context: CaptureContext,
        *,
        window_name: str,
        callbacks: SessionCallbacks | None = None,
    ) -> None:
        import time

        capture = context.capture
        frame_resizer = self.frame_resizer
        self.power_logger.set_activity("direct")
        self._notify_stage("direct", callbacks)
        prev_t = time.perf_counter()
        smoothed_fps = None
        while True:
            ok, frame = capture.read()
            if not ok:
                self.power_logger.set_activity("terminating")
                break
            if frame_resizer is not None:
                frame = frame_resizer(frame)

            observation, timings = self.frame_evaluator.evaluate(frame, collect_timings=True)
            annotated = compose_direct_display(
                frame,
                observation,
                spoof_threshold=self.args.spoof_thr,
            )

            now_t = time.perf_counter()
            dt = max(1e-6, now_t - prev_t)
            inst_fps = 1.0 / dt
            smoothed_fps = inst_fps if smoothed_fps is None else (0.85 * smoothed_fps + 0.15 * inst_fps)
            prev_t = now_t

            fps_text = f"FPS: {smoothed_fps:5.1f}"
            x = 20
            y = 30
            cv2.putText(annotated, fps_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y += 28
            cv2.putText(
                annotated,
                f"Detector: {timings.get('detector_ms', 0.0):.1f} ms",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            y += 28
            cv2.putText(
                annotated,
                f"Recogniser: {timings.get('recognition_ms', 0.0):.1f} ms",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            if self.spoof_service is not None:
                y += 28
                cv2.putText(
                    annotated,
                    f"Spoof: {timings.get('spoof_ms', 0.0):.1f} ms",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            instruction = "Press ESC or q to quit"
            (itw, ith), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ix = max(20, (annotated.shape[1] - itw) // 2)
            iy = annotated.shape[0] - 20
            cv2.putText(annotated, instruction, (ix, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if callbacks and callbacks.on_verification_frame:
                callbacks.on_verification_frame(annotated.copy())
            else:
                cv2.imshow(window_name, annotated)

            self._emit_metrics(
                callbacks,
                {
                    "fps": smoothed_fps,
                    "detector_ms": timings.get("detector_ms", 0.0),
                    "recognition_ms": timings.get("recognition_ms", 0.0),
                    "spoof_ms": timings.get("spoof_ms", 0.0),
                },
            )

            if callbacks and callbacks.poll_cancel and callbacks.poll_cancel():
                self.power_logger.set_activity("terminating")
                self._notify_stage("terminating", callbacks)
                break
            if not callbacks or callbacks.on_verification_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.power_logger.set_activity("terminating")
                    self._notify_stage("terminating", callbacks)
                    break

    def run_guided_session(
        self,
        runner: SessionRunner,
        *,
        context: CaptureContext,
        window_name: str,
        require_guidance: Optional[bool] = None,
        callbacks: SessionCallbacks | None = None,
    ) -> None:
        capture = context.capture
        raw_source = context.raw_source
        resolved_width = context.resolved_width
        resolved_height = context.resolved_height
        if require_guidance is None:
            require_guidance = self.args.mode == "guided"
        activity_callback = self._build_activity_handler(callbacks)
        while True:
            cycle = runner.run_cycle(
                require_guidance=require_guidance,
                on_activity_change=activity_callback,
            )
            if cycle is None:
                self.power_logger.set_activity("terminating")
                if callbacks and callbacks.on_stage_change:
                    callbacks.on_stage_change("terminating")
                break

            self.power_logger.set_activity("processing")
            self._notify_stage("processing", callbacks)
            summary = aggregate_observations(cycle.observations, spoof_threshold=self.args.spoof_thr)
            summary["capture_duration"] = cycle.duration
            timestamp = datetime.now(timezone.utc).isoformat()
            self._emit_metrics(callbacks, cycle.metrics)

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
            append_attendance_log(Path(self.args.attendance_log), log_entry)

            display_frame = cycle.last_frame
            if display_frame is None:
                frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or resolved_height
                frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or resolved_width
                display_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            if callbacks and callbacks.wait_for_next_person:
                continue_hint = "Click Next Person to continue or Stop Session to exit"
            else:
                continue_hint = "Press SPACE to continue or ESC to close"
            final_display = compose_final_display(
                display_frame,
                summary,
                show_spoof_score=self.spoof_service is not None and self.show_summary_scores,
                show_scores=self.show_summary_scores,
                minimal_mode=self.minimal_overlay,
                continue_hint=continue_hint,
            )
            annotated_final = self._annotate_metrics(final_display, cycle.metrics)
            if callbacks and callbacks.on_final_frame:
                callbacks.on_final_frame(annotated_final)
            else:
                cv2.imshow(window_name, annotated_final)

            status_text = "ACCESS GRANTED" if summary["accepted"] else "ACCESS DENIED"
            message = f"[{timestamp}] {status_text}: {summary['identity']}"
            print(message)
            if callbacks and callbacks.on_status:
                callbacks.on_status(f"{status_text}: {summary['identity']}")
            self._notify_stage("results", callbacks)
            score_fragments: list[str] = []
            if summary.get("avg_identity_score") is not None:
                score_fragments.append(f"identity={summary['avg_identity_score'] * 100.0:.1f}%")
            if self.spoof_service is not None and summary.get("avg_spoof_score") is not None:
                spoof_label = "real"
                if summary.get("is_real") is False:
                    spoof_label = "spoof"
                elif summary.get("is_real") is None:
                    spoof_label = "unknown"
                score_fragments.append(f"spoof={summary['avg_spoof_score'] * 100.0:.1f}% ({spoof_label})")
            if score_fragments:
                print("Scores: " + ", ".join(score_fragments))
            if callbacks and callbacks.on_summary:
                enriched = summary.copy()
                enriched["timestamp"] = timestamp
                callbacks.on_summary(enriched)
            self.power_logger.set_activity("waiting")

            self._notify_stage("waiting", callbacks)
            if not runner.wait_for_next_person():
                self.power_logger.set_activity("terminating")
                self._notify_stage("terminating", callbacks)
                break

            self.power_logger.set_activity("ready")
            self._notify_stage("guidance" if require_guidance else "ready", callbacks)

    def _build_activity_handler(self, callbacks: SessionCallbacks | None) -> Callable[[str], None]:
        def handler(activity: str) -> None:
            self.power_logger.set_activity(activity)
            self._notify_stage(activity, callbacks)

        return handler

    @staticmethod
    def _notify_stage(stage: str, callbacks: SessionCallbacks | None) -> None:
        if callbacks is None or callbacks.on_stage_change is None:
            return
        callbacks.on_stage_change(stage)

    @staticmethod
    def _emit_metrics(callbacks: SessionCallbacks | None, metrics: Optional[dict[str, float]]) -> None:
        if callbacks is None or callbacks.on_metrics is None or not metrics:
            return
        callbacks.on_metrics(metrics)

    @staticmethod
    def _annotate_metrics(frame: np.ndarray, metrics: Optional[dict[str, float]]) -> np.ndarray:
        # CLI mode no longer annotates metrics directly; return frame unchanged.
        return frame


__all__ = [
    "AttendancePipeline",
    "DEFAULT_ATTENDANCE_LOG",
    "DEFAULT_FACEBANK",
    "DEFAULT_POWER_LOG",
    "DEFAULT_SPOOF_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "GUIDANCE_BOX_SCALE",
    "GUIDANCE_MIN_SIDE",
    "MOBILEFACENET_INPUT",
    "DEEPIX_TARGET_SIZE",
    "BLAZEFACE_MIN_FACE",
    "SessionCallbacks",
    "WINDOW_WIDTH_LIMIT",
    "WINDOW_HEIGHT_LIMIT",
]
