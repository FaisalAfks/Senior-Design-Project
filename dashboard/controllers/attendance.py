from __future__ import annotations

import threading
from typing import Callable, Optional

from pipelines.attendance import AttendancePipeline, SessionCallbacks

from dashboard.configuration import DemoConfig


class AttendanceSessionController:
    """Wrap the AttendancePipeline for in-window sessions."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.pipeline: Optional[AttendancePipeline] = None
        self.capture_context = None
        self.session_runner = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._error_handler: Optional[Callable[[BaseException], None]] = None
        self._blocked_identity_checker: Optional[Callable[[str], bool]] = None

    def start(self, callbacks: SessionCallbacks, *, on_error: Optional[Callable[[BaseException], None]] = None) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._error_handler = on_error
        pipeline = self._ensure_pipeline()
        self.capture_context = pipeline.open_capture()
        pipeline.show_summary_scores = self.config.display_scores
        merged_callbacks = self._wrap_callbacks(callbacks)
        self.session_runner = pipeline.build_session_runner(
            self.capture_context.capture,
            window_name="Dashboard Session",
            window_limits=(self.capture_context.display_width, self.capture_context.display_height),
            callbacks=merged_callbacks,
            blocked_identity_checker=self._blocked_identity_checker,
        )
        self._thread = threading.Thread(target=self._run_session, args=(merged_callbacks,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._release_capture()
        self.session_runner = None
        self._thread = None

    def _run_session(self, callbacks: SessionCallbacks) -> None:
        if self.pipeline is None or self.capture_context is None:
            return
        try:
            with self.pipeline.power_logger:
                self.pipeline.power_logger.set_activity("warmup")
                self.pipeline.warmup(
                    width=self.capture_context.resolved_width,
                    height=self.capture_context.resolved_height,
                )
                self.pipeline.power_logger.set_activity("ready")
                self.pipeline.run_guided_session(
                    self.session_runner,
                    context=self.capture_context,
                    window_name="Dashboard Session",
                    callbacks=callbacks,
                )
        except Exception as exc:
            self._handle_run_exception(exc)
        finally:
            self._release_capture()

    def _wrap_callbacks(self, callbacks: SessionCallbacks) -> SessionCallbacks:
        def poll_cancel() -> bool:
            if self._stop.is_set():
                return True
            if callbacks.poll_cancel:
                return bool(callbacks.poll_cancel())
            return False

        def status_change(text: str) -> None:
            if self._stop.is_set():
                return
            if callbacks.on_status:
                callbacks.on_status(text)

        def stage_change(stage: str) -> None:
            if self._stop.is_set():
                return
            if callbacks.on_stage_change:
                callbacks.on_stage_change(stage)

        return SessionCallbacks(
            on_guidance_frame=callbacks.on_guidance_frame,
            on_verification_frame=callbacks.on_verification_frame,
            on_final_frame=callbacks.on_final_frame,
            poll_cancel=poll_cancel,
            wait_for_next_person=callbacks.wait_for_next_person,
            on_summary=callbacks.on_summary,
            on_status=status_change,
            on_stage_change=stage_change,
            on_metrics=callbacks.on_metrics,
        )

    def _handle_run_exception(self, exc: BaseException) -> None:
        self._stop.set()
        handler = self._error_handler
        if handler:
            try:
                handler(exc)
            finally:
                self._error_handler = None

    def set_display_scores(self, value: bool) -> None:
        self.config.display_scores = value
        if self.pipeline is not None:
            self.pipeline.show_summary_scores = value

    def set_blocked_identity_checker(self, checker: Optional[Callable[[str], bool]]) -> None:
        self._blocked_identity_checker = checker

    def refresh_facebank(self) -> bool:
        pipeline = self.pipeline
        if pipeline is None or pipeline.recogniser is None:
            return False
        pipeline.recogniser.rebuild_facebank(self.config.facebank_dir)
        return True

    def _release_capture(self) -> None:
        if self.pipeline and self.capture_context:
            try:
                self.pipeline.close_capture(self.capture_context)
            except Exception:
                pass
        self.capture_context = None
        self.session_runner = None

    def _ensure_pipeline(self) -> AttendancePipeline:
        pipeline = self.pipeline
        if pipeline is None:
            args = self.config.to_pipeline_args()
            pipeline = AttendancePipeline(args)
            self.pipeline = pipeline
        return pipeline

    def shutdown(self) -> None:
        self.stop()
        self.pipeline = None


__all__ = ["AttendanceSessionController"]
