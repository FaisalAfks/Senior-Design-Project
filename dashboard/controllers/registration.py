from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from BlazeFace import BlazeFaceService
from dashboard.configuration import DemoConfig, DEFAULT_DETECTOR_THR
from dashboard.utils import _next_facebank_index
from pipelines.attendance import DEFAULT_FACEBANK
from utils.camera import open_video_source
from utils.device import select_device


class RegistrationSession:
    """Background camera capture used during dashboard registration mode."""

    MAX_SAMPLES: Optional[int] = None

    def __init__(self, config: DemoConfig, submit_frame: Callable[[np.ndarray], None]) -> None:
        self.config = config
        self.submit_frame = submit_frame
        self.capture: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.samples: List[np.ndarray] = []
        self.latest_frame: Optional[np.ndarray] = None
        self.detector: Optional[BlazeFaceService] = None
        self._device = select_device(self.config.device)
        self._frame_lock = threading.Lock()

    def start(self) -> None:
        self.capture = open_video_source(
            self.config.source,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
        )
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError("Unable to open camera source for registration.")
        self.detector = BlazeFaceService(score_threshold=DEFAULT_DETECTOR_THR, device=self._device)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self) -> None:
        capture = self.capture
        if capture is None:
            return
        while not self.stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                continue
            with self._frame_lock:
                self.latest_frame = frame.copy()
            self.submit_frame(frame)

    def capture_sample(self) -> int:
        with self._frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            raise RuntimeError("Camera not ready yet.")
        if self.max_samples and len(self.samples) >= self.max_samples:
            raise RuntimeError("Maximum samples captured.")
        if self.detector is None:
            raise RuntimeError("Face detector unavailable.")
        detections = self.detector.detect(frame)
        best = max(detections, key=lambda det: det.score) if detections else None
        if best is None:
            raise RuntimeError("No face detected; align before capturing.")
        aligned = self.detector.detector.align_face(frame, best)
        if aligned is None:
            raise RuntimeError("Unable to align face; adjust your pose and try again.")
        self.samples.append(aligned)
        return len(self.samples)

    def save_samples(self, identity: str) -> int:
        if not self.samples:
            raise RuntimeError("No samples captured.")
        face_dir = DEFAULT_FACEBANK / identity
        face_dir.mkdir(parents=True, exist_ok=True)
        start_index = _next_facebank_index(face_dir)
        saved = 0
        for idx, frame in enumerate(self.samples, start=start_index):
            target = face_dir / f"facebank_{idx:03d}.png"
            if cv2.imwrite(str(target), frame):
                saved += 1
        self.samples.clear()
        return saved

    def stop(self) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=1.0)
            self.thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.detector = None

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def max_samples(self) -> int:
        return self.MAX_SAMPLES or 0


__all__ = ["RegistrationSession"]
