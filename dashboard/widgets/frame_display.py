from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk
from tkinter import ttk


class FrameDisplay:
    """Lightweight OpenCV -> Tkinter bridge with optional letterboxing."""

    def __init__(self, target_label: ttk.Label, *, target_size: Optional[tuple[int, int]] = None) -> None:
        self.label = target_label
        self.target_size = target_size
        self._lock = threading.Lock()
        self._buffer: Optional[np.ndarray] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._active = False

    def start(self, interval_ms: int = 30) -> None:
        if self._active:
            return
        self._active = True
        self._pump(interval_ms)

    def stop(self) -> None:
        self._active = False

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buffer = frame.copy()

    def clear(self) -> None:
        with self._lock:
            self._buffer = None
        self._photo = None
        self.label.configure(image="")

    def _pump(self, interval_ms: int) -> None:
        if not self._active:
            return
        frame = None
        with self._lock:
            if self._buffer is not None:
                frame = self._buffer
                self._buffer = None
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if self.target_size:
                image = self._fit_to_target(image, self.target_size)
            self._photo = ImageTk.PhotoImage(image=image)
            self.label.configure(image=self._photo)
        self.label.after(interval_ms, lambda: self._pump(interval_ms))

    def _fit_to_target(self, image: Image.Image, target: tuple[int, int]) -> Image.Image:
        target_w, target_h = target
        if target_w <= 0 or target_h <= 0:
            return image
        contained = ImageOps.contain(image, target, method=Image.LANCZOS)
        if contained.size == target:
            return contained
        canvas = Image.new("RGB", target, color=(0, 0, 0))
        offset = ((target_w - contained.width) // 2, (target_h - contained.height) // 2)
        canvas.paste(contained, offset)
        return canvas
