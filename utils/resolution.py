"""Utility helpers for parsing and applying resolution constraints."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import cv2
import numpy as np


def parse_max_size(value: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse a WIDTHxHEIGHT expression into integer limits.

    Accepts separators ``x``, ``X``, or a comma. Missing values mean "no limit".
    Raises ``ValueError`` if parsing fails or both limits are missing/non-positive.
    """
    if value is None:
        return None, None
    text = value.strip()
    if not text:
        return None, None

    for separator in ("x", "X", ","):
        if separator in text:
            parts = [part.strip() for part in text.split(separator)]
            break
    else:
        parts = [text]

    if len(parts) == 1:
        width = _coerce_positive(parts[0])
        height = None
    else:
        width = _coerce_positive(parts[0]) if parts[0] else None
        height = _coerce_positive(parts[1]) if len(parts) > 1 and parts[1] else None

    if width is None and height is None:
        raise ValueError("at least one of width or height must be positive")
    return width, height


def scaled_dimensions(
    width: int,
    height: int,
    max_width: Optional[int],
    max_height: Optional[int],
) -> Tuple[int, int]:
    """Return ``(new_width, new_height)`` limited by the provided maximums."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")

    scale = 1.0
    if max_width is not None and max_width > 0 and width > max_width:
        scale = min(scale, max_width / width)
    if max_height is not None and max_height > 0 and height > max_height:
        scale = min(scale, max_height / height)

    if scale >= 1.0:
        return width, height

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    # Guard against rounding overflow.
    new_width = min(new_width, width)
    new_height = min(new_height, height)
    return new_width, new_height


def build_resizer(
    max_width: Optional[int],
    max_height: Optional[int],
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Return a callable that resizes frames to fit within the requested bounds."""
    if max_width is None and max_height is None:
        return None

    def resize(frame):
        height, width = frame.shape[:2]
        new_width, new_height = scaled_dimensions(width, height, max_width, max_height)
        if new_width == width and new_height == height:
            return frame
        interpolation = cv2.INTER_AREA if new_width <= width and new_height <= height else cv2.INTER_LINEAR
        return cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

    return resize


def _coerce_positive(text: str) -> Optional[int]:
    if not text:
        return None
    value = int(text)
    if value <= 0:
        return None
    return value
