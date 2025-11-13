"""Shared overlay helpers for rendering readable panels on video frames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

LineSpec = Union[str, Tuple[str, Tuple[int, int, int]]]


@dataclass
class PanelStyle:
    """Visual configuration for overlay panels."""

    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.65
    text_color: Tuple[int, int, int] = (255, 255, 255)
    bg_color: Tuple[int, int, int] = (0, 0, 0)
    alpha: float = 0.55
    padding: int = 12
    margin: int = 16
    thickness: int = 2
    line_spacing: int = 6
    title_color: Tuple[int, int, int] = (210, 210, 210)
    title_scale: float = 0.8
    title_thickness: int = 2


@dataclass
class BannerStyle:
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.8
    text_color: Tuple[int, int, int] = (255, 255, 255)
    bg_color: Tuple[int, int, int] = (0, 0, 0)
    alpha: float = 0.65
    padding: int = 12
    margin: int = 28
    thickness: int = 2


def draw_text_panel(
    frame: np.ndarray,
    lines: Sequence[LineSpec],
    *,
    anchor: str = "top-left",
    title: Optional[str] = None,
    style: Optional[PanelStyle] = None,
) -> np.ndarray:
    """Render a translucent text panel at the requested anchor."""

    style = style or PanelStyle()
    processed_lines: list[Tuple[str, Tuple[int, int, int]]] = []
    line_sizes: list[Tuple[int, int]] = []
    for entry in lines:
        if isinstance(entry, tuple):
            text, color = entry
        else:
            text, color = entry, style.text_color
        if text:
            processed_lines.append((text, color))
            (w, h), _ = cv2.getTextSize(text, style.font, style.font_scale, style.thickness)
            line_sizes.append((w, h))

    if title is None and not processed_lines:
        return frame

    widths = [size[0] for size in line_sizes]
    heights = [size[1] for size in line_sizes]
    total_height = style.padding * 2
    if heights:
        total_height += sum(heights) + style.line_spacing * max(len(heights) - 1, 0)
    panel_width = max(widths or [0]) + style.padding * 2

    title_height = 0
    if title:
        (tw, th), _ = cv2.getTextSize(title, style.font, style.title_scale, style.title_thickness)
        panel_width = max(panel_width, tw + style.padding * 2)
        title_height = th + style.line_spacing
        total_height += th + style.line_spacing

    h, w = frame.shape[:2]
    panel_width = min(panel_width, w - style.margin * 2)
    total_height = min(total_height, h - style.margin * 2)

    anchor = anchor.lower()
    if anchor in ("top-left", "tl"):
        origin_x = style.margin
        origin_y = style.margin
    elif anchor in ("top-right", "tr"):
        origin_x = w - panel_width - style.margin
        origin_y = style.margin
    elif anchor in ("bottom-left", "bl"):
        origin_x = style.margin
        origin_y = h - total_height - style.margin
    elif anchor in ("bottom-right", "br"):
        origin_x = w - panel_width - style.margin
        origin_y = h - total_height - style.margin
    else:
        origin_x = style.margin
        origin_y = style.margin

    origin_x = max(0, origin_x)
    origin_y = max(0, origin_y)

    overlay = frame.copy()
    top_left = (origin_x, origin_y)
    bottom_right = (origin_x + panel_width, origin_y + total_height)
    cv2.rectangle(overlay, top_left, bottom_right, style.bg_color, -1)
    cv2.addWeighted(overlay, style.alpha, frame, 1 - style.alpha, 0, dst=frame)

    cursor_y = origin_y + style.padding
    if title:
        (tw, th), _ = cv2.getTextSize(title, style.font, style.title_scale, style.title_thickness)
        title_x = origin_x + style.padding
        cv2.putText(
            frame,
            title,
            (title_x, cursor_y + th),
            style.font,
            style.title_scale,
            style.title_color,
            style.title_thickness,
        )
        cursor_y += title_height

    for (text, color), (_, height) in zip(processed_lines, line_sizes or [(0, 0)]):
        text_x = origin_x + style.padding
        cv2.putText(
            frame,
            text,
            (text_x, cursor_y + height),
            style.font,
            style.font_scale,
            color,
            style.thickness,
        )
        cursor_y += height + style.line_spacing

    return frame


def draw_center_banner(
    frame: np.ndarray,
    text: str,
    *,
    position: str = "bottom",
    style: Optional[BannerStyle] = None,
) -> np.ndarray:
    """Render a single-line banner centered horizontally near the top/bottom."""

    if not text:
        return frame
    style = style or BannerStyle()
    h, w = frame.shape[:2]
    overlay = frame.copy()
    (tw, th), _ = cv2.getTextSize(text, style.font, style.font_scale, style.thickness)
    rect_width = tw + style.padding * 2
    rect_height = th + style.padding * 2

    x = max(style.margin, (w - rect_width) // 2)
    y = style.margin if position.lower().startswith("top") else h - rect_height - style.margin
    x = min(x, w - rect_width - style.margin)
    y = max(0, min(y, h - rect_height))

    cv2.rectangle(overlay, (x, y), (x + rect_width, y + rect_height), style.bg_color, -1)
    cv2.addWeighted(overlay, style.alpha, frame, 1 - style.alpha, 0, dst=frame)
    text_origin = (x + style.padding, y + rect_height - style.padding)
    cv2.putText(
        frame,
        text,
        text_origin,
        style.font,
        style.font_scale,
        style.text_color,
        style.thickness,
    )
    return frame


__all__ = ["PanelStyle", "BannerStyle", "draw_text_panel", "draw_center_banner"]
