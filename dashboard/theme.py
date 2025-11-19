from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from PIL import Image, ImageTk

from dashboard.utils import LIVE_FEED_SIZE, UI_PAD


FontSpec = Tuple[str, int, str]


@dataclass
class ThemeConfig:
    base_font: FontSpec = ("TkDefaultFont", 12, "normal")
    label_font: FontSpec = ("TkDefaultFont", 12, "normal")
    label_bold_font: FontSpec = ("TkDefaultFont", 12, "bold")
    heading_font: FontSpec = ("TkDefaultFont", 14, "bold")
    title_font: FontSpec = ("TkDefaultFont", 16, "bold")
    notebook_tab_font: FontSpec = ("TkDefaultFont", 12, "bold")
    ui_pad: int = UI_PAD
    live_feed_size: Tuple[int, int] = LIVE_FEED_SIZE
    icon_paths: Dict[str, Path] = field(default_factory=dict)
    font_scale: float = 1.0


class ThemeManager:
    """Centralized theme manager for fonts, spacing, and icon assets."""

    def __init__(self, config: Optional[ThemeConfig] = None) -> None:
        self.config = config or ThemeConfig()
        self._icon_cache: Dict[str, ImageTk.PhotoImage] = {}
        self._fonts = self._build_font_cache()

    def _build_font_cache(self) -> Dict[str, FontSpec]:
        return {
            "base": self._scale_font(self.config.base_font),
            "label": self._scale_font(self.config.label_font),
            "label-bold": self._scale_font(self.config.label_bold_font),
            "heading": self._scale_font(self.config.heading_font),
            "title": self._scale_font(self.config.title_font),
            "notebook-tab": self._scale_font(self.config.notebook_tab_font),
        }

    def _scale_font(self, spec: FontSpec) -> FontSpec:
        family, size, weight = spec
        scale = max(0.5, self.config.font_scale)
        scaled_size = max(6, int(round(size * scale)))
        return (family, scaled_size, weight)

    def apply(self, root: tk.Misc) -> None:
        self._apply_global_fonts(root)
        style = ttk.Style(root)
        style.configure(".", font=self.font("base"))
        style.configure(
            "TNotebook.Tab",
            padding=(14, 6),
            font=self.font("notebook-tab"),
        )

    def font(self, role: str = "base") -> FontSpec:
        return self._fonts.get(role, self._fonts["base"])

    def _apply_global_fonts(self, root: tk.Misc) -> None:
        font_mapping = {
            "TkDefaultFont": self.font("base"),
            "TkTextFont": self.font("base"),
            "TkTooltipFont": self.font("label"),
            "TkMenuFont": self.font("label"),
            "TkHeadingFont": self.font("heading"),
            "TkCaptionFont": self.font("label"),
            "TkSmallCaptionFont": self.font("label"),
        }
        for name, spec in font_mapping.items():
            try:
                tk_named_font = tkfont.nametofont(name)
            except tk.TclError:
                continue
            family, size, weight = spec
            tk_named_font.configure(family=family, size=size, weight=weight)

    def pad(self, role: str = "default") -> int:
        # Hook for future expansion if we add multiple padding sizes.
        return self.config.ui_pad

    @property
    def live_feed_size(self) -> Tuple[int, int]:
        return self.config.live_feed_size

    def icon(self, name: str) -> Optional[ImageTk.PhotoImage]:
        if name in self._icon_cache:
            return self._icon_cache[name]
        path = self.config.icon_paths.get(name)
        if not path or not Path(path).exists():
            return None
        image = Image.open(path)
        photo = ImageTk.PhotoImage(image)
        self._icon_cache[name] = photo
        return photo


__all__ = ["ThemeConfig", "ThemeManager"]
