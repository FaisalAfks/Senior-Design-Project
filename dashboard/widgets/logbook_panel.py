from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk

from dashboard.theme import ThemeManager
from .log_panel import AttendanceLog


class LogbookPanel(ttk.Frame):
    """Encapsulate the attendance log tree plus navigation and actions."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_prev: Callable[[], None],
        on_next: Callable[[], None],
        on_new: Callable[[], None],
        on_rename: Callable[[], None],
        on_delete: Callable[[], None],
        on_export: Callable[[], None],
        theme: ThemeManager,
    ) -> None:
        super().__init__(master)
        self._on_prev = on_prev
        self._on_next = on_next
        self._title_var = tk.StringVar(value="Loading logs...")
        self.theme = theme

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        pad = theme.pad()
        half_pad = max(1, pad // 2)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=pad, pady=(pad, half_pad))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            textvariable=self._title_var,
            font=theme.font("heading"),
            anchor="center",
        ).grid(row=0, column=0, sticky="ew")

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew", padx=pad, pady=(0, half_pad))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self._log_view = AttendanceLog(body)
        self._log_view.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(self)
        footer.grid(row=2, column=0, sticky="ew", padx=pad, pady=(half_pad, pad))
        footer.columnconfigure(0, weight=1)
        footer.columnconfigure(1, weight=1)

        nav_frame = ttk.Frame(footer)
        nav_frame.grid(row=0, column=0, sticky="w")
        self.prev_button = ttk.Button(nav_frame, text="<< Previous Log", width=18, command=self._on_prev)
        self.prev_button.grid(row=0, column=0, padx=(0, 6))
        self.next_button = ttk.Button(nav_frame, text="Next Log >>", width=18, command=self._on_next)
        self.next_button.grid(row=0, column=1, padx=(6, 0))

        actions_frame = ttk.Frame(footer)
        actions_frame.grid(row=0, column=1, sticky="e")
        ttk.Button(actions_frame, text="New Log", command=on_new).grid(row=0, column=0, padx=4)
        ttk.Button(actions_frame, text="Rename Log", command=on_rename).grid(row=0, column=1, padx=4)
        ttk.Button(actions_frame, text="Delete Log", command=on_delete).grid(row=0, column=2, padx=4)
        ttk.Button(actions_frame, text="Export CSV", command=on_export).grid(row=0, column=3, padx=4)

    # passthrough helpers -------------------------------------------------
    def set_title(self, text: str) -> None:
        self._title_var.set(text)

    def clear_entries(self) -> None:
        self._log_view.clear()

    def load_entries(self, entries: Iterable[dict[str, object]]) -> None:
        self._log_view.load_entries(entries)

    def add_entry(self, *, timestamp: str, identity: str, accepted: bool, source: str) -> None:
        self._log_view.add_entry(
            timestamp=timestamp,
            identity=identity,
            accepted=accepted,
            source=source,
        )

    def set_prev_enabled(self, enabled: bool) -> None:
        self.prev_button.state(["!disabled"] if enabled else ["disabled"])

    def set_next_enabled(self, enabled: bool) -> None:
        self.next_button.state(["!disabled"] if enabled else ["disabled"])
