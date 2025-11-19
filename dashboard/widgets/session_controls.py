from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from dashboard.theme import ThemeManager


class SessionButtons(ttk.Frame):
    """Start/stop/resume controls for the live session."""

    def __init__(self, master: tk.Misc, *, theme: Optional[ThemeManager] = None) -> None:
        super().__init__(master)
        pad = theme.pad("inner") if theme else 6
        self.start_button = ttk.Button(self, text="Start Session", width=18)
        self.start_button.grid(row=0, column=0, padx=(0, pad))
        self.pause_button = ttk.Button(self, text="Stop Session", width=18)
        self.pause_button.grid(row=0, column=1, padx=pad)
        self.pause_button.state(["disabled"])
        self.resume_button = ttk.Button(self, text="Resume Session", width=18)
        self.resume_button.grid(row=0, column=2, padx=(pad, 0))
        self.columnconfigure((0, 1, 2), weight=1)

    def configure_start(self, *, text: str, command: Callable[[], None]) -> None:
        self.start_button.config(text=text, command=command)

    def set_start_enabled(self, enabled: bool) -> None:
        self.start_button.state(["!disabled"] if enabled else ["disabled"])

    def configure_stop(self, command: Callable[[], None]) -> None:
        self.pause_button.config(command=command)

    def set_pause_enabled(self, enabled: bool) -> None:
        self.pause_button.state(["!disabled"] if enabled else ["disabled"])

    def configure_resume(self, command: Callable[[], None]) -> None:
        self.resume_button.config(command=command)

    def set_resume_enabled(self, enabled: bool) -> None:
        self.resume_button.state(["!disabled"] if enabled else ["disabled"])


class RegistrationPanel(ttk.Frame):
    """Inline controls for the guided registration workflow."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_capture: Callable[[], None],
        on_save: Callable[[], None],
        on_cancel: Callable[[], None],
        theme: Optional[ThemeManager] = None,
    ) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.name_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Registration idle.")
        pad = theme.pad("inner") if theme else 6

        ttk.Label(self, text="Identity name", anchor="center").grid(row=0, column=0, columnspan=2, sticky="ew")
        name_entry = ttk.Entry(self, textvariable=self.name_var, justify="center", width=30)
        name_entry.grid(row=1, column=0, columnspan=2, pady=(0, pad))

        button_row = ttk.Frame(self)
        button_row.grid(row=2, column=0, columnspan=2, pady=(0, pad))
        ttk.Button(button_row, text="Capture Sample", command=on_capture, width=18).grid(row=0, column=0, padx=pad)
        ttk.Button(button_row, text="Save Samples", command=on_save, width=18).grid(row=0, column=1, padx=pad)
        ttk.Button(button_row, text="Cancel", command=on_cancel, width=18).grid(row=0, column=2, padx=pad)

        ttk.Label(self, textvariable=self.status_var, anchor="center").grid(row=3, column=0, columnspan=2, sticky="ew")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_name(self, text: str) -> None:
        self.name_var.set(text)

    def get_name(self) -> str:
        return self.name_var.get().strip()
