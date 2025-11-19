from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from dashboard.theme import ThemeManager


class StatusPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, *, theme: ThemeManager) -> None:
        super().__init__(master)
        self.theme = theme
        self.columnconfigure(1, weight=1)
        bold_font = theme.font("label-bold")
        ttk.Label(self, text="Stage:", font=bold_font).grid(row=0, column=0, sticky="w")
        ttk.Label(self, text="Status:", font=bold_font).grid(row=1, column=0, sticky="w")
        self.stage_var = tk.StringVar(value="Idle")
        self.status_var = tk.StringVar(value="Ready to start.")
        ttk.Label(self, textvariable=self.stage_var, font=theme.font("label")).grid(row=0, column=1, sticky="w")
        ttk.Label(self, textvariable=self.status_var, font=theme.font("label")).grid(row=1, column=1, sticky="w")

    def set_stage(self, text: str) -> None:
        self.stage_var.set(text)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
