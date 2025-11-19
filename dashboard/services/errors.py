from __future__ import annotations

import traceback
from typing import Optional

import tkinter as tk
from tkinter import messagebox


def format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc.__class__, exc, exc.__traceback__))


def show_error_dialog(title: str, message: str, *, parent: Optional[tk.Misc] = None) -> None:
    """Mirror UI error dialogs to the terminal for easier debugging."""
    print(f"[Dashboard][Error] {title}: {message}")
    messagebox.showerror(title, message, parent=parent)


__all__ = ["format_exception", "show_error_dialog"]
