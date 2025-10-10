"""Logging helpers for attendance tracking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def append_attendance_log(path: Path, entry: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        json.dump(entry, log_file)
        log_file.write("\n")

