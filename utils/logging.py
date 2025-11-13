"""Logging helpers for attendance tracking."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable

CSV_FIELDS: tuple[str, ...] = (
    "timestamp",
    "source",
    "recognized",
    "identity",
    "avg_identity_score",
    "avg_spoof_score",
    "is_real",
    "accepted",
    "frames_with_detections",
    "capture_duration",
)


def append_attendance_log(path: Path, entry: Dict[str, object]) -> None:
    """Append an attendance record as CSV (preferred) or JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        _append_csv(path, entry, CSV_FIELDS)
    else:
        _append_jsonl(path, entry)


def _append_jsonl(path: Path, entry: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as log_file:
        json.dump(entry, log_file)
        log_file.write("\n")


def _append_csv(path: Path, entry: Dict[str, object], fieldnames: Iterable[str]) -> None:
    fieldnames = tuple(fieldnames)
    write_header = not path.exists() or path.stat().st_size == 0
    normalized = {field: entry.get(field) for field in fieldnames}
    with path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)


__all__ = ["append_attendance_log", "CSV_FIELDS"]
