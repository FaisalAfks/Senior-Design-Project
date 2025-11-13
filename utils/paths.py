"""Shared path utilities for locating large external assets such as datasets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_ROOT = PROJECT_ROOT / "logs"
_DATASET_ENV_VAR = "SDP_DATASET_ROOT"
_GOOGLE_DRIVE_DATASET = Path(r"U:\My Drive\Senior Design Project\Dataset")


def _candidate_dataset_roots() -> List[Path]:
    """Return dataset root candidates ordered by priority."""
    candidates: List[Path] = []
    env_value = os.environ.get(_DATASET_ENV_VAR)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(_GOOGLE_DRIVE_DATASET)
    candidates.append(PROJECT_ROOT / "Dataset")
    # Remove duplicates while preserving order.
    seen: set[Path] = set()
    unique: List[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def list_dataset_roots() -> List[Path]:
    """Expose the ordered dataset root candidates."""
    return _candidate_dataset_roots()


def resolve_dataset_root(*, must_exist: bool = False) -> Path:
    """Resolve the dataset root, preferring existing directories."""
    for candidate in _candidate_dataset_roots():
        if candidate.exists():
            return candidate
    root = _candidate_dataset_roots()[0]
    if must_exist:
        raise FileNotFoundError(
            f"Dataset root not found; checked: {', '.join(str(p) for p in _candidate_dataset_roots())}"
        )
    return root


def dataset_path(*parts: str, must_exist: bool = False) -> Path:
    """Return a path inside the dataset root."""
    root = resolve_dataset_root(must_exist=must_exist)
    path = root.joinpath(*parts)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    return path


def logs_path(*parts: str) -> Path:
    """Return a path inside the project logs directory."""
    return LOGS_ROOT.joinpath(*parts)


__all__ = [
    "PROJECT_ROOT",
    "LOGS_ROOT",
    "logs_path",
    "dataset_path",
    "list_dataset_roots",
    "resolve_dataset_root",
]
