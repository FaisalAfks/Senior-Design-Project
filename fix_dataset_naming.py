#!/usr/bin/env python3
"""Normalise Dataset media filenames to the real/spoof_photo/video_XX schema."""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MEDIA_DIR_NAMES = {"Photos", "Videos"}
LIVENESS_NAMES = {"Real", "Spoof"}
PREFIX_MAP: Dict[Tuple[str, str], str] = {
    ("Real", "Photos"): "real_photo",
    ("Real", "Videos"): "real_video",
    ("Spoof", "Photos"): "spoof_photo",
    ("Spoof", "Videos"): "spoof_video",
}
IGNORED_FILENAMES = {"thumbs.db"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix Dataset filenames to follow the real/spoof naming scheme.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset"),
        help="Root directory containing Known/Unknown data (default: Dataset).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename the files. Without this flag the script only prints the planned changes.",
    )
    return parser.parse_args()


def iter_media_directories(root: Path) -> Iterable[Path]:
    for media_dir in root.rglob("*"):
        if not media_dir.is_dir():
            continue
        if media_dir.name not in MEDIA_DIR_NAMES:
            continue
        if media_dir.parent.name not in LIVENESS_NAMES:
            continue
        yield media_dir


def compute_operations(media_dir: Path) -> List[Tuple[Path, Path]]:
    liveness = media_dir.parent.name
    media_type = media_dir.name
    prefix = PREFIX_MAP.get((liveness, media_type))
    if not prefix:
        return []

    files = sorted(path for path in media_dir.iterdir() if path.is_file())
    if not files:
        return []

    digits = max(2, len(str(len(files))))
    operations: List[Tuple[Path, Path]] = []
    index = 1
    for file_path in files:
        if file_path.name.lower() in IGNORED_FILENAMES:
            continue
        suffix = file_path.suffix.lower()
        if not suffix:
            suffix = ""
        target_name = f"{prefix}_{index:0{digits}d}{suffix}"
        index += 1
        if file_path.name == target_name:
            continue
        target_path = file_path.with_name(target_name)
        operations.append((file_path, target_path))
    return operations


def execute_operations(operations: List[Tuple[Path, Path]]) -> None:
    temp_ops: List[Tuple[Path, Path]] = []
    for src, dst in operations:
        temp_name = f".__tmp__{uuid.uuid4().hex}{src.suffix.lower()}"
        tmp_path = src.with_name(temp_name)
        while tmp_path.exists():
            temp_name = f".__tmp__{uuid.uuid4().hex}{src.suffix.lower()}"
            tmp_path = src.with_name(temp_name)
        src.rename(tmp_path)
        temp_ops.append((tmp_path, dst))

    for tmp_path, dst in temp_ops:
        if dst.exists():
            dst.unlink()
        tmp_path.rename(dst)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    all_operations: List[Tuple[Path, Path]] = []
    for media_dir in iter_media_directories(dataset_root):
        operations = compute_operations(media_dir)
        if operations:
            all_operations.extend(operations)

    if not all_operations:
        print("All filenames already match the expected scheme. Nothing to do.")
        return

    for src, dst in all_operations:
        rel_src = src.relative_to(dataset_root)
        rel_dst = dst.relative_to(dataset_root)
        print(f"{rel_src} -> {rel_dst}")

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to perform these renames.")
        return

    execute_operations(all_operations)
    print("\nRenaming complete.")


if __name__ == "__main__":
    main()

