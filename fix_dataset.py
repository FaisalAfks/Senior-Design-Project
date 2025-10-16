#!/usr/bin/env python3
"""Normalise Dataset filenames and optionally downscale/convert media to standard formats."""
from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageOps
    try:
        LANCZOS = Image.Resampling.LANCZOS  # Pillow >= 9.1
    except AttributeError:  # pragma: no cover - fallback for older Pillow
        LANCZOS = Image.LANCZOS
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    Image = None
    ImageOps = None
    LANCZOS = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

try:
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cv2 = None

from utils.resolution import parse_max_size, scaled_dimensions

MEDIA_DIR_NAMES = {"Photos", "Videos"}
LIVENESS_NAMES = {"Real", "Spoof"}
PREFIX_MAP: Dict[Tuple[str, str], str] = {
    ("Real", "Photos"): "real_photo",
    ("Real", "Videos"): "real_video",
    ("Spoof", "Photos"): "spoof_photo",
    ("Spoof", "Videos"): "spoof_video",
}
IGNORED_FILENAMES = {"thumbs.db"}
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix Dataset filenames to follow the real/spoof naming scheme.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset") / "Validation",
        help="Root directory containing Known/Unknown data (default: Dataset/Validation).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename the files. Without this flag the script only prints the planned changes.",
    )
    parser.add_argument(
        "--photo-max-size",
        type=str,
        default=None,
        help="Optional max PHOTO resolution as WIDTHxHEIGHT (e.g. 1280x720). Larger photos are downscaled preserving aspect ratio.",
    )
    parser.add_argument(
        "--photo-format",
        type=str,
        default="jpg",
        help="Target image format for Photos media (e.g. jpg). Leave empty to skip conversion.",
    )
    parser.add_argument(
        "--video-format",
        type=str,
        default="",
        help="Target container for Videos media (e.g. mp4). Leave empty to skip conversion (default).",
    )
    parser.add_argument(
        "--photo-quality",
        type=int,
        default=95,
        help="JPEG/WebP quality to use when saving resized photos (default: 95).",
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


def plan_photo_resizes(
    dataset_root: Path,
    rename_map: Dict[Path, Path],
    max_width: Optional[int],
    max_height: Optional[int],
) -> List[Tuple[Path, Path, Tuple[int, int], Tuple[int, int]]]:
    if max_width is None and max_height is None:
        return []
    if PIL_IMPORT_ERROR:
        raise RuntimeError(
            "Photo resizing requested but Pillow is not installed. Install Pillow to enable this feature."
        ) from PIL_IMPORT_ERROR

    resize_tasks: List[Tuple[Path, Path, Tuple[int, int], Tuple[int, int]]] = []
    for media_dir in iter_media_directories(dataset_root):
        if media_dir.name != "Photos":
            continue
        for file_path in sorted(path for path in media_dir.iterdir() if path.is_file()):
            if file_path.name.lower() in IGNORED_FILENAMES:
                continue
            if file_path.suffix.lower() not in PHOTO_SUFFIXES:
                continue
            target_path = rename_map.get(file_path, file_path)
            try:
                with Image.open(file_path) as img:
                    img = ImageOps.exif_transpose(img)
                    width, height = img.size
            except Exception as exc:
                print(f"Failed to inspect {file_path}: {exc}", file=sys.stderr)
                continue

            new_width, new_height = scaled_dimensions(width, height, max_width, max_height)
            if new_width == width and new_height == height:
                continue

            resize_tasks.append((file_path, target_path, (width, height), (new_width, new_height)))
    return resize_tasks


def plan_photo_conversions(
    dataset_root: Path,
    rename_map: Dict[Path, Path],
    target_suffix: Optional[str],
) -> List[Tuple[Path, Path]]:
    if not target_suffix:
        return []
    conversions: List[Tuple[Path, Path]] = []
    for media_dir in iter_media_directories(dataset_root):
        if media_dir.name != "Photos":
            continue
        for file_path in sorted(path for path in media_dir.iterdir() if path.is_file()):
            if file_path.name.lower() in IGNORED_FILENAMES:
                continue
            final_path = rename_map.get(file_path, file_path)
            if final_path.suffix.lower() not in PHOTO_SUFFIXES:
                continue
            if final_path.suffix.lower() == target_suffix:
                continue
            conversions.append((final_path, final_path.with_suffix(target_suffix)))
    return conversions


def plan_video_conversions(
    dataset_root: Path,
    rename_map: Dict[Path, Path],
    target_suffix: Optional[str],
) -> List[Tuple[Path, Path]]:
    if not target_suffix:
        return []
    conversions: List[Tuple[Path, Path]] = []
    for media_dir in iter_media_directories(dataset_root):
        if media_dir.name != "Videos":
            continue
        for file_path in sorted(path for path in media_dir.iterdir() if path.is_file()):
            if file_path.name.lower() in IGNORED_FILENAMES:
                continue
            if file_path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            final_path = rename_map.get(file_path, file_path)
            if final_path.suffix.lower() == target_suffix:
                continue
            conversions.append((final_path, final_path.with_suffix(target_suffix)))
    return conversions


def resize_photos(
    tasks: List[Tuple[Path, Path, Tuple[int, int], Tuple[int, int]]],
    quality: int,
) -> None:
    assert Image is not None and ImageOps is not None and LANCZOS is not None  # pragma: no cover - safeguarded earlier

    for original_path, target_path, _, new_size in tasks:
        path = target_path if target_path.exists() else original_path
        try:
            with Image.open(path) as img:
                exif = img.getexif()
                img = ImageOps.exif_transpose(img)
                if img.size == new_size:
                    continue
                resized = img.resize(new_size, resample=LANCZOS)
                save_kwargs = {"quality": quality}
                if exif:
                    try:
                        exif[0x0112] = 1  # Orientation tag
                    except Exception:
                        pass
                    save_kwargs["exif"] = exif.tobytes()
                resized.save(path, **save_kwargs)
        except Exception as exc:
            print(f"Failed to resize {path}: {exc}", file=sys.stderr)


def convert_photos(
    conversions: Sequence[Tuple[Path, Path]],
    *,
    quality: int,
) -> None:
    if not conversions:
        return
    if PIL_IMPORT_ERROR:
        raise RuntimeError(
            "Photo conversion requested but Pillow is not installed. Install Pillow to enable this feature."
        ) from PIL_IMPORT_ERROR

    for current_path, target_path in conversions:
        source_path = current_path
        if not source_path.exists():
            print(f"Skipping missing photo: {source_path}", file=sys.stderr)
            continue
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source_path) as img:
                exif = img.getexif()
                img = ImageOps.exif_transpose(img).convert("RGB")
                suffix = target_path.suffix.lower()
                if suffix in {".jpg", ".jpeg"}:
                    save_format = "JPEG"
                    save_kwargs = {"quality": quality}
                elif suffix == ".png":
                    save_format = "PNG"
                    save_kwargs = {}
                else:
                    save_format = suffix.lstrip(".").upper() or "JPEG"
                    save_kwargs = {}
                if exif and save_format.upper() == "JPEG":
                    try:
                        exif[0x0112] = 1  # reset orientation
                        save_kwargs["exif"] = exif.tobytes()
                    except Exception:
                        pass
                tmp_path = target_path.with_name(f"{target_path.stem}.__tmp__{target_path.suffix}")
                img.save(tmp_path, format=save_format, **save_kwargs)
            if source_path != target_path and source_path.exists():
                try:
                    source_path.unlink()
                except Exception as exc:
                    print(f"Failed to remove original photo {source_path}: {exc}", file=sys.stderr)
            tmp_path.rename(target_path)
        except Exception as exc:
            print(f"Failed to convert photo {source_path}: {exc}", file=sys.stderr)


def convert_videos(
    conversions: Sequence[Tuple[Path, Path]],
) -> None:
    if not conversions:
        return
    if cv2 is None:
        raise RuntimeError(
            "Video conversion requested but OpenCV (cv2) is not installed. Install opencv-python to enable this feature."
        )

    for current_path, target_path in conversions:
        source_path = current_path
        if not source_path.exists():
            print(f"Skipping missing video: {source_path}", file=sys.stderr)
            continue
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            print(f"Failed to open video for conversion: {source_path}", file=sys.stderr)
            continue

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(f"{target_path.stem}.__tmp__{target_path.suffix}")

        writer = None
        wrote_frame = False
        try:
            if width > 0 and height > 0:
                writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None:
                    continue
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
                writer.write(frame)
                wrote_frame = True
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if not wrote_frame:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"No frames converted for {source_path}", file=sys.stderr)
            continue

        if source_path != target_path and source_path.exists():
            try:
                source_path.unlink()
            except Exception as exc:
                print(f"Failed to remove original video {source_path}: {exc}", file=sys.stderr)
        tmp_path.rename(target_path)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    try:
        max_width, max_height = parse_max_size(args.photo_max_size)
    except ValueError as exc:
        print(f"Invalid --photo-max-size value: {exc}", file=sys.stderr)
        sys.exit(1)

    photo_format = (args.photo_format or "").strip().lstrip(".").lower()
    photo_suffix = f".{photo_format}" if photo_format else None

    video_format = (args.video_format or "").strip().lstrip(".").lower()
    video_suffix = f".{video_format}" if video_format else None

    all_operations: List[Tuple[Path, Path]] = []
    for media_dir in iter_media_directories(dataset_root):
        operations = compute_operations(media_dir)
        if operations:
            all_operations.extend(operations)

    rename_map: Dict[Path, Path] = {src: dst for src, dst in all_operations}
    resize_tasks = plan_photo_resizes(dataset_root, rename_map, max_width, max_height)
    photo_conversions = plan_photo_conversions(dataset_root, rename_map, photo_suffix)
    video_conversions = plan_video_conversions(dataset_root, rename_map, video_suffix)

    if not all_operations and not resize_tasks and not photo_conversions and not video_conversions:
        print("All filenames, photo resolutions, and media formats already match the expected scheme. Nothing to do.")
        return

    if all_operations:
        print("Planned renames:")
        for src, dst in all_operations:
            rel_src = src.relative_to(dataset_root)
            rel_dst = dst.relative_to(dataset_root)
            print(f"  {rel_src} -> {rel_dst}")

    if resize_tasks:
        print("\nPhoto resize candidates:")
        for _, target_path, (width, height), (new_width, new_height) in resize_tasks:
            rel_path = target_path.relative_to(dataset_root)
            print(f"  {rel_path}: {width}x{height} -> {new_width}x{new_height}")

    if photo_conversions:
        print("\nPhoto format conversions:")
        for current_path, target_path in photo_conversions:
            rel_current = current_path.relative_to(dataset_root)
            rel_target = target_path.relative_to(dataset_root)
            print(f"  {rel_current} -> {rel_target}")

    if video_conversions:
        print("\nVideo format conversions:")
        for current_path, target_path in video_conversions:
            rel_current = current_path.relative_to(dataset_root)
            rel_target = target_path.relative_to(dataset_root)
            print(f"  {rel_current} -> {rel_target}")

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to perform these changes.")
        return

    if all_operations:
        execute_operations(all_operations)
        print("\nRenaming complete.")

    if resize_tasks:
        resize_photos(resize_tasks, quality=args.photo_quality)
        print("Photo resizing complete.")

    if photo_conversions:
        convert_photos(photo_conversions, quality=args.photo_quality)
        print("Photo format conversion complete.")

    if video_conversions:
        convert_videos(video_conversions)
        print("Video format conversion complete.")


if __name__ == "__main__":
    main()
