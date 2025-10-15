#!/usr/bin/env python3
"""Populate and enrich the facebank with aligned faces from the curated dataset."""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_project_on_path() -> None:
    project_path = str(PROJECT_ROOT)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


_ensure_project_on_path()

from BlazeFace.detector import BlazeFaceDetector
from utils.orientation import (
    DEFAULT_ORIENTATION_CATEGORIES,
    OrientationFeatures,
    categorise_orientation,
    compute_orientation,
    describe_orientation,
    orientation_category_significance,
    orientation_distance,
)

DATASET_KNOWN = PROJECT_ROOT / "Dataset" / "Known"
FACEBANK_ROOT = PROJECT_ROOT / "Facebank"
VIDEO_EXTENSIONS: Sequence[str] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
PHOTO_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg", ".png")
VIDEO_CATEGORY_TARGETS: Tuple[str, ...] = tuple(DEFAULT_ORIENTATION_CATEGORIES)
FACEBANK_PRESERVE_SUFFIXES: Tuple[str, ...] = (".pth", ".npy")


def _clear_facebank_directory(facebank_root: Path, *, preserve_suffixes: Sequence[str]) -> Dict[str, List[str]]:
    summary: Dict[str, List[str]] = {
        "removed_dirs": [],
        "removed_files": [],
        "skipped_files": [],
        "errors": [],
    }
    if not facebank_root.exists():
        return summary

    for entry in sorted(facebank_root.iterdir()):
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
                summary["removed_dirs"].append(entry.name)
                continue
            if entry.suffix.lower() in preserve_suffixes:
                summary["skipped_files"].append(entry.name)
                continue
            entry.unlink()
            summary["removed_files"].append(entry.name)
        except Exception as exc:  # pragma: no cover - filesystem safety
            summary["errors"].append(f"{entry.name}: {exc}")
    return summary


def _iter_ext(paths: Iterable[Path], extensions: Iterable[str]) -> Iterable[Path]:
    lowered = {ext.lower() for ext in extensions}
    for path in paths:
        if path.suffix.lower() in lowered:
            yield path


def _has_facebank_image(person_dir: Path) -> bool:
    if not person_dir.exists():
        return False
    return any(_iter_ext(person_dir.iterdir(), PHOTO_EXTENSIONS))


def _next_facebank_index(person_dir: Path) -> int:
    prefix = "facebank_"
    max_index = 0
    if person_dir.exists():
        for file in person_dir.iterdir():
            if not file.is_file() or not file.name.startswith(prefix):
                continue
            parts = file.stem.split("_")
            if not parts:
                continue
            try:
                idx = int(parts[-1])
            except ValueError:
                continue
            max_index = max(max_index, idx)
    return max_index + 1
@dataclass
class FaceSample:
    face: np.ndarray
    orientation: OrientationFeatures
    score: float
    frame_id: Tuple[str, int]
    source_desc: str


def _prepare_face_sample(
    detector: BlazeFaceDetector,
    image: np.ndarray,
    frame_id: Tuple[str, int],
    source_desc: str,
) -> Tuple[Optional[FaceSample], Optional[str]]:
    detections = detector.detect(image)
    if not detections:
        return None, "no detections"

    detection = max(detections, key=lambda det: det.score * max(det.area(), 1.0))
    face = detector.align_face(image, detection, output_size=(112, 112))
    if face is None:
        face = detector.crop_face(image, detection, output_size=(112, 112))
    if face is None:
        return None, "alignment failed"

    orientation = compute_orientation(detection)
    sample = FaceSample(
        face=face,
        orientation=orientation,
        score=detection.score,
        frame_id=frame_id,
        source_desc=source_desc,
    )
    return sample, None


def _fallback_candidate_for_category(category: str, candidates: Sequence[FaceSample]) -> Optional[FaceSample]:
    scored: List[Tuple[float, float, FaceSample]] = []
    for sample in candidates:
        significance = orientation_category_significance(category, sample.orientation)
        if significance <= 0:
            continue
        scored.append((significance, sample.score, sample))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _select_photo_samples(
    samples: Sequence[FaceSample],
    *,
    desired: int,
    min_distance: float,
) -> Tuple[List[FaceSample], List[str]]:
    if not samples:
        return [], ["no usable photo faces"]

    sorted_samples = sorted(samples, key=lambda s: s.score, reverse=True)
    selected: List[FaceSample] = []
    issues: List[str] = []

    if sorted_samples:
        selected.append(sorted_samples[0])

    remaining = sorted_samples[1:]
    while len(selected) < min(desired, len(sorted_samples)) and remaining:
        best_candidate = None
        best_idx = -1
        best_distance = -1.0
        for idx, sample in enumerate(remaining):
            distance = min(orientation_distance(sample.orientation, ref.orientation) for ref in selected)
            if distance > best_distance:
                best_distance = distance
                best_candidate = sample
                best_idx = idx

        if best_candidate is None:
            break

        if best_distance < min_distance:
            issues.append(
                f"photo diversity below threshold (distance={best_distance:.3f})"
            )
            # Still accept to honour requested count.
            selected.append(best_candidate)
            if best_idx >= 0:
                remaining.pop(best_idx)
            break

        selected.append(best_candidate)
        if best_idx >= 0:
            remaining.pop(best_idx)

    if len(selected) < desired and remaining:
        # Not enough diverse samples; fill with top-scoring remainder.
        for sample in remaining:
            if len(selected) >= desired:
                break
            selected.append(sample)

    if len(selected) < desired:
        issues.append(f"only {len(selected)} photo samples available (requested {desired})")

    return selected[:desired], issues


def _augment_facebank_from_videos(
    detector: BlazeFaceDetector,
    person_dir: Path,
    facebank_dir: Path,
    *,
    max_faces: int,
    frame_step: int,
    yaw_thr: float,
    pitch_thr: float,
    min_score: float,
) -> Tuple[List[Tuple[Path, str, OrientationFeatures]], List[str]]:
    """Extract faces with targeted orientations (left/right/up/down) from videos."""
    videos_dir = person_dir / "Real" / "Videos"
    if not videos_dir.exists() or max_faces <= 0:
        return [], []

    category_targets = list(VIDEO_CATEGORY_TARGETS)
    desired_total = min(max_faces, len(category_targets))

    category_candidates: Dict[str, List[FaceSample]] = {cat: [] for cat in category_targets}
    all_candidates: List[FaceSample] = []
    issues: List[str] = []

    video_files = sorted(_iter_ext(videos_dir.iterdir(), VIDEO_EXTENSIONS), key=lambda p: p.name)
    for video_path in video_files:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            issues.append(f"{video_path.name} (failed to open)")
            continue

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(frame_step, 1) != 0:
                frame_index += 1
                continue

            sample, error = _prepare_face_sample(
                detector,
                frame,
                frame_id=(video_path.name, frame_index),
                source_desc=f"{video_path.name}#{frame_index}",
            )
            if sample is None:
                frame_index += 1
                continue

            if sample.score < min_score:
                frame_index += 1
                continue

            categories = categorise_orientation(sample.orientation, yaw_thr=yaw_thr, pitch_thr=pitch_thr)
            if not categories:
                frame_index += 1
                continue

            all_candidates.append(sample)
            for cat in categories:
                if cat in category_candidates:
                    category_candidates[cat].append(sample)

            frame_index += 1

        capture.release()

    selected_samples: List[Tuple[str, FaceSample]] = []
    used_ids: Set[Tuple[str, int]] = set()

    for cat in category_targets:
        candidates = sorted(category_candidates[cat], key=lambda s: s.score, reverse=True)
        if not candidates:
            fallback = _fallback_candidate_for_category(cat, all_candidates)
            if fallback is not None:
                candidates = [fallback]
        for sample in candidates:
            if sample.frame_id in used_ids:
                continue
            selected_samples.append((cat, sample))
            used_ids.add(sample.frame_id)
            if cat.startswith("yaw"):
                if abs(sample.orientation.yaw_offset) < yaw_thr:
                    issues.append(f"weak {cat.replace('_', ' ')} offset {sample.orientation.yaw_offset:+.2f}")
            elif cat.startswith("pitch"):
                if abs(sample.orientation.pitch_offset) < pitch_thr:
                    issues.append(f"weak {cat.replace('_', ' ')} offset {sample.orientation.pitch_offset:+.2f}")
            break
        else:
            issues.append(f"missing {cat.replace('_', ' ')} sample")

    # Fill remaining slots (if max_faces larger than categories) with diverse leftovers.
    if len(selected_samples) < max_faces:
        remaining_candidates = sorted(all_candidates, key=lambda s: s.score, reverse=True)
        for sample in remaining_candidates:
            if sample.frame_id in used_ids:
                continue
            selected_samples.append(("extra", sample))
            used_ids.add(sample.frame_id)
            if len(selected_samples) >= max_faces:
                break

    saved: List[Tuple[Path, str, OrientationFeatures]] = []
    for category, sample in selected_samples[:max_faces]:
        facebank_dir.mkdir(parents=True, exist_ok=True)
        next_index = _next_facebank_index(facebank_dir)
        output_path = facebank_dir / f"facebank_{next_index:02d}.jpg"
        if cv2.imwrite(str(output_path), sample.face):
            saved.append((output_path, category, sample.orientation))
        else:
            issues.append(f"{sample.source_desc} (failed to save {output_path.name})")

    return saved, issues
def _process_photos(
    detector: BlazeFaceDetector,
    person_dir: Path,
    facebank_dir: Path,
    *,
    desired_count: int,
    min_orientation_distance: float,
) -> Tuple[List[Tuple[Path, OrientationFeatures]], List[str]]:
    real_photos = person_dir / "Real" / "Photos"
    if not real_photos.exists():
        return [], ["missing Real/Photos"]

    candidates = sorted(_iter_ext(real_photos.iterdir(), PHOTO_EXTENSIONS), key=lambda path: path.name)
    if not candidates:
        return [], ["no photo files"]

    samples: List[FaceSample] = []
    issues: List[str] = []

    for idx, candidate in enumerate(candidates):
        image = cv2.imread(str(candidate))
        if image is None:
            issues.append(f"{candidate.name} (failed to read)")
            continue
        sample, error = _prepare_face_sample(
            detector,
            image,
            frame_id=(candidate.name, idx),
            source_desc=candidate.name,
        )
        if sample is None:
            if error:
                issues.append(f"{candidate.name} ({error})")
            continue
        samples.append(sample)

    selected, diversity_issues = _select_photo_samples(
        samples,
        desired=desired_count,
        min_distance=min_orientation_distance,
    )
    issues.extend(diversity_issues)

    saved: List[Tuple[Path, OrientationFeatures]] = []
    for sample in selected:
        facebank_dir.mkdir(parents=True, exist_ok=True)
        next_index = _next_facebank_index(facebank_dir)
        output_path = facebank_dir / f"facebank_{next_index:02d}.jpg"
        if cv2.imwrite(str(output_path), sample.face):
            saved.append((output_path, sample.orientation))
        else:
            issues.append(f"{sample.source_desc} (failed to save {output_path.name})")

    if not saved:
        if not issues:
            issues.append("no usable face samples from photos")
    return saved, issues


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate the facebank from the curated dataset.")
    parser.add_argument(
        "--skip-photo-population",
        action="store_true",
        help="Do not add the initial frontal photo if the facebank is empty.",
    )
    parser.add_argument(
        "--clear-facebank",
        action="store_true",
        help="Delete existing facebank person folders (and non-embedding files) before rebuilding.",
    )
    parser.add_argument(
        "--force-refresh-photos",
        action="store_true",
        help="Re-populate the frontal photo even if images already exist for the identity.",
    )
    parser.add_argument(
        "--no-video-augmentation",
        dest="augment_from_videos",
        action="store_false",
        help="Disable extracting tilted faces from the Real/Videos folder.",
    )
    parser.set_defaults(augment_from_videos=True)
    parser.add_argument(
        "--photo-count",
        type=int,
        default=2,
        help="Number of photo samples to save per identity.",
    )
    parser.add_argument(
        "--photo-distance-thr",
        type=float,
        default=0.18,
        help="Minimum orientation distance between photo samples to consider them distinct.",
    )
    parser.add_argument(
        "--max-video-faces",
        type=int,
        default=4,
        help="Maximum number of additional faces to collect per identity from videos.",
    )
    parser.add_argument(
        "--video-frame-step",
        type=int,
        default=5,
        help="Process every Nth frame when scanning the videos.",
    )
    parser.add_argument(
        "--min-yaw-offset",
        type=float,
        default=0.18,
        help="Minimum absolute nose offset (as a fraction of eye distance) to accept a sideways pose.",
    )
    parser.add_argument(
        "--min-video-score",
        type=float,
        default=0.70,
        help="Minimum BlazeFace detection score required for video frames.",
    )
    parser.add_argument(
        "--min-pitch-offset",
        type=float,
        default=0.10,
        help="Minimum absolute nose height offset (relative to bbox) to accept up/down poses from video.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    detector = BlazeFaceDetector()

    if args.clear_facebank:
        summary = _clear_facebank_directory(FACEBANK_ROOT, preserve_suffixes=FACEBANK_PRESERVE_SUFFIXES)
        removed_dirs = len(summary["removed_dirs"])
        removed_files = len(summary["removed_files"])
        print(f"Cleared facebank entries: {removed_dirs} directories, {removed_files} files.")
        if summary["skipped_files"]:
            print(f"Preserved files: {summary['skipped_files']}")
        if summary["errors"]:
            print(f"Errors during cleanup: {summary['errors']}")

    processed_photos: List[str] = []
    skipped_photos: List[str] = []
    photo_issues_log: List[str] = []
    failures: List[str] = []
    video_additions: List[str] = []
    augmentation_issues: List[str] = []

    for person_dir in sorted(DATASET_KNOWN.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        facebank_dir = FACEBANK_ROOT / name

        has_existing = _has_facebank_image(facebank_dir)
        photo_saved: List[Tuple[Path, OrientationFeatures]] = []
        photo_notes: List[str] = []
        photo_attempted = False

        if not args.skip_photo_population:
            if has_existing and not args.force_refresh_photos:
                skipped_photos.append(name)
            else:
                photo_attempted = True
                photo_saved, photo_notes = _process_photos(
                    detector,
                    person_dir,
                    facebank_dir,
                    desired_count=max(args.photo_count, 0),
                    min_orientation_distance=max(args.photo_distance_thr, 0.0),
                )
                if photo_saved:
                    labels = ", ".join(
                        describe_orientation(
                            orientation,
                            yaw_thr=args.min_yaw_offset,
                            pitch_thr=args.min_pitch_offset,
                        )
                        for _, orientation in photo_saved
                    )
                    processed_photos.append(f"{name} (+{len(photo_saved)} photos) [{labels}]")
                    desired_photos = max(args.photo_count, 0)
                    if desired_photos > 0 and len(photo_saved) < desired_photos:
                        failures.append(f"{name} (photo shortfall {len(photo_saved)}/{desired_photos})")
                elif photo_attempted:
                    failures.append(f"{name} (no photo samples)")
        for note in photo_notes:
            photo_issues_log.append(f"{name} {note}")

        video_saved: List[Tuple[Path, str, OrientationFeatures]] = []
        video_notes: List[str] = []
        if args.augment_from_videos:
            video_saved, video_notes = _augment_facebank_from_videos(
                detector,
                person_dir,
                facebank_dir,
                max_faces=max(args.max_video_faces, 0),
                frame_step=max(args.video_frame_step, 1),
                yaw_thr=args.min_yaw_offset,
                pitch_thr=args.min_pitch_offset,
                min_score=args.min_video_score,
            )
            if video_saved:
                labels = ", ".join(
                    f"{category}:{describe_orientation(orientation, yaw_thr=args.min_yaw_offset, pitch_thr=args.min_pitch_offset)}"
                    for _, category, orientation in video_saved
                )
                video_additions.append(f"{name} (+{len(video_saved)} videos) [{labels}]")
                expected_video = min(max(args.max_video_faces, 0), len(VIDEO_CATEGORY_TARGETS))
                if expected_video > 0 and len(video_saved) < expected_video:
                    failures.append(f"{name} (video shortfall {len(video_saved)}/{expected_video})")
            else:
                failures.append(f"{name} (no video samples)")
        missing_categories = [note for note in video_notes if note.startswith("missing ")]
        for note in video_notes:
            augmentation_issues.append(f"{name} {note}")
        if missing_categories:
            failures.append(f"{name} ({'; '.join(missing_categories)})")

    print(f"Photo additions: {processed_photos}")
    print(f"Skipped photos: {skipped_photos}")
    print(f"Photo issues: {photo_issues_log}")
    print(f"Failures: {failures}")
    if args.augment_from_videos:
        print(f"Video augmentations: {video_additions}")
        print(f"Video issues: {augmentation_issues}")


if __name__ == "__main__":
    main()
