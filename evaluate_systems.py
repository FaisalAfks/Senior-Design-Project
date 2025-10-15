#!/usr/bin/env python3
"""Offline evaluation script for recognition and anti-spoofing systems."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from BlazeFace import BlazeFaceService
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService
from main import DEFAULT_FACEBANK, DEFAULT_SPOOF_WEIGHTS, DEFAULT_WEIGHTS
from utils.device import select_device
from utils.verification import evaluate_frame


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv"}


@dataclass(frozen=True)
class Sample:
    path: Path
    media_type: str
    category: str
    person: str
    liveness: str
    label_recognition: bool
    label_spoof: bool


@dataclass(frozen=True)
class SampleResult:
    sample: Sample
    recognition_score: Optional[float]
    recognition_identity: Optional[str]
    spoof_score: Optional[float]
    frames_examined: int
    detections: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dataset against recognition and anti-spoofing systems.")
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"), help="Root directory containing Known/Unknown datasets.")
    parser.add_argument("--device", default="cpu", help="Torch device string (cpu, cuda, cuda:0, ...).")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to MobileFaceNet weights.")
    parser.add_argument("--facebank", type=Path, default=DEFAULT_FACEBANK, help="Path to facebank directory.")
    parser.add_argument("--spoof-weights", type=Path, default=DEFAULT_SPOOF_WEIGHTS, help="Path to DeePixBiS weights.")
    parser.add_argument("--detector-thr", type=float, default=0.7, help="Minimum BlazeFace detection confidence.")
    parser.add_argument("--max-video-frames", type=int, default=12, help="Maximum number of frames to sample per video.")
    parser.add_argument("--recognizer-thresholds", default="0.5,0.6,0.7,0.8,0.85,0.9", help="Comma-separated identity score thresholds (0-1).")
    parser.add_argument("--spoof-thresholds", default="0.5,0.6,0.7,0.8,0.85,0.86,0.87,0.88,0.89,0.9", help="Comma-separated spoof score thresholds.")
    parser.add_argument("--output-recognizer", type=Path, default=Path("recognizer_metrics.json"), help="Output file for recognizer confusion matrices.")
    parser.add_argument("--output-spoof", type=Path, default=Path("spoof_metrics.json"), help="Output file for anti-spoof confusion matrices.")
    return parser.parse_args()


def parse_thresholds(text: str) -> List[float]:
    thresholds: List[float] = []
    for chunk in text.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        thresholds.append(float(stripped))
    return sorted(set(thresholds))


def gather_samples(dataset_root: Path) -> List[Sample]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    samples: List[Sample] = []
    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in IMAGE_EXTS and suffix not in VIDEO_EXTS:
            continue

        try:
            relative = file_path.relative_to(dataset_root)
        except ValueError:
            continue

        parts = relative.parts
        if len(parts) < 4:
            # Expecting at least Category/Person/Liveness/Subdir/file
            continue

        category, person, liveness = parts[0], parts[1], parts[2]
        if category not in {"Known", "Unknown"}:
            continue
        if liveness not in {"Real", "Spoof"}:
            continue

        media_type = "image" if suffix in IMAGE_EXTS else "video"
        label_recognition = category == "Known"
        label_spoof = liveness == "Real"
        samples.append(
            Sample(
                path=file_path,
                media_type=media_type,
                category=category,
                person=person,
                liveness=liveness,
                label_recognition=label_recognition,
                label_spoof=label_spoof,
            )
        )
    samples.sort(key=lambda sample: (sample.category, sample.person, sample.liveness, sample.path.name))
    return samples


def sample_video_frames(path: Path, limit: int) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return frames

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        indices = np.linspace(0, max(total_frames - 1, 0), num=min(limit, total_frames), dtype=int)
        last_index = -1
        for frame_idx in indices:
            if frame_idx == last_index:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append(frame)
            last_index = frame_idx
    else:
        count = 0
        while count < limit:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(frame)
            count += 1

    capture.release()
    return frames


def evaluate_sample(
    sample: Sample,
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_service: DeePixBiSService,
    *,
    spoof_threshold_hint: float,
    max_video_frames: int,
) -> SampleResult:
    if sample.media_type == "image":
        frame = cv2.imread(str(sample.path))
        frames = [frame] if frame is not None else []
    else:
        frames = sample_video_frames(sample.path, max_video_frames)

    recognition_scores: List[float] = []
    spoof_scores: List[float] = []
    best_identity_score: Optional[float] = None
    best_identity_name: Optional[str] = None
    detections = 0

    for frame in frames:
        observation = evaluate_frame(frame, detector, recogniser, spoof_service, spoof_threshold_hint)
        if observation is None:
            continue
        detections += 1
        if observation.identity_score is not None:
            score_val = float(observation.identity_score)
            recognition_scores.append(score_val)
            if best_identity_score is None or score_val > best_identity_score:
                best_identity_score = score_val
                best_identity_name = observation.identity or "Unknown"
        if observation.spoof_score is not None:
            spoof_scores.append(float(observation.spoof_score))

    recognition_score = best_identity_score if best_identity_score is not None else None
    spoof_score = float(np.mean(spoof_scores)) if spoof_scores else None
    return SampleResult(
        sample=sample,
        recognition_score=recognition_score,
        recognition_identity=best_identity_name,
        spoof_score=spoof_score,
        frames_examined=len(frames),
        detections=detections,
    )


def _serialise_result(
    result: SampleResult,
    score: Optional[float],
    dataset_root: Path,
    *,
    expected_positive: bool,
    predicted_positive: bool,
    threshold: float,
    system: str,
) -> Dict[str, object]:
    sample = result.sample
    try:
        relative = sample.path.relative_to(dataset_root)
    except ValueError:
        relative = sample.path
    rel_str = str(relative).replace("\\", "/")
    if score is None:
        if system == "recognition":
            reason = "No recognition score produced (no aligned faces)."
        else:
            reason = "No spoof score produced (no crops or service disabled)."
    else:
        if system == "recognition":
            expected_label = "Known" if expected_positive else "Unknown"
            predicted_label = "Known" if predicted_positive else "Unknown"
        else:
            expected_label = "Real" if expected_positive else "Spoof"
            predicted_label = "Real" if predicted_positive else "Spoof"
        if predicted_positive and not expected_positive:
            reason = f"Score {score:.3f} >= threshold {threshold:.3f}; expected {expected_label}, predicted {predicted_label}."
        elif (not predicted_positive) and expected_positive:
            reason = f"Score {score:.3f} < threshold {threshold:.3f}; expected {expected_label}, predicted {predicted_label}."
        else:
            reason = f"Score {score:.3f} vs threshold {threshold:.3f}; expected {expected_label}, predicted {predicted_label}."
        if system == "recognition" and result.recognition_identity:
            reason += f" Predicted identity: {result.recognition_identity}."
    payload = {
        "path": rel_str,
        "score": score,
        "reason": reason,
    }
    if system == "recognition" and result.recognition_identity:
        payload["identity"] = result.recognition_identity
    return payload


def compute_confusion(
    results: Sequence[SampleResult],
    thresholds: Sequence[float],
    *,
    label_attr: str,
    score_attr: str,
    dataset_root: Path,
    system: str,
) -> List[Dict[str, object]]:
    matrices: List[Dict[str, object]] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        false_positives: List[Dict[str, object]] = []
        false_negatives: List[Dict[str, object]] = []
        for result in results:
            label = getattr(result.sample, label_attr)
            score = getattr(result, score_attr)
            predicted_positive = score is not None and score >= threshold
            if label and predicted_positive:
                tp += 1
            elif label and not predicted_positive:
                fn += 1
                false_negatives.append(
                    _serialise_result(
                        result,
                        score,
                        dataset_root,
                        expected_positive=True,
                        predicted_positive=False,
                        threshold=threshold,
                        system=system,
                    )
                )
            elif not label and predicted_positive:
                fp += 1
                false_positives.append(
                    _serialise_result(
                        result,
                        score,
                        dataset_root,
                        expected_positive=False,
                        predicted_positive=True,
                        threshold=threshold,
                        system=system,
                    )
                )
            else:
                tn += 1

        metrics = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
        matrices.append(
            {
                "threshold": threshold,
                "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "metrics": metrics,
                "misclassified": {
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                },
            }
        )
    return matrices


def compute_metrics(*, tp: int, fp: int, tn: int, fn: int) -> Dict[str, Optional[float]]:
    total = tp + fp + tn + fn
    metrics: Dict[str, Optional[float]] = {}
    metrics["accuracy"] = (tp + tn) / total if total else None

    positive = tp + fn
    negative = tn + fp
    metrics["tpr"] = tp / positive if positive else None
    metrics["fnr"] = fn / positive if positive else None
    metrics["tnr"] = tn / negative if negative else None
    metrics["fpr"] = fp / negative if negative else None
    metrics["far"] = metrics["fpr"]

    precision_den = tp + fp
    metrics["precision"] = tp / precision_den if precision_den else None
    recall = metrics["tpr"]
    precision = metrics["precision"]
    if precision is not None and recall is not None and (precision + recall) > 0:
        metrics["f1"] = 2 * precision * recall / (precision + recall)
    else:
        metrics["f1"] = None
    return metrics


def summarise_results(results: Sequence[SampleResult]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total_samples": len(results),
        "recognition_positive": sum(1 for res in results if res.sample.label_recognition),
        "recognition_negative": sum(1 for res in results if not res.sample.label_recognition),
        "spoof_positive": sum(1 for res in results if res.sample.label_spoof),
        "spoof_negative": sum(1 for res in results if not res.sample.label_spoof),
        "detections": sum(res.detections for res in results),
        "frames_examined": sum(res.frames_examined for res in results),
    }
    return summary


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()

    recognition_thresholds = parse_thresholds(args.recognizer_thresholds)
    spoof_thresholds = parse_thresholds(args.spoof_thresholds)

    device = select_device(args.device)
    detector = BlazeFaceService(score_threshold=args.detector_thr, device=device)
    recogniser = MobileFaceNetService(
        weights_path=args.weights,
        facebank_dir=args.facebank,
        detector=detector.detector,
        device=device,
        recognition_threshold=0.0,
        tta=False,
        refresh_facebank=False,
    )
    spoof_service = DeePixBiSService(weights_path=args.spoof_weights, device=device)

    samples = gather_samples(args.dataset_root)
    if not samples:
        raise RuntimeError(f"No evaluable media found under {args.dataset_root}")

    print(f"Loaded {len(samples)} samples from {args.dataset_root}")
    results: List[SampleResult] = []
    for index, sample in enumerate(samples, start=1):
        result = evaluate_sample(
            sample,
            detector,
            recogniser,
            spoof_service,
            spoof_threshold_hint=max(spoof_thresholds) if spoof_thresholds else 0.5,
            max_video_frames=max(args.max_video_frames, 1),
        )
        results.append(result)
        rel_path = sample.path.relative_to(args.dataset_root)
        rec_display = "NA" if result.recognition_score is None else f"{result.recognition_score:.3f}"
        spoof_display = "NA" if result.spoof_score is None else f"{result.spoof_score:.3f}"
        print(
            f"[{index:03d}/{len(samples):03d}] {rel_path} "
            f"| frames={result.frames_examined} detections={result.detections} "
            f"| rec_score={rec_display} "
            f"| spoof_score={spoof_display}"
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    base_summary = summarise_results(results)
    base_summary["generated_at"] = timestamp
    base_summary["dataset_root"] = str(args.dataset_root)

    recognizer_payload = dict(base_summary)
    recognizer_payload["thresholds"] = recognition_thresholds
    recognizer_payload["confusion_matrices"] = compute_confusion(
        results,
        recognition_thresholds,
        label_attr="label_recognition",
        score_attr="recognition_score",
        dataset_root=args.dataset_root,
        system="recognition",
    )
    save_json(args.output_recognizer, recognizer_payload)
    print(f"Recognition metrics saved to {args.output_recognizer}")

    spoof_payload = dict(base_summary)
    spoof_payload["thresholds"] = spoof_thresholds
    spoof_payload["confusion_matrices"] = compute_confusion(
        results,
        spoof_thresholds,
        label_attr="label_spoof",
        score_attr="spoof_score",
        dataset_root=args.dataset_root,
        system="spoof",
    )
    save_json(args.output_spoof, spoof_payload)
    print(f"Spoofing metrics saved to {args.output_spoof}")


if __name__ == "__main__":
    main()
