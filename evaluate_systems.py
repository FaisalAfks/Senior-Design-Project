#!/usr/bin/env python3
"""Offline evaluation script for detector, recogniser, and anti-spoof models."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from BlazeFace import BlazeFaceService
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService
from main import DEFAULT_FACEBANK, DEFAULT_SPOOF_WEIGHTS, DEFAULT_WEIGHTS
from utils.device import select_device
from utils.paths import dataset_path, logs_path
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
    spoof_attack: str  # "print", "replay", or "unknown"


@dataclass(frozen=True)
class SampleResult:
    sample: Sample
    detection_score: Optional[float]
    recognition_score: Optional[float]
    recognition_identity: Optional[str]
    spoof_score: Optional[float]
    frames_examined: int
    detections: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector, recogniser, and anti-spoof models.")
    default_validation_root = dataset_path("Validation")
    default_testing_root = dataset_path("Testing")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_validation_root,
        help=f"Validation dataset root (default: {default_validation_root}).",
    )
    parser.add_argument(
        "--testing-root",
        type=Path,
        default=default_testing_root,
        help=f"Testing dataset root for final evaluation (default: {default_testing_root}).",
    )
    parser.add_argument("--device", default="cpu", help="Torch device string (cpu, cuda, cuda:0, ...).")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to MobileFaceNet weights.")
    parser.add_argument("--facebank", type=Path, default=DEFAULT_FACEBANK, help="Path to facebank directory.")
    parser.add_argument("--spoof-weights", type=Path, default=DEFAULT_SPOOF_WEIGHTS, help="Path to DeePixBiS weights.")
    parser.add_argument("--detector-thr", type=float, default=0.7, help="Minimum BlazeFace detection confidence.")
    parser.add_argument("--max-video-frames", type=int, default=15, help="Maximum number of frames to sample per video.")
    parser.add_argument("--detector-thresholds", default="0.5:0.9:0.05", help="Detection thresholds as comma list or range start:end:step (default 0.4:0.9:0.05).")
    parser.add_argument("--recognition-thresholds", default="0.5:0.9:0.02", help="Recognition thresholds as comma list or range start:end:step (default 0.5:0.9:0.05).")
    parser.add_argument("--spoof-thresholds", default="0.5:0.9:0.02", help="Spoof thresholds as comma list or range start:end:step (default 0.5:0.9:0.05).")
    default_summary_output = logs_path("evaluation_summary.json")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=default_summary_output,
        help=f"Output path for combined evaluation summary (default: {default_summary_output}).",
    )
    return parser.parse_args()


def parse_thresholds(raw: str) -> List[float]:
    thresholds: List[float] = []
    for chunk in raw.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        if ":" in stripped:
            parts = [part.strip() for part in stripped.split(":")]
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid threshold range specification: '{stripped}'")
            start = float(parts[0])
            stop = float(parts[1])
            if len(parts) == 3:
                step = float(parts[2])
            else:
                raise ValueError(
                    f"Threshold range requires start:end:step format (step omitted in '{stripped}')"
                )
            if step == 0:
                raise ValueError(f"Threshold step cannot be zero in '{stripped}'")
            value = start
            epsilon = abs(step) / 1_000_000
            if step > 0:
                while value <= stop + epsilon:
                    thresholds.append(round(value, 10))
                    value += step
            else:
                while value >= stop - epsilon:
                    thresholds.append(round(value, 10))
                    value += step
        else:
            thresholds.append(float(stripped))
    thresholds = sorted(set(thresholds))
    return thresholds


def gather_samples(dataset_root: Path) -> List[Sample]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    samples: List[Sample] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXTS and suffix not in VIDEO_EXTS:
            continue

        try:
            parts = path.relative_to(dataset_root).parts
        except ValueError:
            continue
        if len(parts) < 4:
            # Expecting Category/Person/Liveness/...
            continue

        category, person, liveness = parts[0], parts[1], parts[2]
        media_type = "image" if suffix in IMAGE_EXTS else "video"
        label_recognition = category == "Known"
        label_spoof = liveness == "Real"

        if liveness == "Real":
            spoof_attack = "real"
        elif media_type == "image":
            spoof_attack = "print"
        elif media_type == "video":
            spoof_attack = "replay"
        else:
            spoof_attack = "unknown"

        samples.append(
            Sample(
                path=path,
                media_type=media_type,
                category=category,
                person=person,
                liveness=liveness,
                label_recognition=label_recognition,
                label_spoof=label_spoof,
                spoof_attack=spoof_attack,
            )
        )
    samples.sort(key=lambda s: (s.category, s.person, s.liveness, s.path.name))
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

    processed_frames = [frame for frame in frames if frame is not None]
    detection_scores: List[float] = []
    recognition_scores: List[float] = []
    spoof_scores: List[float] = []
    best_identity: Optional[str] = None
    best_recognition: Optional[float] = None
    detections = 0

    for frame in processed_frames:
        observation = evaluate_frame(frame, detector, recogniser, spoof_service, spoof_threshold_hint)
        if observation is None:
            continue
        detections += 1
        if observation.detection is not None and observation.detection.score is not None:
            detection_scores.append(float(observation.detection.score))
        if observation.identity_score is not None:
            score_val = float(observation.identity_score)
            recognition_scores.append(score_val)
            if best_recognition is None or score_val > best_recognition:
                best_recognition = score_val
                best_identity = observation.identity or "Unknown"
        if observation.spoof_score is not None:
            spoof_scores.append(float(observation.spoof_score))

    detection_score = max(detection_scores) if detection_scores else None
    recognition_score = best_recognition
    spoof_score = float(np.mean(spoof_scores)) if spoof_scores else None

    return SampleResult(
        sample=sample,
        detection_score=detection_score,
        recognition_score=recognition_score,
        recognition_identity=best_identity,
        spoof_score=spoof_score,
        frames_examined=len(processed_frames),
        detections=detections,
    )


def evaluate_dataset(
    dataset_root: Path,
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_service: DeePixBiSService,
    *,
    spoof_threshold_hint: float,
    max_video_frames: int,
    dataset_label: str,
) -> List[SampleResult]:
    samples = gather_samples(dataset_root)
    if not samples:
        raise RuntimeError(f"No evaluable media found under {dataset_root}")

    print(f"[{dataset_label}] Loaded {len(samples)} samples from {dataset_root}")
    results: List[SampleResult] = []
    for idx, sample in enumerate(samples, start=1):
        result = evaluate_sample(
            sample,
            detector,
            recogniser,
            spoof_service,
            spoof_threshold_hint=spoof_threshold_hint,
            max_video_frames=max(max_video_frames, 1),
        )
        results.append(result)
        rel_path = sample.path.relative_to(dataset_root)
        det_display = "NA" if result.detection_score is None else f"{result.detection_score:.3f}"
        rec_display = "NA" if result.recognition_score is None else f"{result.recognition_score:.3f}"
        spoof_display = "NA" if result.spoof_score is None else f"{result.spoof_score:.3f}"
        print(
            f"[{dataset_label} {idx:03d}/{len(samples):03d}] {rel_path} "
            f"| det={det_display} rec={rec_display} spoof={spoof_display} "
            f"| frames={result.frames_examined} detections={result.detections}"
        )
    return results


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Optional[float]]:
    total = tp + fp + tn + fn
    metrics: Dict[str, Optional[float]] = {}
    metrics["accuracy"] = (tp + tn) / total if total else None

    positive = tp + fn
    negative = tn + fp
    metrics["tpr"] = tp / positive if positive else None
    metrics["fnr"] = fn / positive if positive else None
    metrics["tnr"] = tn / negative if negative else None
    metrics["fpr"] = fp / negative if negative else None

    denom = tp + fp
    metrics["precision"] = tp / denom if denom else None
    recall = metrics["tpr"]
    precision = metrics["precision"]
    if precision is not None and recall is not None and (precision + recall) > 0:
        metrics["f1"] = 2 * precision * recall / (precision + recall)
    else:
        metrics["f1"] = None
    fnr = metrics.get("fnr")
    fpr = metrics.get("fpr")
    if fnr is not None and fpr is not None:
        metrics["eer"] = (fnr + fpr) / 2.0
        metrics["eer_gap"] = abs(fnr - fpr)
    else:
        metrics["eer"] = None
        metrics["eer_gap"] = None
    return metrics


def make_confusion_matrices(
    results: Sequence[SampleResult],
    thresholds: Sequence[float],
    *,
    score_getter: Callable[[SampleResult], Optional[float]],
    label_getter: Callable[[SampleResult], bool],
    dataset_root: Path,
    positive_label: str,
    negative_label: str,
) -> List[Dict[str, object]]:
    matrices: List[Dict[str, object]] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        false_pos: List[Dict[str, object]] = []
        false_neg: List[Dict[str, object]] = []
        for result in results:
            score = score_getter(result)
            expected_positive = label_getter(result)
            predicted_positive = score is not None and score >= threshold

            if expected_positive and predicted_positive:
                tp += 1
            elif expected_positive and not predicted_positive:
                fn += 1
                false_neg.append(_misclassified_entry(result, dataset_root, score, threshold, expected_positive, positive_label, negative_label))
            elif (not expected_positive) and predicted_positive:
                fp += 1
                false_pos.append(_misclassified_entry(result, dataset_root, score, threshold, expected_positive, positive_label, negative_label))
            else:
                tn += 1
        metrics = compute_metrics(tp, fp, tn, fn)
        matrices.append(
            {
                "threshold": float(threshold),
                "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "metrics": metrics,
                "misclassified": {
                    "false_positives": false_pos,
                    "false_negatives": false_neg,
                },
            }
        )
    return matrices


def _misclassified_entry(
    result: SampleResult,
    dataset_root: Path,
    score: Optional[float],
    threshold: float,
    expected_positive: bool,
    positive_label: str,
    negative_label: str,
) -> Dict[str, object]:
    try:
        relative = result.sample.path.relative_to(dataset_root)
    except ValueError:
        relative = result.sample.path
    expected = positive_label if expected_positive else negative_label
    predicted = positive_label if score is not None and score >= threshold else negative_label
    if score is None:
        reason = f"No score; expected {expected}, predicted {predicted}."
    else:
        reason = f"Score {score:.3f} vs threshold {threshold:.3f}; expected {expected}, predicted {predicted}."
    entry: Dict[str, object] = {
        "path": str(relative).replace("\\", "/"),
        "score": score,
        "reason": reason,
    }
    if result.recognition_identity:
        entry["identity"] = result.recognition_identity
    return entry


def select_best_by_accuracy(conf_matrices: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    best_entry: Optional[Dict[str, object]] = None
    best_accuracy: Optional[float] = None
    for entry in conf_matrices:
        accuracy = entry.get("metrics", {}).get("accuracy")
        if accuracy is None:
            continue
        if (
            best_entry is None
            or accuracy > (best_accuracy or float("-inf"))
            or (
                best_accuracy is not None
                and accuracy == best_accuracy
                and entry["threshold"] > best_entry["threshold"]
            )
        ):
            best_entry = entry
            best_accuracy = float(accuracy)
    return best_entry


def select_best_spoof_threshold(
    print_confusions: Sequence[Dict[str, object]],
    replay_confusions: Sequence[Dict[str, object]],
) -> Optional[Tuple[float, Dict[str, object]]]:
    if not print_confusions or not replay_confusions:
        return None

    print_lookup = {float(entry["threshold"]): entry for entry in print_confusions}
    replay_lookup = {float(entry["threshold"]): entry for entry in replay_confusions}
    shared_thresholds = sorted(set(print_lookup.keys()) & set(replay_lookup.keys()))
    best_threshold: Optional[float] = None
    best_accuracy: Optional[float] = None
    best_payload: Optional[Dict[str, object]] = None

    for threshold in shared_thresholds:
        print_counts = print_lookup[threshold]["counts"]
        replay_counts = replay_lookup[threshold]["counts"]
        total_tp = print_counts["tp"] + replay_counts["tp"]
        total_fp = print_counts["fp"] + replay_counts["fp"]
        total_tn = print_counts["tn"] + replay_counts["tn"]
        total_fn = print_counts["fn"] + replay_counts["fn"]
        total = total_tp + total_fp + total_tn + total_fn
        if total == 0:
            continue
        accuracy = (total_tp + total_tn) / total
        if (
            best_threshold is None
            or accuracy > (best_accuracy or float("-inf"))
            or (
                best_accuracy is not None
                and accuracy == best_accuracy
                and threshold > best_threshold
            )
        ):
            best_threshold = threshold
            best_accuracy = accuracy
            best_payload = {
                "combined_accuracy": accuracy,
                "print_metrics": print_lookup[threshold]["metrics"],
                "replay_metrics": replay_lookup[threshold]["metrics"],
            }
    if best_threshold is None or best_payload is None:
        return None
    return best_threshold, best_payload


def detection_label(sample: Sample) -> bool:
    liveness_lower = sample.liveness.lower()
    return liveness_lower not in {"background", "negative", "noface"}


def recognition_label(sample: Sample) -> bool:
    return sample.label_recognition


def spoof_print_filter(result: SampleResult) -> bool:
    return result.sample.media_type == "image"


def spoof_replay_filter(result: SampleResult) -> bool:
    return result.sample.media_type == "video"


def spoof_label(sample: Sample) -> bool:
    return sample.label_spoof


def build_pipeline_confusion_entry(data: Dict[str, object]) -> Optional[Dict[str, object]]:
    counts = data.get("counts")
    metrics = data.get("metrics")
    if counts is None or metrics is None:
        return None
    thresholds = data.get("thresholds", {})
    return {
        "threshold": float(thresholds.get("recognition") or 0.0),
        "counts": counts,
        "metrics": metrics,
        "misclassified": data.get("misclassified", {}),
        "details": {
            "thresholds": thresholds,
            "use_spoof": data.get("use_spoof"),
        },
    }


def evaluate_pipeline(
    results: Sequence[SampleResult],
    *,
    detection_threshold: float,
    recognition_threshold: float,
    spoof_threshold: Optional[float],
    use_spoof: bool,
    dataset_root: Path,
) -> Dict[str, object]:
    tp = fp = tn = fn = 0
    false_pos: List[Dict[str, object]] = []
    false_neg: List[Dict[str, object]] = []

    for result in results:
        label_positive = result.sample.category == "Known" and result.sample.liveness == "Real"
        detection_pass = result.detection_score is not None and result.detection_score >= detection_threshold
        recognition_pass = result.recognition_score is not None and result.recognition_score >= recognition_threshold
        if use_spoof:
            spoof_pass = result.spoof_score is not None and spoof_threshold is not None and result.spoof_score >= spoof_threshold
        else:
            spoof_pass = True

        predicted_positive = detection_pass and recognition_pass and spoof_pass
        if label_positive and predicted_positive:
            tp += 1
        elif label_positive and not predicted_positive:
            fn += 1
            false_neg.append(
                _pipeline_misclassification(
                    result,
                    dataset_root,
                    detection_threshold,
                    recognition_threshold,
                    spoof_threshold,
                    use_spoof,
                    predicted_positive=False,
                )
            )
        elif (not label_positive) and predicted_positive:
            fp += 1
            false_pos.append(
                _pipeline_misclassification(
                    result,
                    dataset_root,
                    detection_threshold,
                    recognition_threshold,
                    spoof_threshold,
                    use_spoof,
                    predicted_positive=True,
                )
            )
        else:
            tn += 1

    metrics = compute_metrics(tp, fp, tn, fn)
    return {
        "thresholds": {
            "detection": detection_threshold,
            "recognition": recognition_threshold,
            "spoof": spoof_threshold if use_spoof else None,
        },
        "use_spoof": use_spoof,
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": metrics,
        "misclassified": {"false_positives": false_pos, "false_negatives": false_neg},
    }


def _pipeline_misclassification(
    result: SampleResult,
    dataset_root: Path,
    detection_threshold: float,
    recognition_threshold: float,
    spoof_threshold: Optional[float],
    use_spoof: bool,
    *,
    predicted_positive: bool,
) -> Dict[str, object]:
    try:
        relative = result.sample.path.relative_to(dataset_root)
    except ValueError:
        relative = result.sample.path
    reasons: List[str] = []
    if result.detection_score is None:
        reasons.append("No detection score")
    else:
        reasons.append(f"det={result.detection_score:.3f} vs {detection_threshold:.3f}")
    if result.recognition_score is None:
        reasons.append("No recognition score")
    else:
        reasons.append(f"rec={result.recognition_score:.3f} vs {recognition_threshold:.3f}")
    if use_spoof:
        if result.spoof_score is None:
            reasons.append("No spoof score")
        else:
            reasons.append(f"spoof={result.spoof_score:.3f} vs {spoof_threshold:.3f}")
    return {
        "path": str(relative).replace("\\", "/"),
        "detection_score": result.detection_score,
        "recognition_score": result.recognition_score,
        "spoof_score": result.spoof_score,
        "outcome": "accepted" if predicted_positive else "rejected",
        "details": "; ".join(reasons),
    }


def summarise_results(results: Sequence[SampleResult]) -> Dict[str, object]:
    return {
        "total_samples": len(results),
        "with_detection_scores": sum(1 for r in results if r.detection_score is not None),
        "with_recognition_scores": sum(1 for r in results if r.recognition_score is not None),
        "with_spoof_scores": sum(1 for r in results if r.spoof_score is not None),
        "detections": sum(r.detections for r in results),
        "frames_examined": sum(r.frames_examined for r in results),
    }


def filter_results(results: Iterable[SampleResult], predicate: Callable[[SampleResult], bool]) -> List[SampleResult]:
    return [result for result in results if predicate(result)]


def main() -> None:
    args = parse_args()

    detector_thresholds = parse_thresholds(args.detector_thresholds)
    recognition_thresholds = parse_thresholds(args.recognition_thresholds)
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

    validation_results = evaluate_dataset(
        args.dataset_root,
        detector,
        recogniser,
        spoof_service,
        spoof_threshold_hint=max(spoof_thresholds) if spoof_thresholds else 0.5,
        max_video_frames=args.max_video_frames,
        dataset_label="validation",
    )

    # Validation confusion matrices
    detection_confusions = make_confusion_matrices(
        validation_results,
        detector_thresholds,
        score_getter=lambda res: res.detection_score,
        label_getter=lambda res: detection_label(res.sample),
        dataset_root=args.dataset_root,
        positive_label="face",
        negative_label="no_face",
    )
    recognition_confusions = make_confusion_matrices(
        validation_results,
        recognition_thresholds,
        score_getter=lambda res: res.recognition_score,
        label_getter=lambda res: recognition_label(res.sample),
        dataset_root=args.dataset_root,
        positive_label="known",
        negative_label="unknown",
    )

    validation_print_results = filter_results(validation_results, spoof_print_filter)
    validation_replay_results = filter_results(validation_results, spoof_replay_filter)
    spoof_print_confusions = make_confusion_matrices(
        validation_print_results,
        spoof_thresholds,
        score_getter=lambda res: res.spoof_score,
        label_getter=lambda res: spoof_label(res.sample),
        dataset_root=args.dataset_root,
        positive_label="real",
        negative_label="spoof",
    )
    spoof_replay_confusions = make_confusion_matrices(
        validation_replay_results,
        spoof_thresholds,
        score_getter=lambda res: res.spoof_score,
        label_getter=lambda res: spoof_label(res.sample),
        dataset_root=args.dataset_root,
        positive_label="real",
        negative_label="spoof",
    )

    detection_best = select_best_by_accuracy(detection_confusions)
    recognition_best = select_best_by_accuracy(recognition_confusions)
    spoof_best = select_best_spoof_threshold(spoof_print_confusions, spoof_replay_confusions)
    spoof_print_best = select_best_by_accuracy(spoof_print_confusions) if spoof_print_confusions else None
    spoof_replay_best = select_best_by_accuracy(spoof_replay_confusions) if spoof_replay_confusions else None

    if detection_best is None:
        raise RuntimeError("Unable to determine detector threshold.")
    if recognition_best is None:
        raise RuntimeError("Unable to determine recognition threshold.")
    if spoof_best is None:
        raise RuntimeError("Unable to determine spoof threshold from validation results.")

    detection_threshold = float(detection_best["threshold"])
    recognition_threshold = float(recognition_best["threshold"])
    spoof_threshold = float(spoof_best[0])

    print(f"Selected detector threshold: {detection_threshold:.3f} (accuracy={detection_best['metrics'].get('accuracy')})")
    print(f"Selected recognition threshold: {recognition_threshold:.3f} (accuracy={recognition_best['metrics'].get('accuracy')})")
    print(
        "Selected spoof threshold: "
        f"{spoof_threshold:.3f} (combined accuracy={spoof_best[1].get('combined_accuracy')})"
    )

    testing_results: Optional[List[SampleResult]] = None
    if args.testing_root.exists():
        testing_results = evaluate_dataset(
            args.testing_root,
            detector,
            recogniser,
            spoof_service,
            spoof_threshold_hint=spoof_threshold,
            max_video_frames=args.max_video_frames,
            dataset_label="testing",
        )
    else:
        print(f"[testing] Dataset root not found: {args.testing_root} (skipping testing evaluation)")

    pipeline_validation_with = evaluate_pipeline(
        validation_results,
        detection_threshold=detection_threshold,
        recognition_threshold=recognition_threshold,
        spoof_threshold=spoof_threshold,
        use_spoof=True,
        dataset_root=args.dataset_root,
    )
    pipeline_validation_without = evaluate_pipeline(
        validation_results,
        detection_threshold=detection_threshold,
        recognition_threshold=recognition_threshold,
        spoof_threshold=None,
        use_spoof=False,
        dataset_root=args.dataset_root,
    )

    if testing_results is not None:
        pipeline_testing_with = evaluate_pipeline(
            testing_results,
            detection_threshold=detection_threshold,
            recognition_threshold=recognition_threshold,
            spoof_threshold=spoof_threshold,
            use_spoof=True,
            dataset_root=args.testing_root,
        )
        pipeline_testing_without = evaluate_pipeline(
            testing_results,
            detection_threshold=detection_threshold,
            recognition_threshold=recognition_threshold,
            spoof_threshold=None,
            use_spoof=False,
            dataset_root=args.testing_root,
        )
    else:
        pipeline_testing_with = {
            "status": "skipped",
            "reason": f"Dataset root not found: {args.testing_root}",
        }
        pipeline_testing_without = {
            "status": "skipped",
            "reason": f"Dataset root not found: {args.testing_root}",
        }

    timestamp = datetime.now(timezone.utc).isoformat()
    summary: Dict[str, object] = {
        "generated_at": timestamp,
        "validation": {
            "dataset_root": str(args.dataset_root),
            "overview": summarise_results(validation_results),
            "detector": {
                "thresholds": detector_thresholds,
                "confusion_matrices": detection_confusions,
                "selected_threshold": detection_threshold,
            },
            "recognition": {
                "thresholds": recognition_thresholds,
                "confusion_matrices": recognition_confusions,
                "selected_threshold": recognition_threshold,
            },
            "anti_spoof": {
                "thresholds": spoof_thresholds,
                "print_confusion": spoof_print_confusions,
                "replay_confusion": spoof_replay_confusions,
                "selected_threshold": spoof_threshold,
                "selection_details": spoof_best[1],
            },
            "pipeline": {
                "with_anti_spoof": pipeline_validation_with,
                "without_anti_spoof": pipeline_validation_without,
            },
        },
        "testing": {
            "dataset_root": str(args.testing_root),
            "overview": summarise_results(testing_results) if testing_results is not None else None,
            "pipeline": {
                "with_anti_spoof": pipeline_testing_with,
                "without_anti_spoof": pipeline_testing_without,
            },
        },
    }

    validation_total = len(validation_results)
    validation_print_total = len(validation_print_results)
    validation_replay_total = len(validation_replay_results)
    testing_total = len(testing_results) if testing_results is not None else 0

    plots_config: Dict[str, object] = {}
    if detection_confusions:
        plots_config["detector_accuracy"] = {
            "type": "accuracy",
            "title": (
                f"Detector Accuracy | best thr {detection_threshold:.3f} | "
                f"samples {validation_total}"
            ),
            "xlabel": "Detection score threshold",
            "series": [
                {"label": "Detector", "entries": detection_confusions},
            ],
        }
    if recognition_confusions:
        plots_config["recognition_accuracy"] = {
            "type": "accuracy",
            "title": (
                f"Recognition Accuracy | best thr {recognition_threshold:.3f} | "
                f"samples {validation_total}"
            ),
            "xlabel": "Recognition score threshold",
            "series": [
                {"label": "Recognition", "entries": recognition_confusions},
            ],
        }
    if spoof_print_confusions:
        best_print_thr = (
            float(spoof_print_best["threshold"]) if spoof_print_best else spoof_threshold
        )
        plots_config["spoof_print_accuracy"] = {
            "type": "accuracy",
            "title": (
                "Anti-spoof Accuracy (print) | "
                f"best thr {best_print_thr:.3f} | samples {validation_print_total}"
            ),
            "xlabel": "Spoof score threshold",
            "series": [
                {"label": "Print attack", "entries": spoof_print_confusions},
            ],
        }
    if spoof_replay_confusions:
        best_replay_thr = (
            float(spoof_replay_best["threshold"]) if spoof_replay_best else spoof_threshold
        )
        plots_config["spoof_replay_accuracy"] = {
            "type": "accuracy",
            "title": (
                "Anti-spoof Accuracy (replay) | "
                f"best thr {best_replay_thr:.3f} | samples {validation_replay_total}"
            ),
            "xlabel": "Spoof score threshold",
            "series": [
                {"label": "Replay attack", "entries": spoof_replay_confusions},
            ],
        }
    if detection_confusions:
        plots_config["detector_confusion_validation"] = {
            "type": "confusion",
            "dataset_root": str(args.dataset_root),
            "confusion_matrices": detection_confusions,
            "default_threshold": detection_threshold,
            "labels": ["Face", "No Face"],
            "title": (
                f"Detector Validation | thr {detection_threshold:.3f} | "
                f"samples {validation_total}"
            ),
        }
    if recognition_confusions:
        plots_config["recognition_confusion_validation"] = {
            "type": "confusion",
            "dataset_root": str(args.dataset_root),
            "confusion_matrices": recognition_confusions,
            "default_threshold": recognition_threshold,
            "labels": ["Known", "Unknown"],
            "title": (
                f"Recognition Validation | thr {recognition_threshold:.3f} | "
                f"samples {validation_total}"
            ),
        }
    if spoof_print_confusions:
        plots_config["spoof_print_confusion_validation"] = {
            "type": "confusion",
            "dataset_root": str(args.dataset_root),
            "confusion_matrices": spoof_print_confusions,
            "default_threshold": best_print_thr,
            "labels": ["Real", "Spoof"],
            "title": (
                "Anti-spoof Validation (print) | "
                f"thr {best_print_thr:.3f} | samples {validation_print_total}"
            ),
        }
    if spoof_replay_confusions:
        plots_config["spoof_replay_confusion_validation"] = {
            "type": "confusion",
            "dataset_root": str(args.dataset_root),
            "confusion_matrices": spoof_replay_confusions,
            "default_threshold": best_replay_thr,
            "labels": ["Real", "Spoof"],
            "title": (
                "Anti-spoof Validation (replay) | "
                f"thr {best_replay_thr:.3f} | samples {validation_replay_total}"
            ),
        }
    pipeline_val_with_plot = build_pipeline_confusion_entry(pipeline_validation_with)
    if pipeline_val_with_plot:
        plots_config["pipeline_confusion_validation_with"] = {
            "type": "confusion",
            "dataset_root": str(args.dataset_root),
            "confusion_matrices": [pipeline_val_with_plot],
            "default_threshold": pipeline_val_with_plot["threshold"],
            "labels": ["Accept", "Reject"],
            "title": (
                "System Validation with Anti-spoofing | "
                f"det {detection_threshold:.3f} | "
                f"rec {recognition_threshold:.3f} | "
                f"spoof {spoof_threshold:.3f} | "
                f"samples {validation_total}"
            ),
        }
    pipeline_val_without_plot = build_pipeline_confusion_entry(pipeline_validation_without)
    if pipeline_val_without_plot:
        plots_config["pipeline_confusion_validation_without"] = {
            "type": "confusion",
            "dataset_root": str(args.dataset_root),
            "confusion_matrices": [pipeline_val_without_plot],
            "default_threshold": pipeline_val_without_plot["threshold"],
            "labels": ["Accept", "Reject"],
            "title": (
                "System Validation without Anti-spoofing | "
                f"det {detection_threshold:.3f} | "
                f"rec {recognition_threshold:.3f} | "
                f"samples {validation_total}"
            ),
        }
    pipeline_test_with_plot = (
        build_pipeline_confusion_entry(pipeline_testing_with) if isinstance(pipeline_testing_with, dict) else None
    )
    if pipeline_test_with_plot:
        plots_config["pipeline_confusion_testing_with"] = {
            "type": "confusion",
            "dataset_root": str(args.testing_root),
            "confusion_matrices": [pipeline_test_with_plot],
            "default_threshold": pipeline_test_with_plot["threshold"],
            "labels": ["Accept", "Reject"],
            "title": (
                "System Testing with Anti-spoofing | "
                f"det {detection_threshold:.3f} | "
                f"rec {recognition_threshold:.3f} | "
                f"spoof {spoof_threshold:.3f} | "
                f"samples {testing_total}"
            ),
        }
    pipeline_test_without_plot = (
        build_pipeline_confusion_entry(pipeline_testing_without) if isinstance(pipeline_testing_without, dict) else None
    )
    if pipeline_test_without_plot:
        plots_config["pipeline_confusion_testing_without"] = {
            "type": "confusion",
            "dataset_root": str(args.testing_root),
            "confusion_matrices": [pipeline_test_without_plot],
            "default_threshold": pipeline_test_without_plot["threshold"],
            "labels": ["Accept", "Reject"],
            "title": (
                "System Testing without Anti-spoofing | "
                f"det {detection_threshold:.3f} | "
                f"rec {recognition_threshold:.3f} | "
                f"samples {testing_total}"
            ),
        }
    if plots_config:
        summary["plots"] = plots_config

    summary_output = args.summary_output
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Summary saved to {summary_output}")


if __name__ == "__main__":
    main()
