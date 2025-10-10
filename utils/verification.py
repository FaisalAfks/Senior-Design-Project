"""Verification helpers: per-frame evaluation, aggregation, and logging."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2

from BlazeFace import BlazeFaceService, Detection
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService, RecognitionResult


@dataclass
class FaceObservation:
    detection: Detection
    identity: Optional[str]
    identity_score: Optional[float]
    is_recognized: Optional[bool]
    spoof_score: Optional[float]
    is_real: Optional[bool]


def evaluate_frame(
    frame_bgr,
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_service: Optional[DeePixBiSService],
    spoof_threshold: float,
) -> Optional[FaceObservation]:
    detections = detector.detect(frame_bgr)
    if not detections:
        return None

    detection = max(detections, key=lambda det: det.score * max(det.area(), 1.0))
    aligned = detector.detector.align_face(frame_bgr, detection)
    if aligned is None:
        return None

    rec_results = recogniser.recognise_faces([aligned])
    if not rec_results:
        return None
    rec: RecognitionResult = rec_results[0]

    spoof_score = None
    is_real = None
    if spoof_service is not None:
        crop = detector.detector.crop_face(frame_bgr, detection, expand=0.15, output_size=(224, 224))
        if crop is not None:
            score = spoof_service.predict_scores([crop])
            print(score)
            spoof_score = float(score[0])
            is_real = spoof_score >= spoof_threshold

    return FaceObservation(
        detection=detection,
        identity=rec.name,
        identity_score=float(rec.confidence),
        is_recognized=rec.is_recognized,
        spoof_score=spoof_score,
        is_real=is_real,
    )


def aggregate_observations(
    observations: Sequence[FaceObservation],
    *,
    spoof_threshold: float,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "frames_with_detections": len(observations),
        "recognized": False,
        "identity": "Unknown",
        "avg_identity_score": None,
        "avg_spoof_score": None,
        "is_real": None,
        "accepted": False,
    }
    if not observations:
        return summary

    identity_scores: Dict[str, List[float]] = {}
    for obs in observations:
        if obs.is_recognized and obs.identity is not None and obs.identity_score is not None:
            identity_scores.setdefault(obs.identity, []).append(obs.identity_score)

    if identity_scores:
        best_identity = max(identity_scores.items(), key=lambda item: sum(item[1]) / len(item[1]))[0]
        scores = identity_scores[best_identity]
        summary["recognized"] = True
        summary["identity"] = best_identity
        summary["avg_identity_score"] = sum(scores) / len(scores)

    spoof_scores = [obs.spoof_score for obs in observations if obs.spoof_score is not None]
    if spoof_scores:
        avg_spoof = sum(spoof_scores) / len(spoof_scores)
        summary["avg_spoof_score"] = avg_spoof
        summary["is_real"] = avg_spoof >= spoof_threshold

    spoof_result = summary.get("is_real")
    summary["accepted"] = summary["recognized"] and (spoof_result is not False)
    return summary


def append_attendance_log(path: Path, entry: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        json.dump(entry, log_file)
        log_file.write("\n")


def compose_final_display(
    frame,
    observation: Optional[FaceObservation],
    summary: Dict[str, object],
    *,
    show_spoof_score: bool,
):
    annotated = frame.copy()
    status_text = "ACCESS GRANTED" if summary["accepted"] else "ACCESS DENIED"
    status_color = (0, 255, 0) if summary["accepted"] else (0, 0, 255)
    cv2.putText(annotated, status_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)

    detail_lines = [
        f"Identity: {summary['identity']}",
        f"Number of frames: {summary['frames_with_detections']}",
    ]
    if summary["avg_identity_score"] is not None:
        detail_lines.append(f"Avg identity score: {summary['avg_identity_score']:.1f}")
    else:
        detail_lines.append("Avg identity score: --")
    if summary["avg_spoof_score"] is not None:
        detail_lines.append(f"Avg spoof score: {summary['avg_spoof_score']:.2f}")
    duration = summary.get("capture_duration")
    if duration is not None:
        detail_lines.append(f"Capture duration: {duration:.2f}s")

    for idx, line in enumerate(detail_lines, start=1):
        cv2.putText(annotated, line, (20, 50 + idx * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    instruction = "Press SPACE to continue or ESC to close"
    text_size, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = max(20, (annotated.shape[1] - text_size[0]) // 2)
    text_y = annotated.shape[0] - 40
    cv2.putText(annotated, instruction, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return annotated
