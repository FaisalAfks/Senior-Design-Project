"""Verification helpers: per-frame evaluation, aggregation, and overlays."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

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
            score = spoof_service.predict_scores([crop])[0]
            spoof_score = float(score)
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
        if obs.is_recognized and obs.identity and obs.identity_score is not None:
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


def compose_final_display(
    frame,
    summary: Dict[str, object],
    *,
    show_spoof_score: bool,
) -> np.ndarray:
    annotated = frame.copy()
    status_text = "ACCESS GRANTED" if summary["accepted"] else "ACCESS DENIED"
    status_color = (0, 255, 0) if summary["accepted"] else (0, 0, 255)
    cv2.putText(annotated, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)

    detail_lines = [
        f"Identity: {summary['identity']}",
        f"Frames analysed: {summary['frames_with_detections']}",
    ]

    if summary["avg_identity_score"] is not None:
        identity_pct = summary["avg_identity_score"] * 100.0
        detail_lines.append(f"Avg identity score: {identity_pct:.1f}%")
    else:
        detail_lines.append("Avg identity score: --")

    if show_spoof_score:
        if summary["avg_spoof_score"] is not None:
            spoof_pct = summary["avg_spoof_score"] * 100.0
            if summary.get("is_real") is True:
                spoof_status = "REAL"
            elif summary.get("is_real") is False:
                spoof_status = "SPOOF"
            else:
                spoof_status = "UNKNOWN"
            detail_lines.append(f"Avg spoof score: {spoof_pct:.1f}% ({spoof_status})")
        else:
            detail_lines.append("Avg spoof score: --")

    duration = summary.get("capture_duration")
    if duration is not None:
        detail_lines.append(f"Capture duration: {duration:.2f}s")

    for idx, line in enumerate(detail_lines, start=1):
        cv2.putText(
            annotated,
            line,
            (20, 50 + idx * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    instruction = "Press SPACE to continue or ESC to close"
    text_size, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = max(20, (annotated.shape[1] - text_size[0]) // 2)
    text_y = annotated.shape[0] - 40
    cv2.putText(annotated, instruction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return annotated


def run_verification_phase(
    capture: cv2.VideoCapture,
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_service: Optional[DeePixBiSService],
    spoof_threshold: float,
    window_name: str,
    duration_limit: float,
) -> Tuple[List[FaceObservation], Optional[np.ndarray], float]:
    observations: List[FaceObservation] = []
    last_frame: Optional[np.ndarray] = None
    duration_limit = max(0.0, min(duration_limit, 5.0))
    start_time = time.perf_counter()

    while True:
        now = time.perf_counter()
        if now - start_time >= duration_limit:
            break
        ok, frame = capture.read()
        if not ok:
            break
        last_frame = frame.copy()

        observation = evaluate_frame(frame, detector, recogniser, spoof_service, spoof_threshold)
        if observation is not None:
            observations.append(observation)

        overlay = frame.copy()
        elapsed = min(time.perf_counter() - start_time, duration_limit)
        identity_scores = [obs.identity_score for obs in observations if obs.identity_score is not None]
        avg_identity = sum(identity_scores) / len(identity_scores) if identity_scores else None
        spoof_scores = [obs.spoof_score for obs in observations if obs.spoof_score is not None]
        avg_spoof = sum(spoof_scores) / len(spoof_scores) if spoof_scores else None

        cv2.putText(overlay, "Capturing verification frames...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(
            overlay,
            f"Elapsed: {elapsed:.2f}s / {duration_limit:.2f}s",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(overlay, f"Number of frames: {len(observations)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if avg_identity is not None:
            identity_pct = avg_identity * 100.0
            cv2.putText(
                overlay,
                f"Avg identity: {identity_pct:.1f}%",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
        if avg_spoof is not None:
            spoof_pct = avg_spoof * 100.0
            if avg_spoof >= spoof_threshold:
                spoof_label = "REAL"
            else:
                spoof_label = "SPOOF"
            cv2.putText(
                overlay,
                f"Avg spoof: {spoof_pct:.1f}% ({spoof_label})",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    total_time = min(time.perf_counter() - start_time, duration_limit)
    return observations, last_frame, total_time
