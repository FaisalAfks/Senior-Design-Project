"""Verification helpers: per-frame evaluation, aggregation, and overlays."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from BlazeFace import BlazeFaceService, Detection
from DeePixBis import DeePixBiSService
from MobileFaceNet import MobileFaceNetService, RecognitionResult
from utils.overlay import BannerStyle, PanelStyle, draw_center_banner, draw_text_panel


@dataclass
class FaceObservation:
    detection: Detection
    identity: Optional[str]
    identity_score: Optional[float]
    is_recognized: Optional[bool]
    spoof_score: Optional[float]
    is_real: Optional[bool]


class FaceEvaluator:
    """Shared helper to keep verification logic consistent across entry points."""

    def __init__(
        self,
        detector: BlazeFaceService,
        recogniser: MobileFaceNetService,
        spoof_service: Optional[DeePixBiSService],
        spoof_threshold: float,
    ) -> None:
        self.detector = detector
        self.recogniser = recogniser
        self.spoof_service = spoof_service
        self.spoof_threshold = float(spoof_threshold)

    def evaluate(
        self,
        frame_bgr: np.ndarray,
        *,
        collect_timings: bool = False,
    ) -> Tuple[Optional[FaceObservation], Dict[str, float]]:
        if collect_timings:
            return evaluate_frame_with_timing(
                frame_bgr,
                self.detector,
                self.recogniser,
                self.spoof_service,
                self.spoof_threshold,
            )
        observation = evaluate_frame(
            frame_bgr,
            self.detector,
            self.recogniser,
            self.spoof_service,
            self.spoof_threshold,
        )
        return observation, {}


def evaluate_frame_with_timing(
    frame_bgr,
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_service: Optional[DeePixBiSService],
    spoof_threshold: float,
) -> Tuple[Optional[FaceObservation], Dict[str, float]]:
    timings: Dict[str, float] = {"detector_ms": 0.0, "recognition_ms": 0.0, "spoof_ms": 0.0}
    t0 = time.perf_counter()
    detections = detector.detect(frame_bgr)
    timings["detector_ms"] = (time.perf_counter() - t0) * 1000.0
    if not detections:
        return None, timings

    detection = max(detections, key=lambda det: det.score * max(det.area(), 1.0))
    aligned = detector.detector.align_face(frame_bgr, detection)
    if aligned is None:
        return None, timings

    t1 = time.perf_counter()
    rec_results = recogniser.recognise_faces([aligned])
    timings["recognition_ms"] = (time.perf_counter() - t1) * 1000.0
    if not rec_results:
        return None, timings
    rec: RecognitionResult = rec_results[0]

    spoof_score = None
    is_real = None
    if spoof_service is not None:
        crop = detector.detector.crop_face(frame_bgr, detection, expand=0.15, output_size=(224, 224))
        if crop is not None:
            t2 = time.perf_counter()
            score = spoof_service.predict_scores([crop])[0]
            timings["spoof_ms"] = (time.perf_counter() - t2) * 1000.0
            spoof_score = float(score)
            is_real = spoof_score >= spoof_threshold

    return FaceObservation(
        detection=detection,
        identity=rec.name,
        identity_score=float(rec.confidence),
        is_recognized=rec.is_recognized,
        spoof_score=spoof_score,
        is_real=is_real,
    ), timings


def evaluate_frame(
    frame_bgr,
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_service: Optional[DeePixBiSService],
    spoof_threshold: float,
) -> Optional[FaceObservation]:
    obs, _ = evaluate_frame_with_timing(frame_bgr, detector, recogniser, spoof_service, spoof_threshold)
    return obs


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


SUMMARY_PANEL_STYLE = PanelStyle(
    bg_color=(10, 10, 10),
    alpha=0.65,
    padding=14,
    margin=20,
    font_scale=0.7,
    text_color=(235, 235, 235),
    title_color=(255, 255, 255),
    title_scale=0.85,
    title_thickness=2,
    thickness=2,
)
SUMMARY_MIN_MODE_STYLE = BannerStyle(
    bg_color=(0, 0, 0),
    text_color=(255, 255, 255),
    alpha=0.7,
    padding=14,
    font_scale=0.9,
    margin=24,
    thickness=2,
)
METRICS_PANEL_STYLE = PanelStyle(
    bg_color=(18, 20, 25),
    alpha=0.75,
    padding=10,
    margin=16,
    font_scale=0.6,
    text_color=(240, 240, 240),
    title_color=(180, 220, 255),
    title_scale=0.65,
    title_thickness=2,
    thickness=1,
)


def compose_final_display(
    frame,
    summary: Dict[str, object],
    *,
    show_spoof_score: bool,
    show_scores: bool = True,
    minimal_mode: bool = False,
) -> np.ndarray:
    annotated = frame.copy()
    status_text = "ACCESS GRANTED" if summary["accepted"] else "ACCESS DENIED"
    status_color = (0, 255, 0) if summary["accepted"] else (0, 0, 255)

    if minimal_mode:
        identity = summary["identity"] or "Unknown"
        text = f"{status_text} — {identity}"
        return draw_center_banner(
            annotated,
            text,
            position="bottom",
            style=replace(SUMMARY_MIN_MODE_STYLE, text_color=status_color),
        )

    lines: list[Tuple[str, Tuple[int, int, int]]] = [
        (f"Identity: {summary['identity']}", SUMMARY_PANEL_STYLE.text_color),
        (f"Frames analysed: {summary['frames_with_detections']}", SUMMARY_PANEL_STYLE.text_color),
    ]
    if show_scores:
        if summary["avg_identity_score"] is not None:
            identity_pct = summary["avg_identity_score"] * 100.0
            lines.append((f"Avg identity score: {identity_pct:.1f}%", SUMMARY_PANEL_STYLE.text_color))
        else:
            lines.append(("Avg identity score: --", SUMMARY_PANEL_STYLE.text_color))
        if show_spoof_score:
            if summary["avg_spoof_score"] is not None:
                spoof_pct = summary["avg_spoof_score"] * 100.0
                if summary.get("is_real") is True:
                    spoof_status = "REAL"
                elif summary.get("is_real") is False:
                    spoof_status = "SPOOF"
                else:
                    spoof_status = "UNKNOWN"
                lines.append((f"Avg spoof score: {spoof_pct:.1f}% ({spoof_status})", SUMMARY_PANEL_STYLE.text_color))
            else:
                lines.append(("Avg spoof score: --", SUMMARY_PANEL_STYLE.text_color))

    duration = summary.get("capture_duration")
    if duration is not None:
        lines.append((f"Capture duration: {duration:.2f}s", SUMMARY_PANEL_STYLE.text_color))

    panel_style = SUMMARY_PANEL_STYLE
    # draw panel
    annotated = draw_text_panel(
        annotated,
        lines,
        anchor="top-left",
        title=status_text,
        style=PanelStyle(
            font=panel_style.font,
            font_scale=panel_style.font_scale,
            text_color=panel_style.text_color,
            bg_color=panel_style.bg_color,
            alpha=panel_style.alpha,
            padding=panel_style.padding,
            margin=panel_style.margin,
            thickness=panel_style.thickness,
            line_spacing=panel_style.line_spacing,
            title_color=status_color,
            title_scale=panel_style.title_scale,
            title_thickness=panel_style.title_thickness,
        ),
    )

    annotated = draw_center_banner(
        annotated,
        "Press SPACE to continue or ESC to close",
        position="bottom",
        style=BannerStyle(bg_color=(30, 30, 30), alpha=0.5, font_scale=0.7, margin=16),
    )
    return annotated


def compose_direct_display(
    frame: np.ndarray,
    observation: Optional[FaceObservation],
    *,
    spoof_threshold: float,
    show_metrics: bool = True,
    minimal_mode: bool = False,
) -> np.ndarray:
    """Overlay per-frame result near the detected face.

    - Draws the detection box
    - Places the identity (and spoof status if available) above the head
    - Colors green when accepted, red otherwise
    """
    annotated = frame.copy()
    if observation is None:
        if minimal_mode:
            return annotated
        return draw_center_banner(
            annotated,
            "Detecting face...",
            position="top",
            style=BannerStyle(bg_color=(20, 20, 20), text_color=(0, 165, 255), alpha=0.6),
        )

    det = observation.detection
    x1, y1, x2, y2 = det.as_int_bbox()
    name = observation.identity or "Unknown"
    id_score = observation.identity_score
    spoof_score = observation.spoof_score
    is_real = observation.is_real
    recognized = bool(observation.is_recognized)

    # Acceptance aligns with the summary logic: recognized and not spoof (or unknown spoof)
    accepted = recognized and (is_real is not False)
    color = (0, 255, 0) if accepted else (0, 0, 255)

    # Draw bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Build label text
    parts: List[str] = [name]
    if not minimal_mode:
        if id_score is not None:
            parts.append(f"{id_score * 100.0:.1f}%")
        if spoof_score is not None:
            label = "REAL" if (spoof_score >= spoof_threshold) else "SPOOF"
            parts.append(f"{label} {spoof_score * 100.0:.1f}%")
    label_text = " ".join(parts)

    # Place text just above the head; clamp within frame
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = max(5, min(x1, annotated.shape[1] - tw - 5))
    text_y = max(20, y1 - 10)
    if label_text:
        annotated = draw_center_banner(
            annotated,
            label_text,
            position="top",
            style=BannerStyle(bg_color=(0, 0, 0), text_color=color, alpha=0.65, font_scale=0.7),
        )

    return annotated


def run_verification_phase(
    capture: cv2.VideoCapture,
    detector: BlazeFaceService,
    recogniser: MobileFaceNetService,
    spoof_service: Optional[DeePixBiSService],
    spoof_threshold: float,
    window_name: Optional[str],
    duration_limit: float,
    *,
    mode: str = "time",
    frame_limit: int = 30,
    frame_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    display_callback: Optional[Callable[[np.ndarray], None]] = None,
    poll_cancel: Optional[Callable[[], bool]] = None,
    collect_timings: bool = False,
    minimal_mode: bool = False,
) -> Tuple[List[FaceObservation], Optional[np.ndarray], float, Dict[str, float]]:
    observations: List[FaceObservation] = []
    last_frame: Optional[np.ndarray] = None
    duration_limit = max(0.0, min(duration_limit, 5.0))
    frame_limit = max(1, int(frame_limit))
    start_time = time.perf_counter()

    evaluator = FaceEvaluator(detector, recogniser, spoof_service, spoof_threshold)
    processed_frames = 0
    timing_sums = {"detector_ms": 0.0, "recognition_ms": 0.0, "spoof_ms": 0.0}
    timing_count = 0
    prev_frame_time = start_time

    while True:
        if poll_cancel is not None and poll_cancel():
            break
        now = time.perf_counter()
        if mode == "time":
            if now - start_time >= duration_limit:
                break
        else:  # frames mode
            if processed_frames >= frame_limit:
                break
        ok, frame = capture.read()
        if not ok:
            break
        if frame_transform is not None:
            frame = frame_transform(frame)
        last_frame = frame.copy()

        observation, timings = evaluator.evaluate(frame, collect_timings=collect_timings)
        if collect_timings:
            for key in timing_sums:
                timing_sums[key] += timings.get(key, 0.0)
            timing_count += 1
        if observation is not None:
            observations.append(observation)
        processed_frames += 1

        frame_now = time.perf_counter()
        frame_duration = max(1e-6, frame_now - prev_frame_time)
        prev_frame_time = frame_now

        overlay = frame.copy()
        if not minimal_mode:
            identity_scores = [obs.identity_score for obs in observations if obs.identity_score is not None]
            avg_identity = sum(identity_scores) / len(identity_scores) if identity_scores else None
            spoof_scores = [obs.spoof_score for obs in observations if obs.spoof_score is not None]
            avg_spoof = sum(spoof_scores) / len(spoof_scores) if spoof_scores else None
            info_lines = [
                ("Capturing verification frames...", (255, 255, 255)),
                (f"Frames collected: {len(observations)}", (0, 255, 0)),
            ]
            if mode == "time":
                elapsed = min(time.perf_counter() - start_time, duration_limit)
                info_lines.insert(1, (f"Elapsed: {elapsed:.2f}s / {duration_limit:.2f}s", (0, 255, 255)))
            else:
                info_lines.insert(1, (f"Frames: {processed_frames}/{frame_limit}", (0, 255, 255)))
            if avg_identity is not None:
                info_lines.append((f"Avg identity: {avg_identity * 100.0:.1f}%", (0, 200, 255)))
            if avg_spoof is not None:
                spoof_label = "REAL" if avg_spoof >= spoof_threshold else "SPOOF"
                info_lines.append((f"Avg spoof: {avg_spoof * 100.0:.1f}% ({spoof_label})", (0, 200, 255)))
            overlay = draw_text_panel(
                overlay,
                info_lines,
                anchor="top-left",
                title=None,
                style=PanelStyle(margin=20, padding=10, font_scale=0.65, line_spacing=4, alpha=0.55),
            )
        if display_callback is not None:
            display_frame = overlay if not minimal_mode else frame
            if minimal_mode and not observation:
                display_frame = draw_center_banner(
                    display_frame,
                    "Capturing verification frames...",
                    position="top",
                    style=BannerStyle(bg_color=(20, 20, 20), text_color=(255, 255, 0), alpha=0.7, margin=24),
                )
            display_callback(display_frame)
        elif window_name is not None:
            if not minimal_mode:
                cv2.imshow(window_name, overlay)
            else:
                cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
        if poll_cancel is not None and poll_cancel():
            break

    actual_total_time = time.perf_counter() - start_time
    metrics: Dict[str, float] = {}
    if processed_frames > 0 and actual_total_time > 0:
        metrics["avg_fps"] = processed_frames / actual_total_time
    if timing_count > 0:
        for key, total in timing_sums.items():
            metrics[f"avg_{key}"] = total / timing_count
    metrics["frames"] = processed_frames
    return observations, last_frame, actual_total_time, metrics
