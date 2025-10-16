#!/usr/bin/env python3
"""Plot metrics or Jetson power telemetry stored in JSON files."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

ACTIVITY_CANONICAL = {
    "initializing": "setup",
    "camera-initialization": "setup",
    "ready": "idle",
    "waiting": "idle",
    "guidance": "guidance",
    "verification": "verification",
    "processing": "processing",
    "terminating": "shutdown",
    "cleanup": "shutdown",
    "shutdown": "shutdown",
}

PHASE_COLORS = {
    "setup": "#1f77b4",  # blue
    "idle": "#2ca02c",  # green
    "guidance": "#ff7f0e",  # orange
    "verification": "#d62728",  # red
    "processing": "#9467bd",  # purple
    "shutdown": "#7f7f7f",  # gray
    "other": "#bcbd22",  # olive
}

MIN_SEGMENT_DURATION = 1.0  # seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a confusion matrix or Jetson power log from a JSON file.")
    parser.add_argument("metrics", type=Path, help="Path to metrics JSON or Jetson power log JSON.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Exact threshold to plot. If omitted, uses the --strategy heuristic.",
    )
    parser.add_argument(
        "--strategy",
        choices=["max-accuracy", "max-f1"],
        default="max-accuracy",
        help="Heuristic used to pick a threshold when --threshold is not supplied.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom title for the plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="If provided, save the figure to this file instead of displaying it.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution (dots per inch) when saving figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window (requires GUI backend).",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pick_entry(
    matrices: List[Dict[str, Any]],
    *,
    threshold: Optional[float],
    strategy: str,
) -> Dict[str, Any]:
    if threshold is not None:
        for entry in matrices:
            if abs(entry["threshold"] - threshold) < 1e-9:
                return entry
        available = ", ".join(str(round(m["threshold"], 6)) for m in matrices)
        raise ValueError(f"Threshold {threshold} not found. Available thresholds: {available}")

    def metric_value(entry: Dict[str, Any], key: str) -> float:
        value = entry.get("metrics", {}).get(key)
        return float(value) if value is not None else float("-inf")

    if strategy == "max-f1":
        key = "f1"
    else:
        key = "accuracy"
    return max(matrices, key=lambda entry: metric_value(entry, key))


def render_confusion_matrix(
    entry: Dict[str, Any],
    *,
    title: Optional[str],
    display_threshold: str,
    class_labels: tuple[str, str],
) -> plt.Figure:
    counts = entry["counts"]
    matrix = np.array(
        [
            [counts["tp"], counts["fn"]],
            [counts["fp"], counts["tn"]],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    pos_label, neg_label = class_labels
    ax.set_xticks([0, 1], labels=[pos_label, neg_label])
    ax.set_yticks([0, 1], labels=[pos_label, neg_label])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    default_title = f"Threshold = {display_threshold}"
    ax.set_title(title or default_title)

    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{int(value)}", ha="center", va="center", color="black", fontsize=10)

    metrics = entry.get("metrics", {})
    subtitle = ", ".join(
        f"{name.upper()}={metrics[name]:.3f}"
        for name in ("accuracy", "precision", "tpr", "fpr")
        if metrics.get(name) is not None
    )
    if subtitle:
        ax.text(
            0.5,
            -0.15,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def is_power_log(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        return False
    first = samples[0]
    if not isinstance(first, dict):
        return False
    return "total_power_w" in first


def render_power_plot(
    log_payload: Dict[str, Any],
    *,
    title: Optional[str],
) -> plt.Figure:
    samples: List[Dict[str, Any]] = [sample for sample in log_payload.get("samples", []) if isinstance(sample, dict)]
    if not samples:
        raise RuntimeError("No samples found in power log.")

    metadata = log_payload.get("metadata") if isinstance(log_payload.get("metadata"), dict) else {}
    start_timestamp = metadata.get("start_timestamp")

    times: List[float] = []
    power_values: List[float] = []
    for sample in samples:
        total_power = sample.get("total_power_w")
        if total_power is None:
            continue
        elapsed = sample.get("elapsed_s")
        if elapsed is None and start_timestamp is not None:
            timestamp = sample.get("timestamp")
            if timestamp is not None:
                elapsed = float(timestamp) - float(start_timestamp)
        elapsed = float(elapsed) if elapsed is not None else float(sample.get("timestamp", 0.0))
        times.append(max(0.0, elapsed))
        power_values.append(float(total_power))

    if not power_values:
        raise RuntimeError("Power log does not contain usable samples.")

    fig, ax = plt.subplots()
    ax.plot(times, power_values, marker="o", linewidth=1.5, color="tab:blue")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Total power (W)")

    if title is None:
        avg_power = sum(power_values) / len(power_values)
        min_power = min(power_values)
        max_power = max(power_values)
        default_title = "Jetson Power Log"
        default_title += f" · avg={avg_power:.2f}W"
        default_title += f" · min={min_power:.2f}W"
        default_title += f" · max={max_power:.2f}W"
        default_title += f" · points={len(power_values)}"
        ax.set_title(default_title)
    else:
        ax.set_title(title)

    segments = _build_activity_segments(metadata, samples, times)
    if segments:
        _shade_activity_regions(ax, segments)

    fig.tight_layout()
    return fig


def _build_activity_segments(
    metadata: Dict[str, Any],
    samples: List[Dict[str, Any]],
    sampled_times: List[float],
) -> List[Tuple[float, float, str]]:
    events_raw = metadata.get("activity_events")
    if not isinstance(events_raw, list):
        return []

    events: List[Tuple[float, str]] = []
    for item in events_raw:
        if not isinstance(item, dict):
            continue
        timestamp = item.get("timestamp")
        activity = item.get("activity")
        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            continue
        activity_str = str(activity or "unknown")
        events.append((ts, activity_str))

    if not events:
        return []

    events.sort(key=lambda pair: pair[0])

    start_ts_raw = metadata.get("start_timestamp")
    try:
        start_ts = float(start_ts_raw) if start_ts_raw is not None else events[0][0]
    except (TypeError, ValueError):
        start_ts = events[0][0]

    end_ts_raw = metadata.get("end_timestamp")
    end_ts: Optional[float]
    if end_ts_raw is not None:
        try:
            end_ts = float(end_ts_raw)
        except (TypeError, ValueError):
            end_ts = None
    else:
        end_ts = None
    if end_ts is None:
        sample_ts_values = []
        for sample in samples:
            try:
                sample_ts_values.append(float(sample.get("timestamp")))
            except (TypeError, ValueError):
                continue
        if sample_ts_values:
            end_ts = max(sample_ts_values)
    if end_ts is None:
        end_ts = events[-1][0]

    max_elapsed = None
    if sampled_times:
        max_elapsed = max(sampled_times)

    segments: List[Tuple[float, float, str]] = []
    for idx, (event_ts, activity) in enumerate(events):
        next_ts = events[idx + 1][0] if idx + 1 < len(events) else end_ts
        try:
            raw_end_ts = float(next_ts)
        except (TypeError, ValueError):
            raw_end_ts = event_ts
        start_elapsed = max(0.0, event_ts - start_ts)
        end_elapsed = max(start_elapsed, raw_end_ts - start_ts)
        if max_elapsed is not None:
            end_elapsed = min(end_elapsed, max_elapsed)
        if end_elapsed <= start_elapsed:
            if max_elapsed is not None and max_elapsed > start_elapsed:
                end_elapsed = max_elapsed
            else:
                end_elapsed = start_elapsed + 1e-6
        segments.append((start_elapsed, end_elapsed, activity))

    if not segments and sampled_times:
        # Fallback: treat entire run as a single segment with the most recent activity.
        segments.append((0.0, sampled_times[-1], events[-1][1]))

    segments = _canonicalise_segments(segments)
    segments = _collapse_short_segments(segments)
    return segments


def _shade_activity_regions(ax: plt.Axes, segments: List[Tuple[float, float, str]]) -> None:
    if not segments:
        return

    labels_used: Set[str] = set()
    displayed_labels: Set[str] = set()

    for start, end, phase in segments:
        if end <= start:
            continue
        color = PHASE_COLORS.get(phase, PHASE_COLORS["other"])
        label = _format_activity_label(phase, labels_used)
        span_kwargs = {"alpha": 0.25, "color": color, "lw": 0}
        if label is not None:
            span_kwargs["label"] = label
            displayed_labels.add(label)
        ax.axvspan(start, end, **span_kwargs)
        labels_used.add(phase)

    if displayed_labels:
        ax.legend(loc="upper right", title="Activity")


def _format_activity_label(activity_key: str, labels_used: Set[str]) -> Optional[str]:
    if activity_key in labels_used:
        return None
    pretty = activity_key.replace("-", " ").strip().title() or "Unknown"
    return pretty


def _canonicalise_segments(segments: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    canonicalised: List[Tuple[float, float, str]] = []
    for start, end, activity in segments:
        phase = ACTIVITY_CANONICAL.get(activity.lower().strip(), activity.lower().strip())
        if canonicalised and canonicalised[-1][2] == phase:
            prev_start, prev_end, _ = canonicalised[-1]
            canonicalised[-1] = (prev_start, max(prev_end, end), phase)
        else:
            canonicalised.append((start, end, phase))
    return canonicalised


def _collapse_short_segments(segments: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    if not segments:
        return []

    collapsed: List[Tuple[float, float, str]] = []
    carry_start: Optional[float] = None

    for start, end, phase in segments:
        if carry_start is not None:
            start = min(start, carry_start)
            carry_start = None

        duration = end - start
        if duration < MIN_SEGMENT_DURATION:
            if collapsed:
                prev_start, prev_end, prev_phase = collapsed[-1]
                collapsed[-1] = (prev_start, max(prev_end, end), prev_phase)
            else:
                carry_start = start
            continue

        if collapsed and collapsed[-1][2] == phase:
            prev_start, prev_end, _ = collapsed[-1]
            collapsed[-1] = (prev_start, max(prev_end, end), phase)
        else:
            collapsed.append((start, end, phase))

    if carry_start is not None and collapsed:
        prev_start, prev_end, prev_phase = collapsed[-1]
        collapsed[-1] = (min(prev_start, carry_start), prev_end, prev_phase)

    if not collapsed and segments:
        collapsed.append(segments[-1])

    return collapsed


def main() -> None:
    args = parse_args()
    payload = load_metrics(args.metrics)

    if is_power_log(payload):
        fig = render_power_plot(payload, title=args.title)
    else:
        matrices = payload.get("confusion_matrices")
        if not matrices:
            raise RuntimeError(f"No confusion matrices found in {args.metrics}")

        entry = pick_entry(matrices, threshold=args.threshold, strategy=args.strategy)

        scale = 100.0 if entry["threshold"] > 1.0 else 1.0
        display_threshold_val = entry["threshold"] / scale if scale else entry["threshold"]
        display_threshold = (
            f"{display_threshold_val:.4f}".rstrip("0").rstrip(".") if isinstance(display_threshold_val, (int, float)) else str(display_threshold_val)
        )

        metrics_name = payload.get("name") or Path(args.metrics).stem
        name_lower = metrics_name.lower()
        metrics_path_lower = str(args.metrics).lower()
        if "spoof" in name_lower or "spoof" in metrics_path_lower:
            class_labels = ("Real", "Spoof")
        else:
            class_labels = ("Known", "Unknown")

        if args.title:
            title = args.title
        else:
            dataset = payload.get("dataset_root", "")
            prefix = metrics_name
            title_parts = [prefix, f"thr={display_threshold}"]
            if dataset:
                title_parts.append(Path(dataset).name)
            title = " | ".join(title_parts)

        fig = render_confusion_matrix(entry, title=title, display_threshold=display_threshold, class_labels=class_labels)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        print(f"Saved figure to {args.output}")

    if args.show or not args.output:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
