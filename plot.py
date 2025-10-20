#!/usr/bin/env python3
"""Plot metrics or Jetson power telemetry stored in JSON files."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

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
        choices=["max-accuracy", "max-f1", "min-eer"],
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
        "--output-dir",
        type=Path,
        help="Directory to save figures when rendering multiple plots from a combined metrics file.",
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
    parser.add_argument(
        "--plot",
        dest="plot_names",
        action="append",
        help="Name of a plot defined in a combined experiment metrics file (repeatable). Use 'all' to render every plot.",
    )
    parser.add_argument(
        "--list-plots",
        action="store_true",
        help="List available plots defined in the metrics file and exit.",
    )
    return parser.parse_args()


def resolve_metrics_path(path: Path) -> Path:
    """Return the actual metrics file path, falling back to logs/ if needed."""
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = LOGS_DIR.joinpath(*path.parts)
        if candidate.exists():
            return candidate
    return path


def load_metrics(path: Path) -> Dict[str, Any]:
    resolved = resolve_metrics_path(path)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        search_paths = [str(path)]
        if resolved != path:
            search_paths.append(str(resolved))
        raise FileNotFoundError(
            "Metrics file not found; checked: " + ", ".join(search_paths)
        ) from exc


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
        return max(matrices, key=lambda entry: metric_value(entry, key))
    if strategy == "min-eer":
        def eer_key(entry: Dict[str, Any]) -> Tuple[float, float]:
            metrics = entry.get("metrics", {})
            gap = metrics.get("eer_gap")
            eer = metrics.get("eer")
            fpr = metrics.get("fpr")
            fnr = metrics.get("fnr")
            if gap is None and fpr is not None and fnr is not None:
                try:
                    gap = abs(float(fpr) - float(fnr))
                except (TypeError, ValueError):
                    gap = None
            if gap is None:
                gap = float("inf")
            if eer is None and fpr is not None and fnr is not None:
                try:
                    eer = (float(fpr) + float(fnr)) / 2.0
                except (TypeError, ValueError):
                    eer = None
            if eer is None:
                eer = float("inf")
            return float(gap), float(eer)

        return min(matrices, key=eer_key)
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
    total_samples = sum(counts.values())
    matrix = np.array(
        [
            [counts["tp"], counts["fn"]],
            [counts["fp"], counts["tn"]],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    pos_label, neg_label = class_labels
    ax.set_xticks([0, 1], labels=[pos_label, neg_label])
    ax.set_yticks([0, 1], labels=[pos_label, neg_label])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    extra_info_parts: List[str] = []
    if title:
        raw_parts = [part.strip() for part in title.split("|")]
        main_title = raw_parts[0] if raw_parts and raw_parts[0] else title.strip()
        skip_prefixes = ("thr", "threshold", "det", "rec", "spoof", "samples")
        for fragment in raw_parts[1:]:
            cleaned = fragment.strip()
            if not cleaned:
                continue
            lower = cleaned.lower()
            if lower.startswith(skip_prefixes):
                continue
            extra_info_parts.append(cleaned)
    else:
        main_title = "Confusion Matrix"
    fig.suptitle(main_title or "Confusion Matrix")

    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{int(value)}", ha="center", va="center", color="black", fontsize=10)

    raw_metrics = entry.get("metrics")
    metrics: Dict[str, Any] = {}
    if isinstance(raw_metrics, dict):
        metrics = dict(raw_metrics)

    if metrics.get("eer") is None:
        fpr = metrics.get("fpr")
        fnr = metrics.get("fnr")
        if fpr is not None and fnr is not None:
            try:
                metrics["eer"] = (float(fpr) + float(fnr)) / 2.0
            except (TypeError, ValueError):
                pass

    info_lines: List[str] = []
    if extra_info_parts:
        info_lines.append(" | ".join(extra_info_parts))

    details = entry.get("details")
    thresholds_line: Optional[str] = None
    if isinstance(details, dict):
        thresholds = details.get("thresholds")
        if isinstance(thresholds, dict) and thresholds:
            label_map = {
                "detection": "Detection",
                "recognition": "Recognition",
                "spoof": "Spoof",
            }
            formatted = []
            for key in sorted(thresholds):
                label = label_map.get(key, key.replace("_", " ").title())
                try:
                    value = float(thresholds[key])
                    formatted.append(f"{label}: {value:.2f}")
                except (TypeError, ValueError):
                    formatted.append(f"{label}: {thresholds[key]}")
            if formatted:
                thresholds_line = "Thresholds - " + " | ".join(formatted)
    if thresholds_line:
        info_lines.append(thresholds_line)
    else:
        info_lines.append(f"Threshold: {display_threshold}")

    if not any("sample" in line.lower() for line in info_lines):
        info_lines.append(f"Samples: {total_samples}")

    metric_order = [
        ("accuracy", "ACC"),
        ("precision", "PREC"),
        ("tpr", "TPR"),
        ("tnr", "TNR"),
        ("fpr", "FPR"),
        ("fnr", "FNR"),
        ("f1", "F1"),
        ("eer", "EER"),
    ]
    metric_items = [
        f"{label}={metrics[key]:.3f}"
        for key, label in metric_order
        if metrics.get(key) is not None
    ]
    if metric_items:
        items_per_line = 4
        for index in range(0, len(metric_items), items_per_line):
            info_lines.append(" | ".join(metric_items[index : index + items_per_line]))

    if info_lines:
        ax.text(
            0.5,
            -0.28,
            "\n".join(info_lines),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            linespacing=1.4,
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.3)

    axes_group = [ax]
    if cbar is not None:
        axes_group.append(cbar.ax)

    try:
        fig.canvas.draw()
    except Exception:
        pass
    left = min(axis.get_position().x0 for axis in axes_group)
    right = max(axis.get_position().x1 for axis in axes_group)
    width = right - left
    if width > 0 and width < 1:
        desired_left = (1.0 - width) / 2.0
        shift = desired_left - left
        if abs(shift) > 1e-3:
            for axis in axes_group:
                pos = axis.get_position()
                axis.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])

    return fig


def render_accuracy_plot(
    series: Sequence[Tuple[str, Sequence[Dict[str, Any]]]],
    *,
    title: str,
    xlabel: str,
) -> Optional[plt.Figure]:
    filtered_series = [(label, entries) for label, entries in series if entries]
    if not filtered_series:
        return None

    threshold_values = sorted(
        {
            float(entry["threshold"])
            for _, entries in filtered_series
            for entry in entries
        }
    )
    if not threshold_values:
        return None

    sample_size: Optional[int] = None
    for _, entries in filtered_series:
        if entries:
            counts = entries[0].get("counts")
            if counts:
                sample_size = sum(counts.values())
            break

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = color_cycle.by_key().get("color", []) if color_cycle is not None else []
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    # Provide contrasting dash colors so FPR/FNR stand out
    fpr_colors = ["#d62728", "#e377c2", "#bcbd22", "#17becf", "#ff9896", "#c5b0d5"]
    fnr_colors = ["#9467bd", "#8c564b", "#aec7e8", "#98df8a", "#ffbb78", "#c49c94"]

    for index, (label, entries) in enumerate(filtered_series):
        lookup = {float(entry["threshold"]): entry for entry in entries}
        accuracy_vals: List[float] = []
        fpr_vals: List[float] = []
        fnr_vals: List[float] = []
        for threshold in threshold_values:
            metrics = lookup.get(threshold, {}).get("metrics", {})
            accuracy = metrics.get("accuracy")
            fpr = metrics.get("fpr")
            fnr = metrics.get("fnr")
            accuracy_vals.append(float("nan") if accuracy is None else float(accuracy))
            fpr_vals.append(float("nan") if fpr is None else float(fpr))
            fnr_vals.append(float("nan") if fnr is None else float(fnr))

        color = colors[index % len(colors)]
        fpr_color = fpr_colors[index % len(fpr_colors)]
        fnr_color = fnr_colors[index % len(fnr_colors)]
        ax.plot(
            threshold_values,
            accuracy_vals,
            color=color,
            linestyle="-",
            marker="o",
            label=f"{label} accuracy",
        )
        ax.plot(
            threshold_values,
            fpr_vals,
            color=fpr_color,
            linestyle="--",
            marker="x",
            label=f"{label} FPR",
        )
        ax.plot(
            threshold_values,
            fnr_vals,
            color=fnr_color,
            linestyle="--",
            marker="^",
            label=f"{label} FNR",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    extra_info_parts: List[str] = []
    if title:
        raw_parts = [part.strip() for part in title.split("|")]
        main_title = raw_parts[0] if raw_parts and raw_parts[0] else title.strip()
        seen_extra: Set[str] = set()
        for part in raw_parts[1:]:
            cleaned = part.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_extra:
                continue
            seen_extra.add(key)
            extra_info_parts.append(cleaned)
    else:
        main_title = title
    sample_override: Optional[str] = None
    normalised_info: List[str] = []
    for part in extra_info_parts:
        lower = part.lower()
        if "sample" in lower:
            match = _NUMBER_RE.search(part)
            if match:
                sample_override = f"Samples: {match.group(0)}"
            else:
                sample_override = part.capitalize()
            continue
        if lower.startswith("best") and "thr" in lower:
            match = _NUMBER_RE.search(part)
            if match:
                normalised_info.append(f"Best threshold: {match.group(0)}")
            else:
                normalised_info.append(part.replace("thr", "threshold").capitalize())
            continue
        normalised_info.append(part)
    extra_info_parts = normalised_info
    fig.suptitle(main_title or "")

    fpr_values: List[float] = []
    fnr_values: List[float] = []
    for _, entries in filtered_series:
        for entry in entries:
            metrics = entry.get("metrics", {})
            fpr = metrics.get("fpr")
            fnr = metrics.get("fnr")
            if isinstance(fpr, (int, float)):
                fpr_values.append(float(fpr))
            if isinstance(fnr, (int, float)):
                fnr_values.append(float(fnr))

    footer_lines: List[str] = []
    if extra_info_parts:
        footer_lines.append(" | ".join(extra_info_parts))

    threshold_min = threshold_values[0]
    threshold_max = threshold_values[-1]
    if len(threshold_values) == 1:
        threshold_label = f"Threshold: {threshold_min:.4g}"
    else:
        threshold_label = f"Threshold range: {threshold_min:.4g} – {threshold_max:.4g}"
    footer_lines.append(threshold_label)

    supplements: List[str] = []
    if sample_override:
        supplements.append(sample_override)
    elif sample_size is not None:
        supplements.append(f"Samples: {sample_size}")

    step_value: Optional[float] = None
    if len(threshold_values) > 1:
        delta_set = {
            round(threshold_values[idx + 1] - threshold_values[idx], 10)
            for idx in range(len(threshold_values) - 1)
            if abs(threshold_values[idx + 1] - threshold_values[idx]) > 1e-9
        }
        deltas = sorted(delta_set)
        if deltas:
            step_value = deltas[0]
    if step_value is not None:
        supplements.append(f"Step: {step_value:.4g}")

    if supplements:
        footer_lines.append(" | ".join(supplements))

    if footer_lines:
        fig.text(
            0.5,
            0.02,
            "\n".join(footer_lines),
            ha="center",
            va="bottom",
            fontsize=9,
            linespacing=1.3,
        )

    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.2)
    return fig


def save_accuracy_plot(
    series: Sequence[Tuple[str, Sequence[Dict[str, Any]]]],
    output_path: Path,
    *,
    title: str,
    xlabel: str,
) -> None:
    fig = render_accuracy_plot(series, title=title, xlabel=xlabel)
    if fig is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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

    plots_section = payload.get("plots") if isinstance(payload, dict) else None
    if isinstance(plots_section, dict) and plots_section:
        available_names = list(plots_section.keys())
        if args.list_plots or not args.plot_names:
            print("Available plots:")
            for name in available_names:
                title = plots_section.get(name, {}).get("title", name)
                print(f" - {name}: {title}")
            if not args.plot_names:
                print("Re-run with --plot <name> (or --plot all) to render a plot.")
            return

        if args.output and args.output_dir:
            raise SystemExit("Use either --output or --output-dir, not both.")

        selected_names: List[str] = []
        for raw_name in args.plot_names or []:
            if raw_name.lower() == "all":
                for candidate in available_names:
                    if candidate not in selected_names:
                        selected_names.append(candidate)
                continue
            if raw_name not in plots_section:
                available_display = ", ".join(available_names)
                raise SystemExit(f"Unknown plot '{raw_name}'. Available plots: {available_display}")
            if raw_name not in selected_names:
                selected_names.append(raw_name)

        if not selected_names:
            raise SystemExit("No plots selected.")
        if args.output and len(selected_names) > 1:
            raise SystemExit("Cannot use --output when rendering multiple plots; use --output-dir instead.")

        figures: List[plt.Figure] = []
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)

        for name in selected_names:
            config = plots_section.get(name, {})
            plot_type = config.get("type")
            if plot_type is None:
                plot_type = "accuracy" if "series" in config else "confusion"

            if args.title and len(selected_names) == 1:
                plot_title = args.title
            else:
                plot_title = config.get("title", name)

            plot_subtitle = config.get("subtitle")
            plot_figures: List[plt.Figure] = []
            if isinstance(plot_subtitle, str) and plot_subtitle:
                plot_title = f"{plot_title} | {plot_subtitle}" if plot_title else plot_subtitle

            if plot_type == "accuracy":
                series_specs = config.get("series", [])
                if name.startswith("spoof_") and len(series_specs) > 1:
                    for spec in series_specs:
                        label = spec.get("label", "Series")
                        entries = spec.get("entries")
                        if not entries:
                            continue
                        subplot_title = f"{plot_title} ({label})"
                        fig = render_accuracy_plot(
                            [(label, entries)],
                            title=subplot_title,
                            xlabel=config.get("xlabel", "Threshold"),
                        )
                        if fig is not None:
                            plot_figures.append(fig)
                else:
                    series_data: List[Tuple[str, Sequence[Dict[str, Any]]]] = []
                    for spec in series_specs:
                        label = spec.get("label", "Series")
                        entries = spec.get("entries")
                        if not entries:
                            continue
                        series_data.append((label, entries))
                    if not series_data:
                        print(f"Skipping plot '{name}' (no data).")
                        continue
                    fig = render_accuracy_plot(
                        series_data,
                        title=plot_title,
                        xlabel=config.get("xlabel", "Threshold"),
                    )
                    if fig is not None:
                        plot_figures.append(fig)
            elif plot_type == "confusion":
                confusion_matrices = config.get("confusion_matrices")
                if not confusion_matrices:
                    print(f"Skipping plot '{name}' (no confusion matrices).")
                    continue
                default_threshold = config.get("default_threshold")
                threshold_override = args.threshold if args.threshold is not None else default_threshold
                entry = pick_entry(confusion_matrices, threshold=threshold_override, strategy=args.strategy)
                threshold_value = entry.get("threshold")
                scale = 100.0 if threshold_value and threshold_value > 1.0 else 1.0
                display_value = threshold_value / scale if scale else threshold_value
                display_threshold = (
                    f"{display_value:.4f}".rstrip("0").rstrip(".")
                    if isinstance(display_value, (int, float))
                    else str(display_value)
                )
                labels_override = config.get("labels")
                if isinstance(labels_override, list) and len(labels_override) == 2:
                    class_labels = (labels_override[0], labels_override[1])
                else:
                    lower_name = name.lower()
                    if "spoof" in lower_name:
                        class_labels = ("Real", "Spoof")
                    elif "detector" in lower_name:
                        class_labels = ("Face", "No Face")
                    elif "pipeline" in lower_name:
                        class_labels = ("Accept", "Reject")
                    else:
                        class_labels = ("Known", "Unknown")
                fig = render_confusion_matrix(
                    entry,
                    title=plot_title,
                    display_threshold=display_threshold,
                    class_labels=class_labels,
                )
                if fig is not None:
                    plot_figures.append(fig)
            else:
                print(f"Skipping plot '{name}' (unsupported type: {plot_type}).")
                continue

            if not plot_figures:
                print(f"Skipping plot '{name}' (insufficient data).")
                continue

            if args.output and len(plot_figures) > 1:
                raise SystemExit("Cannot save multiple figures to a single --output file; use --output-dir instead.")

            for idx_fig, fig in enumerate(plot_figures):
                target_path: Optional[Path] = None
                if args.output_dir:
                    filename = f"{name}.png" if len(plot_figures) == 1 else f"{name}_{idx_fig + 1}.png"
                    target_path = args.output_dir / filename
                elif args.output:
                    target_path = args.output

                if target_path is not None:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(target_path, dpi=args.dpi)
                    print(f"Saved figure to {target_path}")
                    if args.show or (not args.output and not args.output_dir):
                        figures.append(fig)
                    else:
                        plt.close(fig)
                else:
                    figures.append(fig)

        if not figures and not (args.output_dir and not args.show):
            print("No plots were generated.")

        if figures:
            if args.show or (not args.output and not args.output_dir):
                plt.show()
            else:
                for fig in figures:
                    plt.close(fig)
        return
    elif args.plot_names or args.list_plots:
        raise SystemExit("This metrics file does not define composite plots.")

    if args.output_dir:
        raise SystemExit("--output-dir is only supported when rendering plots from a combined metrics file.")

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

        fig = render_confusion_matrix(
            entry,
            title=title,
            display_threshold=display_threshold,
            class_labels=class_labels,
        )

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
