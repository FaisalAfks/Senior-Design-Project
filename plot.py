#!/usr/bin/env python3
"""Plot confusion matrices stored in recognizer/spoof metrics JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 2x2 confusion matrix from metrics JSON output.")
    parser.add_argument("metrics", type=Path, help="Path to recognizer_metrics.json or spoof_metrics.json.")
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


def main() -> None:
    args = parse_args()
    payload = load_metrics(args.metrics)
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
