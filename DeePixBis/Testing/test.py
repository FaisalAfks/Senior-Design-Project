#!/usr/bin/env python3
"""Evaluate a DeePixBiS checkpoint on a labelled CSV split."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from DeePixBis.core import DeePixBiS, PixWiseBCELoss, SpoofMetrics, build_dataloader, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeePixBiS on a CSV split.")
    parser.add_argument("--csv", required=True, help="Path to CSV listing images and labels.")
    parser.add_argument("--weights", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for accuracy.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeePixBiS()
    load_checkpoint(model, Path(args.weights), device=device)

    metrics = SpoofMetrics(threshold=args.threshold)
    loss_fn = PixWiseBCELoss()
    loader = build_dataloader(
        Path(args.csv),
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
    )
    accuracy = metrics.accuracy(model, loader, device=device)
    loss = metrics.loss(model, loader, loss_fn, device=device)

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Loss: {loss:.4f}")


if __name__ == "__main__":
    main()

