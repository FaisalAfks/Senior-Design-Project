#!/usr/bin/env python3
"""Template for fine-tuning BlazeFace on a custom dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from BlazeFace.models import BlazeFace


class DummyFaceDataset(Dataset):
    """Placeholder dataset yielding (image, targets) pairs.

    Replace this with a dataset that returns images in CHW float format and
    `targets` dictionaries matching the SSD training API expectations.
    """

    def __init__(self, size: int = 100) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.randn(3, 128, 128)
        target = torch.zeros((0, 4))
        return image, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BlazeFace on a custom dataset.")
    parser.add_argument("--weights", type=Path, default=None, help="Existing BlazeFace weights to initialise from.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Training learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--output", type=Path, default=Path("blazeface_finetuned.pth"), help="Output checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlazeFace(back_model=False)
    if args.weights:
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state)

    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # TODO: Replace DummyFaceDataset with a dataset that returns proper SSD targets.
    train_loader = DataLoader(DummyFaceDataset(), batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        for images, targets in train_loader:
            images = images.to(device)
            _ = targets  # Replace with loss computation using targets

            raise NotImplementedError(
                "Implement SSD-style classification & localisation losses for BlazeFace training."
            )

        print(f"Completed epoch {epoch + 1}/{args.epochs}")

    torch.save(model.state_dict(), args.output)
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
