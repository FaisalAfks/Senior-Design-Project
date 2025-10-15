#!/usr/bin/env python3
"""Command line training entry point for DeePixBiS."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from DeePixBis.core import (
    DeePixBiS,
    PixWiseBCELoss,
    SpoofMetrics,
    Trainer,
    build_dataloader,
    freeze_backbone,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeePixBiS on pixel-wise supervision.")
    parser.add_argument("--train-csv", required=True, help="Path to the training CSV file.")
    parser.add_argument("--val-csv", required=True, help="Path to the validation CSV file.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=10, help="Mini-batch size.")
    parser.add_argument("--weights", default=None, help="Optional checkpoint to resume/fine-tune from.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze DenseNet encoder/decoder layers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Metric decision threshold.")
    parser.add_argument("--beta", type=float, default=0.5, help="Pixel / binary loss mixing parameter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeePixBiS()
    if args.weights:
        load_checkpoint(model, Path(args.weights), device=device)
    if args.freeze_backbone:
        freeze_backbone(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = PixWiseBCELoss(beta=args.beta)
    metrics = SpoofMetrics(threshold=args.threshold)

    train_loader = build_dataloader(Path(args.train_csv), batch_size=args.batch_size, train=True)
    val_loader = build_dataloader(Path(args.val_csv), batch_size=args.batch_size, train=False, shuffle=False)

    trainer = Trainer(model, optimizer, loss_fn, metrics, device=device)
    history = trainer.fit(train_loader, val_loader=val_loader, epochs=args.epochs)

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    checkpoint = output_dir / "deeppixbis_finetuned.pth"
    torch.save(model.state_dict(), checkpoint)
    print(f"Checkpoint saved to {checkpoint}")
    if history.accuracy:
        print("Validation accuracy history:", [f"{acc:.2%}" for acc in history.accuracy])
        print("Validation loss history:", [f"{loss:.4f}" for loss in history.loss])


if __name__ == "__main__":
    main()

