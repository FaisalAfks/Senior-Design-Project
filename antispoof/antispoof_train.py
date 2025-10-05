#!/usr/bin/env python3
"""Training script for DeePixBiS anti-spoofing model."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                    RandomRotation, Resize, ToTensor)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from antispoof.antispoof_dataset import PixWiseDataset
from antispoof.antispoof_loss import PixWiseBCELoss
from antispoof.antispoof_metrics import predict, test_accuracy, test_loss
from antispoof.antispoof_model import DeePixBiS
from antispoof.antispoof_trainer import Trainer

WEIGHTS_PATH = ROOT / "weights" / "DeePixBiS.pth"
TRAIN_CSV = ROOT / "antispoof" / "antispoof_train_data.csv"
TEST_CSV = ROOT / "antispoof" / "antispoof_test_data.csv"


def build_dataloaders(batch_size: int = 10):
    train_tfms = Compose([
        Resize([224, 224]),
        RandomHorizontalFlip(),
        RandomRotation(10),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    test_tfms = Compose([
        Resize([224, 224]),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_ds = PixWiseDataset(str(TRAIN_CSV), transform=train_tfms).dataset()
    val_ds = PixWiseDataset(str(TEST_CSV), transform=test_tfms).dataset()

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_dl, val_dl


def main() -> None:
    model = DeePixBiS()
    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = PixWiseBCELoss()
    train_dl, val_dl = build_dataloaders()

    trainer = Trainer(train_dl, val_dl, model, epochs=1, opt=optimizer, loss_fn=loss_fn)
    print("Training Beginning\n")
    trainer.fit()
    print("\nTraining Complete")
    torch.save(model.state_dict(), WEIGHTS_PATH)


if __name__ == "__main__":
    main()
