#!/usr/bin/env python3
"""Core building blocks for DeePixBiS anti-spoofing (model, data, training helpers)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


# --------------------------------------------------------------------------- #
#  Model definition                                                           #
# --------------------------------------------------------------------------- #


class DeePixBiS(nn.Module):
    """Original DeePixBiS head built on the first DenseNet161 blocks."""

    def __init__(
        self,
        weights: Optional[str] = "IMAGENET1K_V1",
    ) -> None:
        super().__init__()
        from torchvision.models import DenseNet161_Weights, densenet161

        torch_weights = None
        if weights:
            torch_weights = getattr(DenseNet161_Weights, weights)
        backbone = densenet161(weights=torch_weights)
        features = list(backbone.features.children())
        self.encoder = nn.Sequential(*features[:8])
        self.decoder = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.classifier = nn.Linear(14 * 14, 1)

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(tensor)
        depth_map = torch.sigmoid(self.decoder(encoded))
        logits = self.classifier(depth_map.view(-1, 14 * 14))
        scores = torch.sigmoid(logits).flatten()
        return depth_map, scores


# --------------------------------------------------------------------------- #
#  Dataset + transforms                                                       #
# --------------------------------------------------------------------------- #


def default_transforms(train: bool) -> T.Compose:
    common = [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    if not train:
        return T.Compose(common)
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


class PixWiseDataset(Dataset):
    """Pixel-wise depth map dataset used by DeePixBiS."""

    def __init__(
        self,
        csv_path: Path | str,
        *,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        map_size: int = 14,
        smoothing: float = 0.99,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        self.transform = transform or default_transforms(train=False)
        self.map_size = int(map_size)
        self.real_weight = float(smoothing)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[idx]
        image_path = Path(row["name"])
        label = float(row["label"])
        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))
        mask_value = self.real_weight if label == 1 else 1.0 - self.real_weight
        mask = torch.full((1, self.map_size, self.map_size), mask_value, dtype=torch.float32)
        return image, mask, torch.tensor(label, dtype=torch.float32)


def build_dataloader(
    csv_path: Path | str,
    *,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    train: bool = True,
    smoothing: float = 0.99,
) -> DataLoader:
    dataset = PixWiseDataset(
        csv_path,
        transform=default_transforms(train=train),
        smoothing=smoothing,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if train else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# --------------------------------------------------------------------------- #
#  Loss + metrics                                                             #
# --------------------------------------------------------------------------- #


class PixWiseBCELoss(nn.Module):
    """Combined pixel-wise + binary BCE loss used by DeePixBiS."""

    def __init__(self, beta: float = 0.5) -> None:
        super().__init__()
        self.beta = beta
        self.criterion = nn.BCELoss()

    def forward(
        self,
        predictions_mask: torch.Tensor,
        predictions_scores: torch.Tensor,
        target_mask: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        pixel_loss = self.criterion(predictions_mask, target_mask)
        binary_loss = self.criterion(predictions_scores, target_labels)
        return pixel_loss * self.beta + binary_loss * (1 - self.beta)


class SpoofMetrics:
    """Utility metrics for DeePixBiS evaluation."""

    def __init__(self, threshold: float = 0.5, score: str = "combined") -> None:
        if score not in {"pixel", "binary", "combined"}:
            raise ValueError("score must be 'pixel', 'binary', or 'combined'")
        self.threshold = threshold
        self.score = score

    def predict(self, mask: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.score == "pixel":
                aggregated = mask.mean(dim=(1, 2, 3))
            elif self.score == "binary":
                aggregated = scores
            else:
                aggregated = (mask.mean(dim=(1, 2, 3)) + scores) / 2
            preds = (aggregated > self.threshold).float()
        return preds, aggregated

    def accuracy(self, model: nn.Module, loader: DataLoader, device: torch.device) -> float:
        correct = 0.0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, masks, labels in loader:
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                net_masks, net_scores = model(images)
                preds, _ = self.predict(net_masks, net_scores)
                correct += torch.sum(preds.to(device) == labels).item()
                total += labels.numel()
        return correct / max(1, total)

    def loss(self, model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
        running = 0.0
        batches = 0
        model.eval()
        with torch.no_grad():
            for images, masks, labels in loader:
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                net_masks, net_scores = model(images)
                loss = loss_fn(net_masks, net_scores, masks, labels)
                running += float(loss.item())
                batches += 1
        return running / max(1, batches)


# --------------------------------------------------------------------------- #
#  Training helpers                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class TrainingHistory:
    epochs: int
    accuracy: List[float]
    loss: List[float]


class Trainer:
    """Opinionated but concise DeePixBiS trainer."""

    def __init__(
        self,
        model: DeePixBiS,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        metrics: SpoofMetrics,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        log_every: int = 10,
    ) -> TrainingHistory:
        history_acc: List[float] = []
        history_loss: List[float] = []

        for epoch in range(epochs):
            self.model.train()
            for step, (images, masks, labels) in enumerate(train_loader, start=1):
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                pred_masks, pred_scores = self.model(images)
                loss = self.loss_fn(pred_masks, pred_scores, masks, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if log_every and step % log_every == 0:
                    print(f"[Epoch {epoch+1}/{epochs}] step {step}: loss={loss.item():.4f}")

            if val_loader is None:
                continue
            acc = self.metrics.accuracy(self.model, val_loader, self.device)
            loss_val = self.metrics.loss(self.model, val_loader, self.loss_fn, self.device)
            history_acc.append(acc)
            history_loss.append(loss_val)
            print(f"Epoch {epoch+1}: val_acc={acc:.3%} val_loss={loss_val:.4f}")

        return TrainingHistory(epochs=epochs, accuracy=history_acc, loss=history_loss)


def freeze_backbone(model: DeePixBiS) -> None:
    """Freeze encoder/decoder layers – used for fine-tuning."""
    for module in (model.encoder, model.decoder):
        module.requires_grad_(False)


def _normalise_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Adapt legacy DeePixBiS checkpoints to the current module naming."""
    # Original repository used `enc`, `dec`, and `linear` module names.
    replacements = {
        "enc.": "encoder.",
        "dec.": "decoder.",
        "linear.": "classifier.",
    }
    normalised = {}
    for key, value in state.items():
        new_key = key
        for src, dst in replacements.items():
            if src in new_key:
                new_key = new_key.replace(src, dst)
        normalised[new_key] = value
    return normalised


def load_checkpoint(model: DeePixBiS, path: Path | str, *, device: Optional[torch.device] = None) -> None:
    """Load weights from disk into the provided DeePixBiS model."""
    map_location = device if device is not None else "cpu"
    state = torch.load(path, map_location=map_location)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        state = _normalise_state_dict(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            warn_msgs = []
            if missing:
                warn_msgs.append(f"Missing keys: {sorted(missing)}")
            if unexpected:
                warn_msgs.append(f"Unexpected keys: {sorted(unexpected)}")
            warn = "DeePixBiS checkpoint loaded with adjustments. " + " ".join(warn_msgs)
            print(warn)
    else:
        model.load_state_dict(state)
