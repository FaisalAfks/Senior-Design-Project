#!/usr/bin/env python3
"""DeePixBiS anti-spoofing model with optional denseNet weights."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torchvision.models import DenseNet161_Weights, densenet161


class DeePixBiS(nn.Module):
    """Original DeePixBiS head built on the first 8 DenseNet-161 blocks."""

    def __init__(
        self,
        weights: Optional[DenseNet161_Weights] = DenseNet161_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()
        backbone = densenet161(weights=weights)
        features = list(backbone.features.children())
        self.enc = nn.Sequential(*features[:8])  # preserve original key names
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.enc(x)
        depth_map = torch.sigmoid(self.dec(encoded))
        logits = self.linear(depth_map.view(-1, 14 * 14))
        scores = torch.sigmoid(logits).flatten()
        return depth_map, scores
