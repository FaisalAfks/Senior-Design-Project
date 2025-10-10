"""Device resolution helpers shared across scripts."""
from __future__ import annotations

import torch


def select_device(device_name: str | None) -> torch.device:
    name = (device_name or "").strip()
    if not name or name.lower() == "cpu":
        return torch.device("cpu")

    lowered = name.lower()
    if lowered.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")
        return torch.device(name)

    if lowered.startswith("mps"):
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but MPS is not available.")
        return torch.device("mps")

    raise ValueError(f"Unsupported device string: {device_name}")

