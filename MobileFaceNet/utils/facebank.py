#!/usr/bin/env python3
"""CLI helpers for maintaining the MobileFaceNet facebank."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from BlazeFace import BlazeFaceService
from MobileFaceNet.models.mobilefacenet import MobileFaceNet, load_facebank, prepare_facebank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild or inspect the MobileFaceNet facebank.")
    parser.add_argument("--facebank", default="facebank", help="Path to the facebank directory")
    default_weights = Path(__file__).resolve().parents[1] / "Weights" / "MobileFace_Net"
    parser.add_argument("--weights", default=str(default_weights), help="MobileFaceNet weights to load")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation when rebuilding")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the facebank embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    facebank_dir = Path(args.facebank)
    if args.rebuild:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        model = MobileFaceNet(512).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        detector = BlazeFaceService().detector
        embeddings, names = prepare_facebank(
            model,
            path=facebank_dir,
            tta=args.tta,
            detector=detector,
            device=device,
        )
        print(f"Rebuilt facebank with {len(names) - 1} identities at {facebank_dir}")
    else:
        embeddings, names = load_facebank(facebank_dir)
        print(f"Loaded facebank containing {len(names) - 1} identities")
        print("Embedding shape:", tuple(embeddings.shape))


if __name__ == "__main__":
    main()

