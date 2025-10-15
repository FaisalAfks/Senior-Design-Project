#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Take cropped face from image

@author: AIRocker
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
for path in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.path.append(str(PACKAGE_ROOT / "Testing" / "MTCNN"))

from MTCNN import create_mtcnn_net
from MobileFaceNet.utils.align_trans import *
import cv2
import argparse
import torch

parser = argparse.ArgumentParser(description='take ID from Picture')
parser.add_argument('--image','-i', default='images/Sheldon.jpg', type=str,help='input the image of the person')
parser.add_argument('--name','-n', default='Sheldon', type=str,help='input the name of the person')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image = cv2.imread(args.image)
weights_dir = PACKAGE_ROOT / "Testing" / "MTCNN" / "weights"
bboxes, landmarks = create_mtcnn_net(
    image,
    20,
    device,
    p_model_path=str(weights_dir / "pnet_Weights"),
    r_model_path=str(weights_dir / "rnet_Weights"),
    o_model_path=str(weights_dir / "onet_Weights"),
)

warped_face = Face_alignment(image, default_square=True, landmarks=landmarks)

data_path = PROJECT_ROOT / "Facebank"
save_path = data_path / args.name
save_path.mkdir(parents=True, exist_ok=True)

def _next_facebank_index(directory: Path) -> int:
    prefix = "facebank_"
    max_index = 0
    for file in directory.glob(f"{prefix}*"):
        if not file.is_file():
            continue
        parts = file.stem.split("_")
        if not parts:
            continue
        try:
            idx = int(parts[-1])
        except ValueError:
            continue
        if idx > max_index:
            max_index = idx
    return max_index + 1


next_idx = _next_facebank_index(save_path)
filename = f"facebank_{next_idx:02d}.jpg"
cv2.imwrite(str(save_path / filename), warped_face[0])
