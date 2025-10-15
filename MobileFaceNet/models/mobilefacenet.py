#!/usr/bin/env python3
"""MobileFaceNet model, facebank utilities, and alignment helpers."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Linear, PReLU, Parameter, Sequential
from torchvision import transforms as T

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS_PATH = PACKAGE_ROOT / "Weights" / "MobileFace_Net"
DEFAULT_FACEBANK_DIR = PROJECT_ROOT / "Facebank"

from BlazeFace.detector import BlazeFaceDetector

__all__ = [
    "MobileFaceNet",
    "Arcface",
    "l2_norm",
    "prepare_facebank",
    "load_facebank",
    "DEFAULT_FACEBANK_DIR",
    "WEIGHTS_PATH",
    "transformation_from_points",
    "align_faces_from_landmarks",
]


# ---------------------------------------------------------------------------
# Embedding model definitions
# ---------------------------------------------------------------------------

class Flatten(nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.size(0), -1)


def l2_norm(tensor: torch.Tensor, axis: int = 1) -> torch.Tensor:
    norm = torch.norm(tensor, 2, axis, keepdim=True)
    return torch.div(tensor, norm)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels)
        self.prelu = PReLU(out_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.conv(tensor)
        tensor = self.bn(tensor)
        tensor = self.prelu(tensor)
        return tensor


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.conv(tensor)
        tensor = self.bn(tensor)
        return tensor


class DepthWise(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        residual: bool = False,
        kernel: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (1, 1),
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(groups, out_channels, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        shortcut = tensor if self.residual else None
        tensor = self.conv(tensor)
        tensor = self.conv_dw(tensor)
        tensor = self.project(tensor)
        if shortcut is not None:
            tensor = shortcut + tensor
        return tensor


class Residual(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_block: int,
        groups: int,
        kernel: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        blocks = [
            DepthWise(
                channels,
                channels,
                residual=True,
                kernel=kernel,
                padding=padding,
                stride=stride,
                groups=groups,
            )
            for _ in range(num_block)
        ]
        self.model = Sequential(*blocks)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128)
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256)
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256)
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.conv1(tensor)
        tensor = self.conv2_dw(tensor)
        tensor = self.conv_23(tensor)
        tensor = self.conv_3(tensor)
        tensor = self.conv_34(tensor)
        tensor = self.conv_4(tensor)
        tensor = self.conv_45(tensor)
        tensor = self.conv_5(tensor)
        tensor = self.conv_6_sep(tensor)
        tensor = self.conv_6_dw(tensor)
        tensor = self.conv_6_flatten(tensor)
        tensor = self.linear(tensor)
        tensor = self.bn(tensor)
        return l2_norm(tensor)


class Arcface(nn.Module):
    def __init__(
        self,
        embedding_size: int = 512,
        classnum: int = 51332,
        *,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        nn.init.xavier_uniform_(self.kernel)
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.mm = self.sin_m * margin
        self.threshold = math.cos(math.pi - margin)

    def forward(self, embeddings: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        batch = embeddings.size(0)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cond_mask = (cos_theta - self.threshold) <= 0
        keep_val = cos_theta - self.mm
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta.clone()
        idx = torch.arange(0, batch, dtype=torch.long)
        output[idx, label] = cos_theta_m[idx, label]
        output *= self.scale
        return output


# ---------------------------------------------------------------------------
# Landmark alignment helpers (ported from align_trans)
# ---------------------------------------------------------------------------

REFERENCE_FIVE_POINTS_SQUARE = np.array(
    [
        [38.29459953, 51.69630051],
        [73.53179932, 51.50139999],
        [56.02519989, 71.73660278],
        [41.54930115, 92.3655014],
        [70.72990036, 92.20410156],
    ],
    dtype=np.float32,
)
REFERENCE_FIVE_POINTS_RECT = np.array(
    [
        [30.29459953, 51.69630051],
        [65.53179932, 51.50139999],
        [48.02519989, 71.73660278],
        [33.54930115, 92.3655014],
        [62.72990036, 92.20410156],
    ],
    dtype=np.float32,
)

OUTPUT_SIZE_SQUARE: Tuple[int, int] = (112, 112)
OUTPUT_SIZE_RECT: Tuple[int, int] = (112, 96)

LandmarkInput = Union[Sequence[float], Sequence[Sequence[float]], np.ndarray]


def _as_array(landmarks: Optional[LandmarkInput]) -> np.ndarray:
    if landmarks is None:
        return np.empty((0, 5, 2), dtype=np.float32)

    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 5, 2), dtype=np.float32)

    if arr.ndim == 1:
        if arr.size % 10 != 0:
            raise ValueError("Landmark vector length must be a multiple of 10 (5 points).")
        arr = arr.reshape(-1, 5, 2)
    elif arr.ndim == 2:
        if arr.shape == (5, 2):
            arr = arr.reshape(1, 5, 2)
        elif arr.shape[1] == 10:
            arr = arr.reshape(-1, 5, 2)
        else:
            raise ValueError(f"Expected landmarks shaped (N, 10) or (5, 2); received {arr.shape}")
    elif arr.ndim == 3:
        if arr.shape[1:] != (5, 2):
            raise ValueError(f"Expected landmarks shaped (N, 5, 2); received {arr.shape}")
    else:
        raise ValueError(f"Unsupported landmark dimensions: {arr.shape}")

    return arr.astype(np.float32, copy=False)


def transformation_from_points(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)

    if points1.shape != (5, 2) or points2.shape != (5, 2):
        raise ValueError("Both point sets must be shaped (5, 2).")

    c1 = points1.mean(axis=0)
    c2 = points2.mean(axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.linalg.norm(points1) / np.sqrt(points1.size)
    if s1 < 1e-8:
        raise ValueError("Degenerate source landmarks; cannot compute transform.")

    s2 = np.linalg.norm(points2) / np.sqrt(points2.size)
    points1 /= s1
    points2 /= s2

    u, _, vt = np.linalg.svd(points1.T @ points2)
    if np.linalg.det(u @ vt) < 0:
        vt[-1, :] *= -1.0
    r = (u @ vt).T
    scale = s2 / s1

    transform = np.eye(3)
    transform[:2, :2] = scale * r
    transform[:2, 2] = (c2 - scale * (r @ c1)).astype(np.float64)
    return transform


def align_faces_from_landmarks(
    image_bgr: np.ndarray,
    landmarks: Optional[LandmarkInput],
    *,
    default_square: bool = True,
    output_size: Optional[Tuple[int, int]] = None,
    return_matrices: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
    prepared = _as_array(landmarks)
    if prepared.size == 0:
        empty: List[np.ndarray] = []
        return (empty, []) if return_matrices else empty

    reference = REFERENCE_FIVE_POINTS_SQUARE if default_square else REFERENCE_FIVE_POINTS_RECT
    base_size = OUTPUT_SIZE_SQUARE if default_square else OUTPUT_SIZE_RECT
    size = base_size if output_size is None else (int(output_size[0]), int(output_size[1]))

    faces: List[np.ndarray] = []
    matrices: List[np.ndarray] = []

    for landmark_set in prepared:
        matrix = transformation_from_points(landmark_set, reference)
        aligned = cv2.warpAffine(
            image_bgr,
            matrix[:2],
            dsize=(size[0], size[1]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        faces.append(aligned)
        matrices.append(matrix)

    return (faces, matrices) if return_matrices else faces


# ---------------------------------------------------------------------------
# Facebank helpers (formerly in facebank.py)
# ---------------------------------------------------------------------------

_WEIGHTS_TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def _iter_visible(path: Path) -> Iterable[Path]:
    for entry in sorted(path.iterdir()):
        if not entry.name.startswith("."):
            yield entry


def _detect_and_align(detector: BlazeFaceDetector, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    detections = detector.detect(image_bgr)
    if not detections:
        return None
    best = max(detections, key=lambda det: det.score)
    return detector.align_face(image_bgr, best)


def prepare_facebank(
    model: MobileFaceNet,
    *,
    path: Path = DEFAULT_FACEBANK_DIR,
    tta: bool = True,
    detector: Optional[BlazeFaceDetector] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    model.eval()
    detector = detector or BlazeFaceDetector()
    device = device or next(model.parameters()).device

    path.mkdir(parents=True, exist_ok=True)
    embeddings: List[torch.Tensor] = []
    names = [""]

    for person_dir in _iter_visible(path):
        if not person_dir.is_dir():
            continue

        person_embeddings: List[torch.Tensor] = []
        for image_path in _iter_visible(person_dir):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            if image.shape == (112, 112, 3):
                face = image
            else:
                face = _detect_and_align(detector, image)
            if face is None:
                continue

            tensor = _WEIGHTS_TRANSFORM(face).to(device).unsqueeze(0)
            if tta:
                mirror = cv2.flip(face, 1)
                mirror_tensor = _WEIGHTS_TRANSFORM(mirror).to(device).unsqueeze(0)
                embedding = l2_norm(model(tensor) + model(mirror_tensor))
            else:
                embedding = model(tensor)
            person_embeddings.append(embedding)

        if not person_embeddings:
            continue

        fused = torch.cat(person_embeddings).mean(0, keepdim=True)
        embeddings.append(fused)
        names.append(person_dir.name)

    if not embeddings:
        raise RuntimeError("No embeddings were generated; ensure facebank images contain faces.")

    embeddings_tensor = torch.cat(embeddings)
    names_array = np.array(names)
    torch.save(embeddings_tensor, path / "facebank.pth")
    np.save(path / "names", names_array)
    return embeddings_tensor, names_array


def load_facebank(path: Path = DEFAULT_FACEBANK_DIR) -> Tuple[torch.Tensor, np.ndarray]:
    embeddings = torch.load(path / "facebank.pth")
    names = np.load(path / "names.npy")
    return embeddings, names


def _cli_rebuild_facebank() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"MobileFaceNet weights not found at {WEIGHTS_PATH}")
    model = MobileFaceNet(512).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    prepare_facebank(model, detector=BlazeFaceDetector(), device=device)


if __name__ == "__main__":
    _cli_rebuild_facebank()

