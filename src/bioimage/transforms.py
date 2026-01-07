from __future__ import annotations

from typing import Literal

import torch
from torchvision import transforms as T


class PerImageStandardize:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor shape: C x H x W
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True)
        return (tensor - mean) / (std + self.eps)


NormalizeMode = Literal["per-image", "imagenet", "none"]


def build_transforms(
    train: bool,
    image_size: int = 224,
    normalize: NormalizeMode = "per-image",
) -> T.Compose:
    steps: list[object] = []
    if train:
        steps.extend(
            [
                T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.2),
            ]
        )
    else:
        steps.extend([
            T.Resize(int(image_size * 1.15)),
            T.CenterCrop(image_size),
        ])

    steps.append(T.ToTensor())

    if normalize == "imagenet":
        steps.append(
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    elif normalize == "per-image":
        steps.append(PerImageStandardize())

    if train:
        steps.append(
            T.RandomErasing(
                p=0.15,
                scale=(0.02, 0.08),
                ratio=(0.3, 3.3),
                value="random",
            )
        )

    return T.Compose(steps)
