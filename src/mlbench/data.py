"""Dataset loading and preprocessing using HuggingFace datasets + albumentations."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_transforms(train: bool = True) -> A.Compose:
    """Return albumentations transforms for CIFAR-10."""
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2(),
    ])


class CIFAR10Dataset(Dataset):
    """Wraps a HuggingFace CIFAR-10 split with albumentations transforms."""

    def __init__(self, split: str = "train", transform: A.Compose | None = None, max_samples: int | None = None):
        ds = load_dataset("cifar10", split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.data[idx]
        image = np.array(item["img"])
        label = item["label"]
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def get_dataloaders(
    batch_size: int = 32,
    max_samples: int | None = 500,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for CIFAR-10."""
    train_ds = CIFAR10Dataset("train", get_transforms(True), max_samples)
    val_ds = CIFAR10Dataset("test", get_transforms(False), max_samples)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl
