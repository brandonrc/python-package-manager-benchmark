import numpy as np
import torch
from unittest.mock import patch

from mlbench.data import get_transforms, CIFAR10Dataset, get_dataloaders


def test_train_transforms_output_shape():
    transform = get_transforms(train=True)
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    result = transform(image=image)["image"]
    assert result.shape == (3, 32, 32)
    assert isinstance(result, torch.Tensor)


def test_val_transforms_output_shape():
    transform = get_transforms(train=False)
    image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    result = transform(image=image)["image"]
    assert result.shape == (3, 32, 32)


def test_dataset_returns_tensor_and_label(mock_hf_dataset):
    with patch("mlbench.data.load_dataset", return_value=mock_hf_dataset):
        ds = CIFAR10Dataset(split="train", transform=get_transforms(False), max_samples=8)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)
        assert isinstance(label, int)
        assert 0 <= label <= 9


def test_dataset_respects_max_samples(mock_hf_dataset):
    with patch("mlbench.data.load_dataset", return_value=mock_hf_dataset):
        ds = CIFAR10Dataset(split="train", transform=get_transforms(False), max_samples=4)
        assert len(ds) == 4


def test_dataloaders_return_batches(mock_hf_dataset):
    with patch("mlbench.data.load_dataset", return_value=mock_hf_dataset):
        train_dl, val_dl = get_dataloaders(batch_size=4, max_samples=8)
        images, labels = next(iter(train_dl))
        assert images.shape[0] <= 4
        assert images.shape[1:] == (3, 32, 32)
        assert labels.shape[0] <= 4
