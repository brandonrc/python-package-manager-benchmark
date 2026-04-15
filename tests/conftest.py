import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure project root is on path for benchmark imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockHFDataset:
    """Mock HuggingFace dataset matching CIFAR-10 format."""

    def __init__(self, n=16):
        self._data = [
            {
                "img": Image.fromarray(
                    np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                ),
                "label": i % 10,
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def select(self, indices):
        new = MockHFDataset.__new__(MockHFDataset)
        new._data = [self._data[i] for i in indices if i < len(self._data)]
        return new


@pytest.fixture
def mock_hf_dataset():
    return MockHFDataset(n=16)


@pytest.fixture
def sample_batch():
    """4 synthetic CIFAR-10-shaped images + labels."""
    return torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))


@pytest.fixture
def sample_model():
    from mlbench.model import ImageClassifier

    return ImageClassifier(num_classes=10)
