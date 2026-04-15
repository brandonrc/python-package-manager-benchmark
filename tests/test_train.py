import os
from unittest.mock import patch

from mlbench.train import train


def test_training_completes_one_epoch(mock_hf_dataset, tmp_path):
    with patch("mlbench.data.load_dataset", return_value=mock_hf_dataset):
        trainer, model = train(
            max_epochs=1,
            max_samples=16,
            batch_size=4,
            output_dir=str(tmp_path),
        )
        assert model is not None
        assert trainer.state.finished


def test_training_produces_checkpoint(mock_hf_dataset, tmp_path):
    with patch("mlbench.data.load_dataset", return_value=mock_hf_dataset):
        trainer, model = train(
            max_epochs=1,
            max_samples=16,
            batch_size=4,
            output_dir=str(tmp_path),
        )
        # Lightning creates checkpoints in default_root_dir
        assert any(tmp_path.rglob("*.ckpt"))
