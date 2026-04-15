"""Training entrypoint using PyTorch Lightning."""

from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from mlbench.data import get_dataloaders
from mlbench.model import ImageClassifier


def train(
    max_epochs: int = 2,
    max_samples: int = 500,
    batch_size: int = 32,
    output_dir: str = "outputs",
) -> tuple[L.Trainer, ImageClassifier]:
    """Train an ImageClassifier on CIFAR-10 and return the trainer + model."""
    model = ImageClassifier()
    train_dl, val_dl = get_dataloaders(batch_size=batch_size, max_samples=max_samples)

    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        filename="best-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=output_dir,
        enable_progress_bar=False,
        logger=False,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(model, train_dl, val_dl)
    return trainer, model


if __name__ == "__main__":
    train()
