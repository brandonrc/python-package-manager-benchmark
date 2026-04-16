"""Model evaluation: metrics, confusion matrix, ONNX export."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchmetrics.functional import accuracy, f1_score


def compute_metrics(
    model: torch.nn.Module,
    dataloader,
    num_classes: int = 10,
) -> dict:
    """Run inference on dataloader and return accuracy, F1, predictions, labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy(all_preds, all_labels, task="multiclass", num_classes=num_classes)
    f1 = f1_score(all_preds, all_labels, task="multiclass", num_classes=num_classes, average="macro")

    return {
        "accuracy": acc.item(),
        "f1_macro": f1.item(),
        "predictions": all_preds.numpy(),
        "labels": all_labels.numpy(),
    }


def plot_confusion_matrix(
    labels,
    predictions,
    output_path: str = "confusion_matrix.png",
    class_names: list[str] | None = None,
) -> str:
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def export_onnx(
    model: torch.nn.Module,
    output_path: str = "model.onnx",
    input_shape: tuple = (1, 3, 32, 32),
) -> str:
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    return output_path
