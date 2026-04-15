import os

import torch

from mlbench.evaluate import compute_metrics, plot_confusion_matrix, export_onnx


def test_compute_metrics_returns_expected_keys(sample_model):
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    dataloader = [(images[:4], labels[:4]), (images[4:], labels[4:])]

    metrics = compute_metrics(sample_model, dataloader, num_classes=10)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "predictions" in metrics
    assert "labels" in metrics


def test_compute_metrics_values_in_range(sample_model):
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    dataloader = [(images, labels)]

    metrics = compute_metrics(sample_model, dataloader, num_classes=10)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_plot_confusion_matrix_saves_file(tmp_path):
    labels = [0, 1, 2, 0, 1, 2, 3, 4]
    predictions = [0, 1, 1, 0, 2, 2, 3, 4]
    output = str(tmp_path / "cm.png")
    result = plot_confusion_matrix(labels, predictions, output_path=output)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0


def test_export_onnx_saves_file(sample_model, tmp_path):
    output = str(tmp_path / "model.onnx")
    result = export_onnx(sample_model, output_path=output)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0
