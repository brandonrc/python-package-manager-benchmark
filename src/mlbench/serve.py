"""Gradio inference demo for the CIFAR-10 classifier."""

from __future__ import annotations

import numpy as np
import torch

from mlbench.model import ImageClassifier


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def create_app(model_path: str | None = None):
    """Create a Gradio interface for image classification."""
    import gradio as gr

    if model_path:
        model = ImageClassifier.load_from_checkpoint(model_path)
    else:
        model = ImageClassifier()
    model.eval()

    def predict(image):
        if image is None:
            return {}
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - np.array([0.4914, 0.4822, 0.4465])) / np.array([0.2470, 0.2435, 0.2616])
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        return {CIFAR10_CLASSES[i]: probs[i].item() for i in range(10)}

    app = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=5),
        title="CIFAR-10 Classifier",
    )
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
