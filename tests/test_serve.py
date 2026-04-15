from mlbench.serve import create_app, CIFAR10_CLASSES


def test_create_app_returns_interface():
    app = create_app()
    assert app is not None


def test_cifar10_classes_has_10_entries():
    assert len(CIFAR10_CLASSES) == 10
    assert "airplane" in CIFAR10_CLASSES
    assert "truck" in CIFAR10_CLASSES
