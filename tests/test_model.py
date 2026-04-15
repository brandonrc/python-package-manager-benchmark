import torch


def test_model_forward_shape(sample_model, sample_batch):
    images, _ = sample_batch
    output = sample_model(images)
    assert output.shape == (4, 10)


def test_model_forward_produces_logits(sample_model, sample_batch):
    images, _ = sample_batch
    output = sample_model(images)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_training_step_returns_scalar_loss(sample_model, sample_batch):
    loss = sample_model.training_step(sample_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_validation_step_logs_loss(sample_model, sample_batch):
    sample_model.validation_step(sample_batch, batch_idx=0)
    # Should not raise


def test_configure_optimizers_returns_optimizer(sample_model):
    optimizer = sample_model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
