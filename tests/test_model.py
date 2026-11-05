import torch
from src.model import PowerPlantANN


def test_model_output_shape():
    model = PowerPlantANN()
    assert model(torch.randn(5, 4)).shape == (5, 1)


def test_model_has_trainable_parameters():
    model = PowerPlantANN()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert all(parameter.requires_grad for parameter in model.parameters())
