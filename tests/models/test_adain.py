#!/usr/bin/env python3

from pathlib import Path

import pytest
import torch

from instant_video_models.models.adain import AdaIN


@pytest.fixture
def model():
    return AdaIN(
        Path(__file__).parents[2]
        / "extern"
        / "adain"
        / "input"
        / "style"
        / "woman_with_hat_matisse.jpg"
    )


def test_basic_forward(model):
    content = torch.zeros(1, 3, 256, 512)
    result = model(content)
    assert result.shape == content.shape


def test_load_weights(model):
    weights = torch.load(Path(__file__).parents[2] / "weights" / "adain" / "adain.pth")
    model.load_state_dict(weights)


if __name__ == "__main__":
    pytest.main([__file__])
