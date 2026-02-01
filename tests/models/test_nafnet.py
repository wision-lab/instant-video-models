#!/usr/bin/env python3

from pathlib import Path

import pytest
import torch

from instant_video_models.models.nafnet import NAFNet


@pytest.fixture
def model():
    return NAFNet()


def test_basic_forward(model):
    model = model.cpu()
    x = torch.zeros(1, 3, 32, 32).cpu()
    result = model(x)
    assert result.shape == x.shape


def test_load_weights(model):
    weights = torch.load(
        Path(__file__).parents[2] / "weights" / "nafnet" / "nafnet_sidd_width32.pth"
    )
    model.load_state_dict(weights)


if __name__ == "__main__":
    pytest.main([__file__])
