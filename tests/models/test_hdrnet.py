#!/usr/bin/env python3

from pathlib import Path

import pytest
import torch

from stability.models.hdrnet import HDRNet


@pytest.fixture
def model():
    return HDRNet()


def test_basic_forward(model):
    x = torch.zeros(1, 3, 1024, 1024)
    result = model(x)
    assert result.shape == x.shape


def test_load_weights(model):
    weights = torch.load(
        Path(__file__).parents[2]
        / "weights"
        / "hdrnet"
        / "local_laplacian"
        / "normal_1024.pth"
    )
    model.load_state_dict(weights)


if __name__ == "__main__":
    pytest.main([__file__])
