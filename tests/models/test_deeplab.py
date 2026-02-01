#!/usr/bin/env python3

import pytest
import torch

from instant_video_models.models.deeplab import Deeplab


def test_basic_forward():
    model = Deeplab()
    model.eval()
    x = torch.zeros(1, 3, 256, 512)
    result = model(x).argmax(dim=1)
    assert result.shape == (1, 256, 512)


if __name__ == "__main__":
    pytest.main([__file__])
