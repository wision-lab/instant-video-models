#!/usr/bin/env python3

import pytest
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from stability.metrics.ssim import SSIM


@pytest.fixture
def rng():
    generator = torch.Generator()
    generator.manual_seed(42)
    return generator


def test_ssim(rng):
    size = (3, 256, 512)
    pred = torch.rand(size, generator=rng)
    true = torch.rand(size, generator=rng)
    metric = SSIM(max_value=1.0, assume_batch_dim=False)
    expected = structural_similarity_index_measure(
        pred.unsqueeze(dim=0), true.unsqueeze(dim=0), data_range=1.0
    )
    metric.update(pred, true)
    actual = metric.compute()
    assert expected == pytest.approx(actual, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
