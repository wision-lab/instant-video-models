#!/usr/bin/env python3

import pytest
import torch

from instant_video_models.metrics.norm_instability import NormInstability


@pytest.fixture
def rng():
    generator = torch.Generator()
    generator.manual_seed(42)
    return generator


@pytest.mark.parametrize("order", [0, 1, 2, float("inf")])
def test_norm_stability(order, rng):
    shape = (5, 6, 7)
    a = torch.rand(shape, generator=rng)
    b = torch.rand(shape, generator=rng)
    metric = NormInstability(assume_batch_dim=False, order=order)
    expected = torch.linalg.vector_norm(a - b, ord=order)
    metric.update(a)
    metric.update(b)
    actual = metric.compute()
    assert expected == pytest.approx(actual, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
