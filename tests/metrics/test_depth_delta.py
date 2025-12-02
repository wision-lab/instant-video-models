#!/usr/bin/env python3

import pytest
import torch

from stability.metrics.depth_delta import DepthDelta


def test_depth_delta():
    threshold = 3.0
    disparity_pred = torch.tensor([0.1, 0.5])
    disparity_true = torch.tensor([0.4, 0.2])
    metric = DepthDelta(assume_batch_dim=False, threshold=threshold)
    expected = (1.0 + 0.0) / 2.0
    metric.update(disparity_pred, disparity_true)
    actual = metric.compute()
    assert expected == pytest.approx(actual, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
