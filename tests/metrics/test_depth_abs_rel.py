#!/usr/bin/env python3

import pytest
import torch

from stability.metrics.depth_abs_rel import DepthAbsRel


def test_depth_abs_rel():
    disparity_pred = torch.tensor([0.1, 0.5])
    disparity_true = torch.tensor([0.4, 0.2])
    metric = DepthAbsRel(assume_batch_dim=False)
    expected = (3.0 + 0.6) / 2.0
    metric.update(disparity_pred, disparity_true)
    actual = metric.compute()
    assert expected == pytest.approx(actual, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
