#!/usr/bin/env python3

import pytest
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio

from instant_video_models.metrics.psnr import PSNR


def test_psnr():
    pred = torch.tensor([0.1, 0.5])
    true = torch.tensor([0.4, 0.2])
    metric = PSNR(max_value=1.0, assume_batch_dim=False)
    expected = peak_signal_noise_ratio(pred, true, data_range=1.0)
    metric.update(pred, true)
    actual = metric.compute()
    assert expected == pytest.approx(actual, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
