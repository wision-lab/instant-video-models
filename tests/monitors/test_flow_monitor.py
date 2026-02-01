#!/usr/bin/env python3

import pytest
import torch

from instant_video_models.monitors.flow_monitor import FlowMonitor


@pytest.fixture
def rng():
    generator = torch.Generator()
    generator.manual_seed(42)
    return generator


def test_input_flow_monitor(rng):
    # Don't check the flow values, just confirm that overall behavior is correct
    shape = (1, 3, 128, 256)
    a = torch.rand(shape, generator=rng)
    b = torch.rand(shape, generator=rng)
    monitor = FlowMonitor()
    assert not monitor.available()
    monitor(a)
    assert not monitor.available()
    monitor(b)
    assert monitor.available()
    assert monitor.retrieve().shape == (1, 2, 128, 256)


if __name__ == "__main__":
    pytest.main([__file__])
