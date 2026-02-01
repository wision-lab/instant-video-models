#!/usr/bin/env python3

import math

import pytest
import torch

from instant_video_models.metrics.transport_instability import (
    TransportInstability,
    transport_distance,
)


@pytest.fixture
def rng():
    generator = torch.Generator()
    generator.manual_seed(42)
    return generator


def test_transport_stability_basic():
    a = [
        [0, 0, 0],
        [0, 8, 0],
        [0, 0, 0],
    ]
    b = [
        [2, 1, 2],
        [1, 0, 1],
        [2, 1, 2],
    ]
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    # Max distance 1.2
    gamma = 0.6

    # Cost to bring each node to the correct state, read in row-major order
    expected = 2 * gamma + 1 + 2 * gamma + 1 + 4 * gamma + 1 + 2 * gamma + 1 + 2 * gamma

    actual_positive = transport_distance(a, b, gamma=gamma)
    assert expected == pytest.approx(actual_positive, abs=1e-4)
    actual_negative = transport_distance(-a, -b, gamma=gamma)
    assert expected == pytest.approx(actual_negative, abs=1e-4)


def test_transport_stability_batched():
    a1 = [
        [1, 2],
        [0, 1],
    ]
    b1 = [
        [0, 1],
        [3, 1],
    ]
    a2 = [
        [3, 3],
        [0, 5],
    ]
    b2 = [
        [3, 5],
        [3, 0],
    ]
    a = torch.tensor([a1, a2], dtype=torch.float32)
    b = torch.tensor([b1, b2], dtype=torch.float32)

    # Max distance 1.2
    gamma = 0.6

    # Cost to bring each node to the correct state, read in row-major order
    expected_1 = 1 + 0 + 3 * gamma + 0
    expected_2 = 0 + 2 + 3 + 0
    expected = torch.tensor([expected_1, expected_2], dtype=torch.float32)

    actual = transport_distance(a, b, gamma=gamma)
    assert expected == pytest.approx(actual, abs=1e-4)


@pytest.mark.parametrize("size", [8, 12, 16, 20])
def test_transport_stability_line(size):
    a = torch.zeros(size, size)
    b = torch.zeros(size, size)
    c = torch.zeros(size, size)

    # Max distance 3.2
    gamma = 1.6

    a[0, :] = 1.0
    b[3, :] = 1.0  # Within max distance
    c[4, :] = 1.0  # Outside max distance
    expected_ab = 3 * size
    actual_ab = transport_distance(a, b, gamma=gamma)
    assert expected_ab == pytest.approx(actual_ab, abs=1e-4)
    expected_ac = 2 * gamma * size
    actual_ac = transport_distance(a, c, gamma=gamma)
    assert expected_ac == pytest.approx(actual_ac, abs=1e-4)


def test_transport_stability_mixed():
    a = [
        [0, 0, -2],
        [4, 0, 0],
        [0, 0, 0],
    ]
    b = [
        [1, 1, -1],
        [0, -1, 0],
        [1, 1, 2],
    ]
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    # Max distance 1.6
    gamma = 0.8

    # Cost to bring each node to the correct state, read in row-major order
    expected = 1 + 1 + gamma + gamma + 0 + 0 + 1 + math.sqrt(2) + 2 * gamma

    actual_positive = transport_distance(a, b, gamma=gamma)
    assert expected == pytest.approx(actual_positive, abs=1e-4)
    actual_negative = transport_distance(-a, -b, gamma=gamma)
    assert expected == pytest.approx(actual_negative, abs=1e-4)


def test_transport_stability_non_square():
    a = [
        [1, 2, 1, 2],
        [3, 2, 1, 2],
    ]
    b = [
        [3, 4, 3, 2],
        [1, 2, 3, 4],
    ]
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    # Max distance 2.2
    gamma = 1.1

    # Cost to bring each node to the correct state, read in row-major order
    expected = 2 + 2 * gamma + 2 * gamma + 0 + 0 + 0 + 2 * gamma + 2 * gamma
    actual = transport_distance(a, b, gamma=gamma)
    assert actual == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("gamma", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_transport_stability_random(rng, gamma):
    size = 8
    for _ in range(10):
        a = torch.rand((size, size), generator=rng)
        b = torch.rand((size, size), generator=rng)

        # We're just checking that there is no exception here.
        transport_distance(a, b, gamma=gamma)


def test_transport_stability_tiled(rng):
    size = 7
    tile_size = 4
    a = torch.rand((size, size), generator=rng)
    b = torch.rand((size, size), generator=rng)
    gamma = 1.6
    expected = 0.0
    for i in range(2):
        i_slice = slice(i * tile_size, (i + 1) * tile_size)
        for j in range(2):
            j_slice = slice(j * tile_size, (j + 1) * tile_size)
            expected += transport_distance(
                a[i_slice, j_slice], b[i_slice, j_slice], gamma=gamma
            )
    actual = transport_distance(a, b, gamma=gamma, tile_size=tile_size)
    assert expected == pytest.approx(actual, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
