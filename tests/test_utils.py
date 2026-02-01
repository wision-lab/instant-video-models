#!/usr/bin/env python3

import matplotlib as mpl
import pytest

from instant_video_models.utils import ColorGenerator


def test_color_generator():
    cmap_name = "rainbow"
    cmap = mpl.colormaps[cmap_name]
    color_generator = ColorGenerator(cmap_name=cmap_name, convert_to_hex=False)
    positions = [
        0.0,
        1.0,
        0.5,
        0.25,
        0.75,
        0.125,
        0.375,
        0.625,
        0.875,
        0.0625,
        0.1875,
        0.3125,
        0.4375,
        0.5625,
        0.6875,
        0.8125,
        0.9375,
    ]
    colors_expected = [cmap(x) for x in positions]
    colors_actual = []
    for i in range(len(positions)):
        colors_actual.append(color_generator(i))
    assert colors_actual == colors_expected


if __name__ == "__main__":
    pytest.main([__file__])
