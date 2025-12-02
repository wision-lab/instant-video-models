#!/usr/bin/env python3

from pathlib import Path

import torch
from torchvision.transforms.v2 import ElasticTransform
from torchvision.utils import save_image


def main():
    input_image = torch.zeros((3, 256, 256))
    grid_size = 8
    for i in range(grid_size):
        input_image[:, i :: grid_size * 2] = 1.0
    transform = ElasticTransform(alpha=50.0, sigma=5.0)
    transformed_image = transform(input_image)

    plots_dir = Path("outputs", "plots")
    plots_dir.mkdir(exist_ok=True)
    save_image(input_image, plots_dir / "elastic_transform_input.png")
    save_image(transformed_image, plots_dir / "elastic_transform_output.png")


if __name__ == "__main__":
    main()
