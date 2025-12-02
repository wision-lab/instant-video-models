#!/usr/bin/env python3

from pathlib import Path

import torch


def main():
    base_path = Path("weights", "deeplab")
    input_weights = torch.load(
        base_path / "best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    )
    input_weights = input_weights["model_state"]
    output_weights = {}
    for key, value in input_weights.items():
        output_weights[f"model.{key}"] = value
    output_filepath = base_path / "deeplab_mobilenet_cityscapes.pth"
    torch.save(output_weights, output_filepath)
    print(f"Saved weights to {output_filepath}.")


if __name__ == "__main__":
    main()
