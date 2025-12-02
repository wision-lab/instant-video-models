#!/usr/bin/env python3

from pathlib import Path

import torch


def main():
    base_path = Path("weights", "resnet")
    input_weights = torch.load(Path("weights", "resnet", "resnet_v6.pth"))
    output_weights = {}
    for key, value in input_weights.items():
        output_weights[f"model.{key}"] = value
    base_path.mkdir(parents=True, exist_ok=True)
    output_filepath = base_path / "resnet_base.pth"
    torch.save(output_weights, output_filepath)
    print(f"Saved weights to {output_filepath}.")


if __name__ == "__main__":
    main()
