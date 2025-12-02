#!/usr/bin/env python3

from pathlib import Path

import torch


def main():
    base_path = Path("weights", "nafnet")
    input_weights = torch.load(Path("weights", "nafnet", "NAFNet-SIDD-width32.pth"))
    input_weights = input_weights["params"]
    output_weights = {}
    for key, value in input_weights.items():
        output_weights[f"module.{key}"] = value
    base_path.mkdir(parents=True, exist_ok=True)
    output_filepath = base_path / "nafnet_sidd_width32.pth"
    torch.save(output_weights, output_filepath)
    print(f"Saved weights to {output_filepath}.")


if __name__ == "__main__":
    main()
