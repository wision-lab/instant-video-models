#!/usr/bin/env python3

from pathlib import Path

import torch

from stability.models.adain import VGG_LAYERS


def main():
    base_path = Path("weights", "adain")
    decoder_weights = torch.load(base_path / "decoder.pth")
    vgg_normalised_weights = torch.load(base_path / "vgg_normalised.pth")
    output_weights = {}
    for key, value in decoder_weights.items():
        output_weights[f"decoder.{key}"] = value
    for key, value in vgg_normalised_weights.items():
        if int(key.split(".")[0]) >= VGG_LAYERS:
            continue
        output_weights[f"vgg_style.{key}"] = value
        output_weights[f"vgg_content.{key}"] = value
    base_path.mkdir(parents=True, exist_ok=True)
    output_filepath = base_path / "adain.pth"
    torch.save(output_weights, output_filepath)
    print(f"Saved weights to {output_filepath}.")


if __name__ == "__main__":
    main()
