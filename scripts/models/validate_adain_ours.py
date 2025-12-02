#!/usr/bin/env python3

# This script generates an AdaIN result using our codebase.

from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.transforms.v2.functional import convert_image_dtype
from torchvision.utils import save_image

from stability.models.adain import AdaIN


def main():
    input_dir = Path("extern", "adain", "input")
    output_dir = Path("outputs", "validate_adain")
    weights_dir = Path("weights", "adain")
    output_dir.mkdir(exist_ok=True)

    # In order to get a perfect match with the original authors' codebase, load the
    # style and content images using PIL.Image.open(). When using
    # torchvision.io.read_image() as we do here, we get a handful of small (<=1/255)
    # differences in the output images.
    model = AdaIN(
        input_dir / "style" / "the_resevoir_at_poitiers.jpg",
        weights_filepath=weights_dir / "adain.pth",
    )
    content = convert_image_dtype(
        read_image(str(input_dir / "content" / "chicago.jpg")), torch.float32
    )

    output = model(content.unsqueeze(dim=0)).squeeze(dim=0)
    output_filepath = output_dir / "ours.png"
    save_image(output, str(output_filepath))
    print(f"Saved output to {output_filepath}.")


if __name__ == "__main__":
    main()
