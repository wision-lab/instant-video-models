#!/usr/bin/env python3

from pathlib import Path

import torch


def main():
    corruptions = [
        "chunk_drop",
        "elastic_transform",
        "frame_drop",
        "jpeg_compression",
        "salt_pepper_noise",
    ]
    output_dirpath = Path("weights", "nafnet", "nfs_denoising_moderate")
    output_dirpath.mkdir(parents=True, exist_ok=True)
    for c1 in corruptions:
        for c2 in corruptions:
            if c1 == c2:
                continue
            output_weights = {}
            for i, c in enumerate([c1, c2]):
                base_dirpath = Path(
                    "outputs",
                    "nafnet_nfs_denoising_moderate",
                    f"train_spatial_ema_stabilizer_{c}_composable",
                    "0.2",
                )
                input_filepath = sorted(base_dirpath.iterdir())[-1] / "weights_best.pth"
                for key, value in torch.load(input_filepath).items():
                    key = key.replace(".stabilizer.", f".ensemble_{i + 1}.")
                    key = key.replace(
                        ".controller_backbone.", f".controller_backbone_{i + 1}."
                    )
                    output_weights[key] = value
            output_filepath = output_dirpath / f"composed_{c1}_{c2}.pth"
            torch.save(output_weights, output_filepath)
            print(f"Saved weights to {output_filepath}.")


if __name__ == "__main__":
    main()
