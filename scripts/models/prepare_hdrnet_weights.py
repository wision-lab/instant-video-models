#!/usr/bin/env python3

from pathlib import Path

import tensorflow as tf
import torch


def prepare_weights(checkpoint_path):
    reader = tf.train.load_checkpoint(checkpoint_path)

    output_weights = {
        "guidance_net.0.weight": torch.tensor(reader.get_tensor("inference/guide/ccm"))
        .permute(1, 0)
        .unsqueeze(dim=-1)
        .unsqueeze(dim=-1),
        "guidance_net.0.bias": torch.tensor(
            reader.get_tensor("inference/guide/ccm_bias")
        ),
        "guidance_net.2.shifts": torch.tensor(
            reader.get_tensor("inference/guide/shifts")
        )
        .unsqueeze(dim=2)
        .permute(0, 3, 1, 2, 4),
        "guidance_net.2.slopes": torch.tensor(
            reader.get_tensor("inference/guide/slopes")
        ).permute(0, 3, 1, 2, 4),
    }

    def prepare_linear_weights(input_base, output_base, permutation=(3, 2, 0, 1)):
        output_weights[f"{output_base}.weight"] = torch.tensor(
            reader.get_tensor(f"{input_base}/weights")
        ).permute(permutation)
        try:
            output_weights[f"{output_base}.bias"] = torch.tensor(
                reader.get_tensor(f"{input_base}/biases")
            )
        except tf.errors.NotFoundError:
            pass

    for i, j in (0, 1), (2, 2), (4, 3), (6, 4):
        prepare_linear_weights(
            f"inference/coefficients/splat/conv{j}", f"low_level_net.{i}"
        )
    for i, j in (0, 1), (2, 2):
        prepare_linear_weights(
            f"inference/coefficients/local/conv{j}", f"local_net.{i}"
        )
    for i, j in (0, 1), (2, 2):
        prepare_linear_weights(
            f"inference/coefficients/global/conv{j}", f"global_net.{i}"
        )
    for i, j in (5, 1), (7, 2), (9, 3):
        prepare_linear_weights(
            f"inference/coefficients/global/fc{j}",
            f"global_net.{i}",
            permutation=(1, 0),
        )
    prepare_linear_weights("inference/coefficients/prediction/conv1", "fusion_net")
    prepare_linear_weights("inference/guide/channel_mixing", "guidance_net.3")

    return output_weights


def main():
    base_path = Path("weights", "hdrnet")
    for path in base_path.rglob("*"):
        if not (path / "checkpoint").is_file():
            continue
        output_weights = prepare_weights(path)
        output_filepath = f"{path}.pth"
        torch.save(output_weights, output_filepath)
        print(f"Saved weights to {output_filepath}.")


if __name__ == "__main__":
    main()
