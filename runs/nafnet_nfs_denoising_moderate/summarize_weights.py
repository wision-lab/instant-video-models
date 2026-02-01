#!/usr/bin/env python3

from pathlib import Path
import torch


def main():
    result_dir = Path(
        "outputs",
        "nafnet_nfs_denoising_moderate",
        "train_simple_learned",
        "0.2",
    )
    print("| Module name | Feature dimension | Logit mean |")
    print("| --- | --- | --- |")
    w = torch.load(next(result_dir.iterdir()) / "weights_best.pth", map_location="cpu")
    for s in ".encoders.", ".middle_blks.", ".decoders.", "module.stabilizer.":
        for k, v in w.items():
            if (s not in k) or (".stabilizer." not in k):
                continue
            print(
                f"| {k.replace('.stabilizer.logits', '')} | {v.shape[1]} | {v.mean():.2f} |"
            )


if __name__ == "__main__":
    main()
