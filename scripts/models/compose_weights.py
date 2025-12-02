#!/usr/bin/env python3

from argparse import ArgumentParser

import torch


def main():
    parser = ArgumentParser()
    parser.add_argument("input_filepaths", nargs="+")
    parser.add_argument("output_filepath")
    args = parser.parse_args()
    output_weights = {}
    for i, input_filepath in enumerate(args.input_filepaths):
        input_weights = torch.load(input_filepath)
        for key, value in input_weights.items():
            key = key.replace(".stabilizer.", f".ensemble_{i + 1}.")
            key = key.replace(".controller_backbone.", f".controller_backbone_{i + 1}.")
            output_weights[key] = value
    torch.save(output_weights, args.output_filepath)
    print(f"Saved weights to {args.output_filepath}.")


if __name__ == "__main__":
    main()
