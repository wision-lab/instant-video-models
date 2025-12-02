#!/usr/bin/env bash

# This script generates an AdaIN result using the original authors' codebase.

# Exit immediately if any command fails
set -e

export CUDA_VISIBLE_DEVICES=""

input_dir=./extern/adain/input
output_dir=./outputs/validate_adain
weights_dir=./weights/adain
mkdir -p "$output_dir"

# Setting the style and content sizes to 0 implies keeping the original size.
python ./extern/adain/test.py \
  --content "$input_dir/content/chicago.jpg" \
  --style "$input_dir/style/the_resevoir_at_poitiers.jpg" \
  --output "$output_dir" \
  --decoder "$weights_dir/decoder.pth" \
  --vgg "$weights_dir/vgg_normalised.pth" \
  --content_size 0 \
  --style_size 0 \
  --save_ext ".png"

mv "$output_dir/"{chicago_stylized_the_resevoir_at_poitiers,original}.png

echo "Saved output to $output_dir/original.png."
