#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# cd to the project root directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm outputs/videos/segment_*.mp4

./scripts/videos/make_comparison_video.py \
  outputs/videos/jellyfish_extreme/2025_05_23-01_18_57/output.mp4 \
  outputs/videos/segment_01.mp4 \
  -t 'Denoising Under Extreme Noise' \
  -b 'Instability in the base model is most apparent under extreme noise.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/ducks_extreme/2025_05_23-01_09_04/output.mp4 \
  outputs/videos/segment_02.mp4 \
  -t 'Denoising Under Extreme Noise' \
  -b 'Instability in the base model is most apparent under extreme noise.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_extreme/2025_05_23-01_14_33/output.mp4 \
  outputs/videos/segment_03.mp4 \
  -t 'Denoising Under Extreme Noise' \
  -b 'Instability in the base model is most apparent under extreme noise.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_strong/2025_05_23-00_46_38/output.mp4 \
  outputs/videos/segment_04.mp4 \
  -t 'Denoising Under Strong Noise' \
  -b 'At lower levels of noise, differences are more apparent in uniform regions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_moderate/2025_05_23-00_27_05/output.mp4 \
  outputs/videos/segment_05.mp4 \
  -t 'Denoising Under Moderate Noise' \
  -b 'At lower levels of noise, differences are more apparent in uniform regions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_moderate_chunk_drop/2025_05_23-02_14_50/output.mp4 \
  outputs/videos/segment_06.mp4 \
  -t 'Denoising Robustness - Patch Drop' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_moderate_elastic_transform/2025_05_23-02_21_28/output.mp4 \
  outputs/videos/segment_07.mp4 \
  -t 'Denoising Robustness - Elastic Transform' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_moderate_frame_drop/2025_05_23-02_25_54/output.mp4 \
  outputs/videos/segment_08.mp4 \
  -t 'Denoising Robustness - Frame Drop' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_moderate_jpeg_compression/2025_05_23-02_31_02/output.mp4 \
  outputs/videos/segment_09.mp4 \
  -t 'Denoising Robustness - JPEG Artifacts' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_moderate_salt_pepper_noise/2025_05_23-02_37_25/output.mp4 \
  outputs/videos/segment_10.mp4 \
  -t 'Denoising Robustness - Impulse Noise' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_laplacian_strong/2025_05_23-01_31_24/output.mp4 \
  outputs/videos/segment_11.mp4 \
  -t 'Strong Detail Enhancement' \
  -b 'Instability for image enhancement is less prominent, but still present.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_laplacian_moderate/2025_05_23-01_24_29/output.mp4 \
  outputs/videos/segment_12.mp4 \
  -t 'Moderate Detail Enhancement' \
  -b 'Instability for image enhancement is less prominent, but still present.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_laplacian_moderate_chunk_drop/2025_05_23-01_38_39/output.mp4 \
  outputs/videos/segment_13.mp4 \
  -t 'Enhancement Robustness - Patch Drop' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_laplacian_moderate_elastic_transform/2025_05_23-01_41_42/output.mp4 \
  outputs/videos/segment_14.mp4 \
  -t 'Enhancement Robustness - Elastic Transform' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_laplacian_moderate_frame_drop/2025_05_23-01_46_38/output.mp4 \
  outputs/videos/segment_15.mp4 \
  -t 'Enhancement Robustness - Frame Drop' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_laplacian_moderate_jpeg_compression/2025_05_23-01_57_13/output.mp4 \
  outputs/videos/segment_16.mp4 \
  -t 'Enhancement Robustness - JPEG Artifacts' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

./scripts/videos/make_comparison_video.py \
  outputs/videos/highway_laplacian_moderate_salt_pepper_noise/2025_05_23-02_02_57/output.mp4 \
  outputs/videos/segment_17.mp4 \
  -t 'Enhancement Robustness - Impulse Noise' \
  -b 'Stabilizers can be trained for robustness against transient input corruptions.'

rm -f outputs/videos/concat.txt
for f in $(ls outputs/videos/segment_*.mp4 | sort)
do
  b=$(basename "$f")
  echo "file '$b'" >> outputs/videos/concat.txt
done
# Slow down by 3x.
ffmpeg -y -f concat -safe 0 -i outputs/videos/concat.txt -filter:v setpts=3*PTS,fps=20 -c:v libx264 -crf 28 -preset veryslow outputs/videos/full.mp4
