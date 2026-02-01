#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# cd to the project root directory
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# cd to the vision_sim render directory
cd data/vision_sim/renders

for scene in kitchen1 interior-scene loft bathroom2 cocina-ii bathroom5 italianflat restroom minimarket bath
do
    cd "${scene}/002-00002-01200"
    ffmpeg -y -i frames/frame_%06d.png -c:v libx264 -qp 0 frames.mp4  # Lossless
    cd ../../
done
