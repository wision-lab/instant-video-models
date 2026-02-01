#!/usr/bin/env bash

# DAVIS dataset download links:
# https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip
# https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017_semantics-Full-resolution.zip

# Exit immediately if any command fails
set -e

# cd to the project root directory
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Prepare a dataset directory
mkdir -p data/davis_2017
cd data/davis_2017

if [[ ! -d DAVIS ]]
then
  # Download images and object annotations
  zip_filename=DAVIS-2017-trainval-Full-Resolution.zip
  wget "https://data.vision.ee.ethz.ch/csergi/share/davis/$zip_filename"
  unzip "$zip_filename"
  rm "$zip_filename"

  # Download object category labels
  zip_filename=DAVIS-2017_semantics-Full-resolution.zip
  wget "https://data.vision.ee.ethz.ch/csergi/share/davis/$zip_filename"
  unzip "$zip_filename"
  rm "$zip_filename"
fi
