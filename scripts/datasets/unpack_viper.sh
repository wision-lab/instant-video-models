#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# cd to the project root directory
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Move to the dataset directory.
cd data/viper

# Unpack images and annotations.
for zip_filename in *.zip
do
  completed_filename=${zip_filename}.completed
  if [[ ! -f $completed_filename ]]
  then
    unzip -o "$zip_filename"
    rm "$zip_filename"
    touch "$completed_filename"
  fi
done
