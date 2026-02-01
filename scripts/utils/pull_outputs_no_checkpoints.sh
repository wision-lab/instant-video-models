#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# cd to the project root directory
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

for host in "$@"
do
  echo "========== $host =========="

  # PyCharm uses -zar by default
  # Use the --update option to prevent overwriting newer results with older ones
  # Ignore *_last.pth files (used when resuming a checkpoint)
  rsync -zar --progress --update "$host:instant-video-models/outputs" . --exclude "*_last.pth"
done
