#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# cd to the project root directory
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

for host in "$@"
do
  echo "========== $host =========="

  # PyCharm uses -zar by default
  # Use the --delete option to delete remote files that don't exist locally
  # Ignore the following:
  # - The .git directory
  # - The .gitmodules file (meaningless without .git)
  # - The data, outputs, and weights directories (to avoid remote deletions)
  # - Anything excluded by .gitignore
  rsync -zar --progress --delete . "$host:stability" \
    --exclude '/.git' \
    --exclude '/.gitmodules' \
    --exclude '/data' \
    --exclude '/outputs' \
    --exclude '/weights' \
    --filter ':- .gitignore'
done
