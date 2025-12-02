#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# Prompt for confirmation as the --delete option is destructive.
read -r -p \
'This command will delete remote files that do not exist locally. Be sure to pull
outputs from the remote before proceeding. Type "y" to proceed: ' confirm
if [[ $confirm != y ]]
then
  echo "Stopping."
  exit 1
fi

# cd to the project root directory
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

for host in "$@"
do
  echo "========== $host =========="

  # PyCharm uses -zar by default
  # Use the --update option to prevent overwriting newer results with older ones
  # Use the --delete option to delete remote files that don't exist locally
  # Use the --existing option to skip creating new remote files
  rsync -zar --progress --update --delete --existing outputs "$host:stability"
done
