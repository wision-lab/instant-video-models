#!/usr/bin/env bash

# Slurm wrapper for train.py. Usage:
# sbatch -J <job-name> ./scripts/train_slurm.sh <run_config> <config_overrides...>
# where <job-name> is the name to give this job, <run_config> is the filepath of the yml
# training config, and <config_overrides> (optional) is a list of key=value overrides.

# To override the time limit, use the -t/--time command-line argument.

# We are getting CUDA/memory errors with rtx4000ada GPUs. Explicitly specify rtxa4500,
# which are known to work. This replaces --constraint=ampere|ada|hopper.

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa4500:1
#SBATCH --mem=32GB
#SBATCH --output=outputs/slurm/%x.txt
#SBATCH --partition=research
#SBATCH --time=2-00:00:00

./scripts/train.py "$@"
