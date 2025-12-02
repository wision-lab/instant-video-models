#!/usr/bin/env bash

# Slurm wrapper for evaluate.py. Usage:
# sbatch -J <job-name> ./scripts/evaluate_slurm.sh <run_config> <config_overrides...>
# where <job-name> is the name to give this job, <run_config> is the filepath of the yml
# training config, and <config_overrides> (optional) is a list of key=value overrides.

# To override the time limit, use the -t/--time command-line argument.

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa4500:1
#SBATCH --mem=32GB
#SBATCH --output=outputs/slurm/%x.txt
#SBATCH --partition=research
#SBATCH --time=08:00:00

./scripts/evaluate.py "$@"
