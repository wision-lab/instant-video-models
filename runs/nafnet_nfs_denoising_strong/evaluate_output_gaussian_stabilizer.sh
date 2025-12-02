#!/usr/bin/env bash

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa4500:1
#SBATCH --job-name=eval_output_gaussian_strong
#SBATCH --mem=32GB
#SBATCH --output=outputs/slurm/%x.txt
#SBATCH --partition=research
#SBATCH --time=1-00:00:00

for sigma in 0.5 1.0 2.0 3.0 4.0 6.0
do
  ./scripts/evaluate.py runs/nafnet_nfs_denoising_strong/evaluate_output_gaussian_stabilizer.yml sigma=$sigma
done
