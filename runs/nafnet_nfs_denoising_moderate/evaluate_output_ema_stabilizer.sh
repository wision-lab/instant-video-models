#!/usr/bin/env bash

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa4500:1
#SBATCH --job-name=eval_output_ema_moderate
#SBATCH --mem=32GB
#SBATCH --output=outputs/slurm/%x.txt
#SBATCH --partition=research
#SBATCH --time=1-00:00:00

for alpha in 0.99 0.98 0.95 0.9 0.8 0.6
do
  ./scripts/evaluate.py runs/nafnet_nfs_denoising_moderate/evaluate_output_ema_stabilizer.yml alpha=$alpha
done
