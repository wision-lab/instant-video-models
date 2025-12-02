#!/usr/bin/env bash

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_learned_ema_strong
#SBATCH --mem=32GB
#SBATCH --output=outputs/slurm/%x.txt
#SBATCH --partition=research
#SBATCH --time=1-00:00:00

for lambda in 0.1 0.2 0.4 0.8
do
  weights_filepath=$(ls -1 outputs/nafnet_nfs_denoising_strong/train_learned_ema_stabilizer/$lambda/*/weights_best.pth | sort | tail -1)
  ./scripts/evaluate.py runs/nafnet_nfs_denoising_strong/evaluate_learned_ema_stabilizer.yml lambda=$lambda override_weights_filepath="\${_original_dir}/$weights_filepath"
done
