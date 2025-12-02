#!/usr/bin/env bash

#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_controlled_ema_davis
#SBATCH --mem=32GB
#SBATCH --output=outputs/slurm/%x.txt
#SBATCH --partition=research
#SBATCH --time=1-00:00:00

for lambda in 0.1 0.2 0.4 0.8
do
  if compgen -G "outputs/nafnet_davis_denoising/evaluate_controlled_ema_stabilizer/$lambda/*/metrics.csv" > /dev/null
  then
    continue
  fi
  weights_filepath=$(ls -1 outputs/nafnet_davis_denoising/train_controlled_ema_stabilizer/$lambda/*/weights_best.pth | sort | tail -1)
  ./scripts/evaluate.py runs/nafnet_davis_denoising/evaluate_controlled_ema_stabilizer.yml lambda=$lambda override_weights_filepath="\${_original_dir}/$weights_filepath"
done
