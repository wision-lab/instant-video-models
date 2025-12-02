#!/usr/bin/env bash

for lambda in 0.1 0.2 0.4 0.8
do
  sbatch -J train_learned_ema_moderate_$lambda ./scripts/train_slurm.sh runs/nafnet_nfs_denoising_moderate/train_learned_ema_stabilizer.yml lambda=$lambda
done
