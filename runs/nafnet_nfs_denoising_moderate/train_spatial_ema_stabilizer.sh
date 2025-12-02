#!/usr/bin/env bash

for lambda in 0.1 0.2 0.4 0.8 8.0
do
  sbatch -J train_spatial_ema_moderate_$lambda ./scripts/train_slurm.sh runs/nafnet_nfs_denoising_moderate/train_spatial_ema_stabilizer.yml lambda=$lambda
done
