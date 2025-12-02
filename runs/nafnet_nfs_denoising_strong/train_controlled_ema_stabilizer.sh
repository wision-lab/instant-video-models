#!/usr/bin/env bash

for lambda in 0.1 0.2 0.4 0.8
do
  sbatch -J train_controlled_ema_strong_$lambda ./scripts/train_slurm.sh runs/nafnet_nfs_denoising_strong/train_controlled_ema_stabilizer.yml lambda=$lambda
done
