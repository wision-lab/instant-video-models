#!/usr/bin/env bash

for lambda in 0.1 0.2 0.4 0.8 8.0
do
  sbatch -J train_spatial_ema_hdrnet_moderate_$lambda ./scripts/train_slurm.sh runs/hdrnet_nfs_laplacian_moderate/train_spatial_ema_stabilizer.yml lambda=$lambda
done
