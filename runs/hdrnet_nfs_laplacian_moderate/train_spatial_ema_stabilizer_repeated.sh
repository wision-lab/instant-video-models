#!/usr/bin/env bash

for seed in {1..8}
do
  sbatch \
    -J "train_spatial_ema_hdrnet_moderate_repeated_$seed" \
    ./scripts/train_slurm.sh \
    runs/hdrnet_nfs_laplacian_moderate/train_spatial_ema_stabilizer.yml \
    "seed=$seed" \
    "_name=train_spatial_ema_stabilizer_repeated_$seed"
done
