#!/usr/bin/env bash

for seed in {1..8}
do
  weights_filepath=$(find "outputs/hdrnet_nfs_laplacian_moderate/train_spatial_ema_stabilizer_repeated_${seed}/" -name weights_best.pth | sort | tail -1)
  sbatch \
    -J "evaluate_spatial_ema_hdrnet_moderate_repeated_$seed" \
    ./scripts/evaluate_slurm.sh \
    runs/hdrnet_nfs_laplacian_moderate/evaluate_spatial_ema_stabilizer.yml \
    "lambda=0.4" \
    "override_weights_filepath=\${_original_dir}/$weights_filepath" \
    "seed=$seed" \
    "_name=evaluate_spatial_ema_stabilizer_repeated_$seed"
done
