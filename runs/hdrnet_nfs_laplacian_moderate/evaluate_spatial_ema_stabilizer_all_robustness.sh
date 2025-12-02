#!/usr/bin/env bash

for s in chunk_drop elastic_transform frame_drop gaussian_noise jpeg_compression salt_pepper_noise
do
    weights_filepath=$(ls -1 outputs/hdrnet_nfs_laplacian_moderate/train_spatial_ema_stabilizer_${s}/*/*/weights_best.pth | sort | tail -1)
    sbatch -J laplacian_${s} ./scripts/evaluate_slurm.sh runs/hdrnet_nfs_laplacian_moderate/evaluate_spatial_ema_stabilizer_${s}.yml override_weights_filepath="\${_original_dir}/$weights_filepath"
    sbatch -J laplacian_${s}_baseline ./scripts/evaluate_slurm.sh runs/hdrnet_nfs_laplacian_moderate/evaluate_spatial_ema_stabilizer_${s}_baseline.yml
done
