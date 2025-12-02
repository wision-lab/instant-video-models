#!/usr/bin/env bash

for s in chunk_drop elastic_transform frame_drop jpeg_compression salt_pepper_noise
do
    weights_filepath=$(ls -1 outputs/nafnet_nfs_denoising_moderate/train_spatial_ema_stabilizer_${s}/*/*/weights_best.pth | sort | tail -1)
    sbatch -t 1-00:00:00 -J moderate_${s} ./scripts/evaluate_slurm.sh runs/nafnet_nfs_denoising_moderate/evaluate_spatial_ema_stabilizer_${s}.yml override_weights_filepath="\${_original_dir}/$weights_filepath"
    sbatch -t 1-00:00:00 -J moderate_${s}_baseline ./scripts/evaluate_slurm.sh runs/nafnet_nfs_denoising_moderate/evaluate_spatial_ema_stabilizer_${s}_baseline.yml
done
