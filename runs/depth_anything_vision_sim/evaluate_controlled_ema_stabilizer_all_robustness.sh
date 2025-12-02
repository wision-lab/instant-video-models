#!/usr/bin/env bash

for s in chunk_drop elastic_transform frame_drop gaussian_noise jpeg_compression salt_pepper_noise
do
    weights_filepath=$(ls -1 outputs/depth_anything_vision_sim/train_controlled_ema_stabilizer_${s}/*/*/weights_best.pth | sort | tail -1)
    sbatch -J vision_sim_${s} ./scripts/evaluate_slurm.sh runs/depth_anything_vision_sim/evaluate_controlled_ema_stabilizer_${s}.yml override_weights_filepath="\${_original_dir}/$weights_filepath"
    sbatch -J vision_sim_${s}_baseline ./scripts/evaluate_slurm.sh runs/depth_anything_vision_sim/evaluate_controlled_ema_stabilizer_${s}_baseline.yml
done
