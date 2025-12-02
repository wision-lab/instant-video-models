#!/usr/bin/env bash

for s in rain snow
do
    weights_filepath=$(ls -1 outputs/nafnet_robust_spring_denoising_moderate/train_spatial_ema_stabilizer_${s}_unfrozen_ablation/*/*/weights_best.pth | sort | tail -1)
    ./scripts/evaluate.py \
        runs/nafnet_robust_spring_denoising_moderate/evaluate_base_model.yml \
        _name=evaluate_spatial_ema_stabilizer_${s}_unfrozen_ablation \
        corruption_type=$s \
        override_weights_filepath="\${_original_dir}/$weights_filepath"
done
