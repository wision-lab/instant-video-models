#!/usr/bin/env bash

c="_salt_pepper_noise"
name=evaluate_spatial_ema_stabilizer_ce${c}
weights_filepath=$(ls -1 "outputs/deeplab_viper/train_spatial_ema_stabilizer_ce${c}"/*/*/weights_best.pth | sort | tail -1)
./scripts/evaluate.py \
    "runs/deeplab_viper/${name}.yml" \
    override_weights_filepath="\${_original_dir}/$weights_filepath" \
    _name="${name}"
./scripts/evaluate.py \
    "runs/deeplab_viper/evaluate_spatial_ema_stabilizer_ce${c}.yml" \
    stabilizers=[] \
    controllers=[] \
    _name="${name}_baseline"
