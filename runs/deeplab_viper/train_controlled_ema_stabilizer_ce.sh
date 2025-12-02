#!/usr/bin/env bash

for lambda in 0.1 0.2 0.4 0.8 8.0
do
  sbatch -J deeplab_viper_controlled_ce_$lambda -t 4-00:00:00 ./scripts/train_slurm.sh runs/deeplab_viper/train_controlled_ema_stabilizer_ce.yml lambda=$lambda
done
