#!/usr/bin/env bash

for lambda in 0.1 0.2 0.4 0.8
do
  sbatch -J train_simple_recurrent_vision_sim_$lambda ./scripts/train_slurm.sh runs/depth_anything_vision_sim/train_simple_recurrent_stabilizer.yml lambda=$lambda
done
