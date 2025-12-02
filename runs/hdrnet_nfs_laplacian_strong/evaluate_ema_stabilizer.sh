#!/usr/bin/env bash

for alpha in 0.99 0.98 0.95 0.9 0.8 0.6
do
  sbatch -J eval_ema_strong_${alpha} ./scripts/evaluate_slurm.sh runs/hdrnet_nfs_laplacian_strong/evaluate_ema_stabilizer.yml alpha=$alpha
done
