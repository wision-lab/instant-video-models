#!/usr/bin/env bash

for f in runs/nafnet_nfs_denoising_moderate/evaluate_stabilizer_composition/*.yml
do
  name=$(basename "$f" .yml)
  baseline_name=${name}_baseline
  if [[ $name ==  _defaults ]]
  then
    continue
  fi
  sbatch -t 16:00:00 -J "$name" ./scripts/evaluate_slurm.sh "$f" \
    "_name=$name" "override_weights_filepath=\${_original_dir}/weights/nafnet/nfs_denoising_moderate/composed_${name}.pth"
  sbatch -t 16:00:00 -J "$baseline_name" ./scripts/evaluate_slurm.sh "$f" \
    "_name=$baseline_name" "controllers=[]" "stabilizers=[]"
done
