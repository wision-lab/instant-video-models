#!/bin/bash

# # Array of epsilon values
# epsilons=(0.01 0.02 0.03 0.04 0.05)

# # Array of decay values
# decays=(1 0.98 0.95 0.9 0.8 0.5)

epsilons=(0)

# Array of decay values
decays=(0.8 0.5)

# Configuration file path
config="configs/finetune_resnet/config.yml"

# Output file
output_file="stabilized_no_reset_outputs.txt"

# Loop through all combinations of epsilon and decay
for epsilon in "${epsilons[@]}"
do
    for decay in "${decays[@]}"
    do
        echo "Running model with epsilon=$epsilon and decay=$decay" | tee -a "$output_file"
        # Run the Python script and append output to the file
        python adversarial/no_reset_stabilized_attack.py --config "$config" --epsilon $epsilon --decay $decay >> "$output_file"
        echo "----------------------------------------" >> "$output_file"
    done
done