#!/bin/bash

# Set variables at the beginning of the script
model_path='SwinIR_20240928080130'
step=500000
output_file="results/mode1_best.txt"  # New variable for output file name

# Function to run test command in parallel
run_test() {
    local config=$1
    local gpu=$2
    local dataset=$(basename "$config" .yaml)
    
    # Run the command in the background
    # python main_test_swinir.py --config "$config" \
    #     --model "/home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/${step}_model.pth" \
    #     --gpu "$gpu" > "results/${dataset}.log" 2>&1 &
    python main_test_swinir.py --config "$config" \
        --model "/home/mayanze/PycharmProjects/SwinTF/experiments/${model_path}/best_model.pth" \
        --gpu "$gpu" > "results/${dataset}.log" 2>&1 &
}

# Run tests for X2 configuration in parallel
run_test "/home/mayanze/PycharmProjects/SwinTF/config/X2/BSDS100.yaml" 4
run_test "/home/mayanze/PycharmProjects/SwinTF/config/X2/set5.yaml" 5
run_test "/home/mayanze/PycharmProjects/SwinTF/config/X2/Set14test.yaml" 5
run_test "/home/mayanze/PycharmProjects/SwinTF/config/X2/urban100test.yaml" 6
run_test "/home/mayanze/PycharmProjects/SwinTF/config/X2/manga109test.yaml" 7

# Wait for all background processes to finish
wait

# Combine all log files into a single result file
cat results/*.log > "$output_file"

# Clean up individual log files
rm results/*.log

echo "All tests completed. Results saved to $output_file"