#!/bin/bash

# Directory containing the model files
model_dir='/home/mayanze/PycharmProjects/SwinTF/experiments/SwinIR_20240503_113731'
gpu_id='6,7'
yadapt='True'
output_file='urban100output.txt'

# Find all .pth files containing 'model' in their names, sort them numerically, and iterate over each file
find "$model_dir" -name '*model*.pth' | sort -t '_' -k3,3n | while read model_path; do
    echo "Running model: $model_path" >> "$output_file"
    python predict_adapter.py /home/mayanze/PycharmProjects/SwinTF/config/test_config/urban100test.yaml "$model_path" "$gpu_id" "$yadapt" >> "$output_file" 2>&1
done