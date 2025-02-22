#!/bin/bash

# Create a temporary directory for organizing files
mkdir -p temp_pack/dataset/testsets/urban100_noise
mkdir -p temp_pack/config/urban100_test/noise
mkdir -p temp_pack/experiments/SwinIR_20250218160256
mkdir -p temp_pack/nets
mkdir -p temp_pack/data
mkdir -p temp_pack/utils

# Copy model files
cp sam_vit_h_4b8939.pth temp_pack/
cp 001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth temp_pack/
cp 001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth temp_pack/

# Copy experiment model
cp experiments/SwinIR_20250218160256/500000_model.pth temp_pack/experiments/SwinIR_20250218160256/

# Copy test dataset
cp -r dataset/testsets/urban100_noise/hr temp_pack/dataset/testsets/urban100_noise/
cp -r dataset/testsets/urban100_noise/noise_* temp_pack/dataset/testsets/urban100_noise/

# Copy config files
cp -r config/urban100_test/noise/*.yaml temp_pack/config/urban100_test/noise/

# Copy Python files
cp main_adapter_ddp.py temp_pack/
cp main_test_swinir.py temp_pack/
cp run_urban100noise.sh temp_pack/

# Copy necessary modules
cp -r nets/swinir*.py temp_pack/nets/
cp -r data/dataloader.py temp_pack/data/
cp -r data/extract_sam_features.py temp_pack/data/
cp -r utils/utils_dist.py temp_pack/utils/
cp -r utils/utils_data.py temp_pack/utils/
cp -r utils/utils_image.py temp_pack/utils/
cp -r utils/util_calculate_psnr_ssim.py temp_pack/utils/
cp compute_dataset.py temp_pack/

# Remove any .npy files
find temp_pack -name "*.npy" -type f -delete

# Create zip file
zip -r cloud_test_package.zip temp_pack/*

# Clean up
rm -rf temp_pack

echo "Package created as cloud_test_package.zip"