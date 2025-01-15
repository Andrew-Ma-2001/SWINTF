import re
import pandas as pd
from collections import defaultdict

def load_text_file(file_path):
    """Load the contents of a text file and return it as a string."""
    with open(file_path, 'r') as file:
        return file.read()

# Load the text from a file
file_path = '/home/mayanze/PycharmProjects/SwinTF/set14_noise_pixelshuffle_network.txt'
text = load_text_file(file_path)

# Regular expression to match the config file and PSNR_Y value
config_pattern = re.compile(r'--config\s+([^\s]+)')
psnr_pattern = re.compile(r'-- Average PSNR_Y/SSIM_Y:\s+([\d.]+) dB')

# Find all matches
configs = config_pattern.findall(text)
psnrs = psnr_pattern.findall(text)

# Dictionary to store PSNR_Y values grouped by config
config_psnr_dict = defaultdict(list)

# Process each config and PSNR_Y value
for config, psnr in zip(configs, psnrs):
    # Extract specific parameters from the config file name
    noise_sig_match = re.search(r'noise_sig_([\d.]+)', config)
    noise_match = re.search(r'_noise_([\d.]+)', config)
    quality_match = re.search(r'_quality_([\d.]+)', config)
    
    # Collect non-zero parameters
    filtered_params = []
    if noise_sig_match and noise_sig_match.group(1) != '0.0':
        filtered_params.append(f"noise_sig {noise_sig_match.group(1)}")
    if noise_match and noise_match.group(1) != '0.0':
        filtered_params.append(f"noise {noise_match.group(1)}")
    if quality_match and quality_match.group(1) != '0':
        filtered_params.append(f"quality {quality_match.group(1)}")
    
    # Determine the config key
    config_key = ', '.join(filtered_params) if filtered_params else "original"
    
    # Append the PSNR_Y value to the list for this config
    config_psnr_dict[config_key].append(psnr)

# Function to extract sorting keys from config keys
def extract_sort_key(config_key):
    if config_key == "original":
        return (("original", float('inf')), )  # Ensure "original" is sorted last
    parts = config_key.split(', ')
    sort_key = []
    for part in parts:
        name, value = part.split()
        sort_key.append((name, float(value)))
    return tuple(sort_key)

# Sort the dictionary keys using the custom sort key
sorted_keys = sorted(config_psnr_dict.keys(), key=extract_sort_key)

# Convert the dictionary to a DataFrame with sorted columns
df = pd.DataFrame({k: pd.Series(config_psnr_dict[k]) for k in sorted_keys})

# Write the DataFrame to a CSV file
df.to_csv('urban100_test_noise_adapter_previous.csv', index=False)

print("CSV file 'psnr_results.csv' has been created.")
