import pandas as pd
import matplotlib.pyplot as plt

# Data for the first model testing on various datasets
data1 = {
    'Steps': [20000, 60000, 230000, 290000, 300000, 330000, 340000, 370000, 400000],
    'Set5': [37.58, 37.78, 37.89, 38.02, 38.03, 38.01, 37.99, 38.04, 38.08],
    'Set14': [33.47, 33.70, 33.78, 33.98, 33.97, 34.00, 33.98, 34.02, 33.99],
    # 'BSDS100': [31.95, 32.05, 32.15, 32.21, 32.21, 32.20, 32.19, 32.21, 32.23],
    # 'Manga109': [37.90, 38.41, 38.65, 39.09, 39.22, 39.05, 39.18, 39.13, 39.12],
    # 'Urban100': [31.76, 32.43, 32.60, 33.06, 33.06, 33.06, 33.04, 33.09, 33.10]
}

# Data for the second model testing on the same datasets
data2 = {
    'Steps': [20000, 60000, 230000, 290000, 300000, 330000, 340000, 370000, 400000],
    'Set5': [38.09,37.99 ,38.12, 37.77, 37.85, 37.32, 37.55, 36.23, 37.62],
    'Set14': [34.00,33.94 ,34.01, 33.77, 32.87, 32.73, 33.06, 32.37, 33.85],
    # 'BSDS100': [31.95, 32.05, 32.15, 32.21, 32.21, 32.20, 32.19, 32.21, 32.23, 32.21],
    # 'Manga109': [37.90, 38.41, 38.65, 39.09, 39.22, 39.05, 39.18, 39.13, 39.12, 39.17],
    # 'Urban100': [31.76, 32.43, 32.60, 33.06, 33.06, 33.06, 33.04, 33.09, 33.10, 33.09]
}



# Convert data to DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Plotting
plt.figure(figsize=(12, 8))

# Define colors and markers for clarity
colors = ['blue', 'green', 'red', 'purple', 'brown']
markers = ['o', '^', 's', 'p', '*', 'x', '+', 'D', '<', '>']

# Iterate over each dataset to create separate plots for each
# for i, dataset in enumerate(['Set5', 'Set14', 'BSDS100', 'Manga109', 'Urban100']):
for i, dataset in enumerate(['Set5', 'Set14']):
    plt.subplot(3, 2, i + 1)
    plt.plot(df1['Steps'], df1[dataset], label='SwinIR', marker=markers[i], color=colors[i])
    plt.plot(df2['Steps'], df2[dataset], label='SwinIRAdapter', linestyle='--', marker=markers[i], color=colors[i])
    plt.title(dataset)
    plt.xlabel('Training Steps')
    plt.ylabel('Average PSNR (dB)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
# Save the plot
plt.savefig('average_psnr_vs_model_step_combined.png')
