import os
import pandas as pd

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

# Sortt the csv files based on the number in the file name
csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

all_psnrs = []
for csv_file in csv_files:
    step = csv_file.split('_')[-1].split('.')[0]
    df = pd.read_csv(csv_file)
    # Rename the column 'PSNR' to f'{step}_PSNR'
    df.rename(columns={'PSNR': f'{step}_PSNR'}, inplace=True)
    # Get the PSNR column
    psnr_col = df[f'{step}_PSNR']
    data_col = df['Dataset Name']
    all_psnrs.append(psnr_col)

# Concatenate all the PSNR columns into a single dataframe
all_psnrs = pd.concat(all_psnrs, axis=1)

# Add the dataset name column
all_psnrs.insert(0, 'Dataset Name', data_col)

# Save the dataframe to a new csv file
all_psnrs.to_csv('all_psnrs.csv', index=False)
