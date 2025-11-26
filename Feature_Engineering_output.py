"""
NFL Data Merging — OUTPUT version
"""

import pandas as pd
import glob
import os

base_path = r"C:\Users\thoml\OneDrive\Documenten\Uhasselt master jaar 2 sem 1\Machine learning\Kaggle competition\nfl-big-data-bowl-2026-prediction\train"

data_pattern = os.path.join(base_path, "*.csv")
csv_files = glob.glob(data_pattern)

print(f"Found {len(csv_files)} CSV files")

dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# merge alle files
training_df = pd.concat(dataframes, ignore_index=True)

expected = {'game_id','play_id','nfl_id','frame_id','x','y'}

print("Missing expected columns:", expected - set(training_df.columns))

# check ranges
print("\nX range:")
print(training_df['x'].describe())

print("\nY range:")
print(training_df['y'].describe())

# duplicates?
dups = training_df.duplicated().sum()
print(f"\nDuplicates: {dups}")

# save output CSV
output_path = os.path.join(base_path, "training_data_output.csv")
training_df.to_csv(output_path, index=False)

print(f"\nCombined {len(csv_files)} files into {output_path}")

