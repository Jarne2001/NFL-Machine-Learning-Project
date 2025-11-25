"""
NFL Data Merging
"""

# Merge all input data into 1 big datafile
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_pattern = os.path.join("input_2023_w??.csv")

csv_files = glob.glob(data_pattern)

dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

training_df = pd.concat(dataframes, ignore_index=True)

expected = {'game_id','play_id','nfl_id','frame_id','x','y','s','a','dir','player_role','ball_land_x','ball_land_y'}
print("Missing expected:", expected - set(training_df.columns))

# range checking
print(training_df['x'].describe(), training_df['y'].describe())

# duplicates?
print("Duplicates:", training_df.duplicated().sum())

# Save to a new CSV
training_df.to_csv(("training_data.csv"), index=False)

print(f"Combined {len(csv_files)} files into training_data.csv")
