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

# Save to a new CSV
training_df.to_csv(("training_data.csv"), index=False)

print(f"Combined {len(csv_files)} files into training_data.csv")
