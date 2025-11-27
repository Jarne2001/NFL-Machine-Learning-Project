import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create the training data
# Load in data with features
release_df = pd.read_csv("features_data.csv")

out_pattern = "output_2023_w??.csv"
fps = 10.0   # frames per second

# Load post-pass (output) frames
out_files = sorted(glob.glob("output_2023_w??.csv"))

out_dfs = [pd.read_csv(f, usecols=['game_id','play_id','nfl_id','frame_id','x','y']) for f in out_files]
output_df = pd.concat(out_dfs, ignore_index=True, sort=False)

# Rename output columns (avoid collisions)
output_df = output_df.rename(columns={'frame_id':'output_frame', 'x':'after_pass_x', 'y':'after_pass_y'})

# Make sure the join keys are numeric and consistent
for c in ['game_id','play_id','nfl_id','output_frame']:
    output_df[c] = pd.to_numeric(output_df[c], errors='coerce')
for c in ['game_id','play_id','nfl_id']:
    if c in release_df.columns:
        release_df[c] = pd.to_numeric(release_df[c], errors='coerce')

release_df = release_df.rename(columns={'frame_id':'release_frame'})

# Drop weird rows with missing keys
output_df = output_df.dropna(subset=['game_id','play_id','nfl_id','output_frame'])
release_df = release_df.dropna(subset=['game_id','play_id','nfl_id'])

output_df[['game_id','play_id','nfl_id','output_frame']] = output_df[['game_id','play_id','nfl_id','output_frame']].astype(int)
release_df[['game_id','play_id','nfl_id']] = release_df[['game_id','play_id','nfl_id']].astype(int)

# Merge datasets (repeats the release row for every post-pass frame row)
training_data = output_df.merge(release_df, on=['game_id','play_id','nfl_id'], how='inner', suffixes=('_out','_rel'))

# Calculate frame offset and time (1 = first post-pass frame)
training_data['frame_offset'] = training_data['output_frame'].astype(int)
training_data['time_s'] = training_data['frame_offset'] / fps

print("Merged rows:", len(training_data))
print("Example header:")
print(training_data.head(3)[['game_id','play_id','nfl_id','output_frame','frame_offset','time_s', 'after_pass_x','after_pass_y','x','y']])
training_data.head()
frame_counts = training_data.groupby(['game_id','play_id','nfl_id'])['frame_offset'].nunique()
print("Number of frames per player (first 10):")
print(frame_counts.head(10))

# Training_data checks
# Ensure that most players have more than 1 frame
print("Fraction of players with >1 frame:", (frame_counts > 1).mean())
# Right incrementals?
offsets_ok = training_data.groupby(['game_id','play_id','nfl_id'])['frame_offset'].apply(lambda x: (x.min()==1) and (x.is_monotonic_increasing))
print("All players have monotonic increasing frame_offset starting at 1:", offsets_ok.all())
# Missing values?
missing_targets = training_data[['after_pass_x','after_pass_y']].isna().mean() * 100
print("Percent missing target values:\n", missing_targets)

training_data.to_csv("final_training_data.csv", index=False)
