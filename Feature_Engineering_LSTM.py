"""
NFL Data Feature Engineering — LSTM-ready (correct version)
Target = player x,y positions from OUTPUT files, not ball landing.
"""

import pandas as pd
import numpy as np
import pickle
import glob

# Configuration
INPUT_PATH = r"C:\Users\thoml\OneDrive\Documenten\Uhasselt master jaar 2 sem 1\Machine learning\Kaggle competition\training_data.csv"
OUTPUT_PATTERN = r"C:\Users\thoml\OneDrive\Documenten\Uhasselt master jaar 2 sem 1\Machine learning\Kaggle competition\nfl-big-data-bowl-2026-prediction\train\output_2023_w??.csv"
history_len = 15
eps = 1e-6

# Load input data
print("Loading input data...")
training_df = pd.read_csv(INPUT_PATH)
df = training_df.copy()

# Load output data (targets)
print("Loading output data...")
output_files = sorted(glob.glob(OUTPUT_PATTERN))
output_df = pd.concat([pd.read_csv(f) for f in output_files], ignore_index=True)
print(f"Searching in: {OUTPUT_PATTERN}")
print(f"Found files: {output_files}")
print(f"Loaded {len(output_files)} output files, {len(output_df):,} rows")

# Normalize play direction
rad = np.deg2rad(df['dir'].fillna(0).to_numpy())
df['vx'] = df['s'].fillna(0).to_numpy() * np.cos(rad)
df['vy'] = df['s'].fillna(0).to_numpy() * np.sin(rad)

mask_left = df['play_direction'] == 'left'
df.loc[mask_left, 'x'] = 120.0 - df.loc[mask_left, 'x']
df.loc[mask_left, 'ball_land_x'] = 120.0 - df.loc[mask_left, 'ball_land_x']
df.loc[mask_left, 'vx'] = -df.loc[mask_left, 'vx']

new_rad = np.arctan2(df['vy'], df['vx'])
df['dir'] = (np.rad2deg(new_rad) % 360)

# Apply same normalization to output data
play_dirs = training_df[['game_id', 'play_id', 'play_direction']].drop_duplicates()
output_df = output_df.merge(play_dirs, on=['game_id', 'play_id'], how='left')

mask_left_out = output_df['play_direction'] == 'left'
output_df.loc[mask_left_out, 'x'] = 120.0 - output_df.loc[mask_left_out, 'x']

# Feature engineering
height = df['player_height'].str.split('-', expand=True)
feet = pd.to_numeric(height[0], errors='coerce')
inches = pd.to_numeric(height[1], errors='coerce')
df['player_height_inches'] = feet * 12 + inches

df['player_birth_date'] = pd.to_datetime(df['player_birth_date'], errors='coerce')
df['player_age'] = 2023 - df['player_birth_date'].dt.year

df['dir_sin'] = np.sin(np.deg2rad(df['dir']))
df['dir_cos'] = np.cos(np.deg2rad(df['dir']))
df['o_sin'] = np.sin(np.deg2rad(df['o'].fillna(0)))
df['o_cos'] = np.cos(np.deg2rad(df['o'].fillna(0)))

df['ball_dx'] = df['ball_land_x'] - df['x']
df['ball_dy'] = df['ball_land_y'] - df['y']
df['distance_to_ball'] = np.hypot(df['ball_dx'], df['ball_dy'])

df['dir_rad'] = np.deg2rad(df['dir'])

df['eta_to_ball'] = (df['distance_to_ball'] / df['s'].replace(0, eps)).clip(upper=15)

speed_norm = np.hypot(df['vx'], df['vy']).replace(0, eps)
df['heading_alignment'] = (
    df['vx'] * df['ball_dx'] + df['vy'] * df['ball_dy']
) / (speed_norm * df['distance_to_ball'].replace(0, eps))

df['projection_x'] = (df['x'] + df['vx']).clip(0, 120)
df['projection_y'] = (df['y'] + df['vy']).clip(0, 53.3)

df['is_targeted'] = (df['player_role'] == 'Targeted Receiver').astype(int)
df['is_passer'] = (df['player_role'] == 'Passer').astype(int)
df['is_defense'] = (df['player_side'] == 'Defense').astype(int)

# Features used for LSTM input sequences
sequence_features = [
    'x', 'y', 's', 'a', 'vx', 'vy',
    'dir_sin', 'dir_cos', 'o_sin', 'o_cos',
    'ball_dx', 'ball_dy', 'ball_land_x', 'ball_land_y',
    'distance_to_ball', 'eta_to_ball', 'heading_alignment',
    'projection_x', 'projection_y',
    'is_targeted', 'is_passer', 'is_defense',
    'player_height_inches', 'player_weight', 'player_age'
]

n_features = len(sequence_features)

# Create LSTM sequences
print("Creating LSTM sequences...")

df_sorted = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
output_grouped = output_df.groupby(['game_id', 'play_id', 'nfl_id'])

X_list = []
y_list = []
metadata = []
skipped = 0

for (game_id, play_id, nfl_id), group in df_sorted.groupby(['game_id', 'play_id', 'nfl_id']):
    # Check if target frames exist
    try:
        output_group = output_grouped.get_group((game_id, play_id, nfl_id))
        output_group = output_group.sort_values('frame_id')
    except KeyError:
        skipped += 1
        continue

    group = group.tail(history_len)
    seq = group[sequence_features].values
    seq = np.nan_to_num(seq, nan=0.0)

    n_frames = seq.shape[0]
    seq_padded = np.zeros((history_len, n_features), dtype=np.float32)
    seq_padded[-n_frames:, :] = seq

    target_xy = output_group[['x', 'y']].values.astype(np.float32)

    X_list.append(seq_padded)
    y_list.append(target_xy)

    metadata.append({
        'game_id': game_id,
        'play_id': play_id,
        'nfl_id': nfl_id,
        'n_input_frames': n_frames,
        'n_output_frames': len(target_xy),
        'last_x': group['x'].iloc[-1],
        'last_y': group['y'].iloc[-1],
        'ball_land_x': group['ball_land_x'].iloc[-1],
        'ball_land_y': group['ball_land_y'].iloc[-1],
        'player_role': group['player_role'].iloc[-1]
    })

print(f"Created {len(X_list):,} samples (skipped {skipped:,} without output)")

# Stack input sequences
X = np.stack(X_list, axis=0)

# Pad target sequences to max length
max_output_frames = max(len(y) for y in y_list)
y = np.zeros((len(y_list), max_output_frames, 2), dtype=np.float32)
for i, target in enumerate(y_list):
    y[i, :len(target), :] = target

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

np.save("X_lstm.npy", X)
np.save("y_lstm.npy", y)

with open("lstm_meta.pkl", "wb") as f:
    pickle.dump({
        'metadata': metadata,
        'feature_names': sequence_features,
        'history_len': history_len,
        'max_output_frames': max_output_frames
    }, f)


