"""
NFL Data Feature Engineering
"""

import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

training_df = pd.read_csv("training_data.csv")

# Normalizing play direction (otherwise model must learn symmetry) (move to the right)
# This reduces variance and speeds up training

df = training_df.copy()

# Velocity components (assumes dir is degrees; adjust if heading convention differs)
rad = np.deg2rad(df['dir'].fillna(0).to_numpy())
df['vx'] = df['s'].fillna(0).to_numpy() * np.cos(rad)
df['vy'] = df['s'].fillna(0).to_numpy() * np.sin(rad)

# Canonicalize: make offense always move to +x
mask_left = df['play_direction'] == 'left'
# Mirror x coordinates across field length = 120
df.loc[mask_left, 'x'] = 120.0 - df.loc[mask_left, 'x']
df.loc[mask_left, 'ball_land_x'] = 120.0 - df.loc[mask_left, 'ball_land_x']

# mirror horizontally longitudinal velocity
df.loc[mask_left, 'vx'] = -df.loc[mask_left, 'vx']

new_rad = np.arctan2(df['vy'], df['vx'])
df['dir'] = (np.rad2deg(new_rad) % 360)

# Feature engineering

defenders_radius=5.0
linear_projection=1.0
eps=1e-6
last_n = 3

height = df['player_height'].str.split('-', expand=True)

# Convert to numeric
feet = pd.to_numeric(height[0], errors='coerce')
inches = pd.to_numeric(height[1], errors='coerce')

# Compute total height in inches
df['player_height_inches'] = feet * 12 + inches

df['player_birth_date'] = pd.to_datetime(df['player_birth_date'], errors='coerce')

df['player_age'] = 2023 - df['player_birth_date'].dt.year

# Angles because models dont enjoy wrapping around angles
rad = np.deg2rad(df['dir'])
df['dir_sin'] = np.sin(rad)
df['dir_cos'] = np.cos(rad)

rad_o = np.deg2rad(df['o'].fillna(0))
df['o_sin'] = np.sin(rad_o)
df['o_cos'] = np.cos(rad_o)
# Obtain release_df frame (last input frame) per player
last_frame = df['frame_id'] == df.groupby(['game_id','play_id','nfl_id'])['frame_id'].transform('max')
release_df = df[last_frame].copy().reset_index(drop=True) 

# Ball landing + features
df['ball_dx'] = df['ball_land_x'] - df['x']
df['ball_dy'] = df['ball_land_y'] - df['y']
df['distance_to_ball'] = np.hypot(df['ball_dx'], df['ball_dy'])
release_df['ball_dx'] = release_df['x'].copy()
release_df['ball_dy'] = release_df['y'].copy()
release_df['ball_dx'] = release_df['ball_land_x'] - release_df['x']
release_df['ball_dy'] = release_df['ball_land_y'] - release_df['y']
release_df['distance_to_ball'] = np.hypot(release_df['ball_dx'], release_df['ball_dy'])
release_df['ball_angle'] = np.arctan2(release_df['ball_dy'], release_df['ball_dx'])   # radians
# signed angular difference in radians, in [-pi, pi]
release_df['dir_rad'] = np.deg2rad(release_df['dir'])
release_df['angle_diff_ball'] = ((release_df['ball_angle'] - release_df['dir_rad'] + np.pi) % (2*np.pi)) - np.pi
release_df['eta_to_ball'] = release_df['distance_to_ball'] / release_df['s'].replace(0, eps)
release_df['eta_to_ball'] = release_df['eta_to_ball'].clip(upper=15)
speed_norm = np.hypot(release_df['vx'], release_df['vy']).replace(0, eps)
release_df['heading_alignment'] = (release_df['vx'] * release_df['ball_dx'] + release_df['vy'] * release_df['ball_dy']) / (speed_norm * release_df['distance_to_ball'].replace(0, eps))

# Linear projection features 
release_df['projection_x'] = release_df['x'] + release_df['vx'] * linear_projection
release_df['projection_y'] = release_df['y'] + release_df['vy'] * linear_projection
release_df['projection_x'] = release_df['projection_x'].clip(0, 120)
release_df['projection_y'] = release_df['projection_y'].clip(0, 53.3)
release_df['projection_distance_to_ball'] = np.hypot(release_df['projection_x'] - release_df['ball_land_x'], 
                                                  release_df['projection_y'] - release_df['ball_land_y'])
release_df['endzone_distance'] = 120.0 - release_df['x']
# Deltas from previous frames
df_sorted = df.sort_values(['game_id','play_id','nfl_id','frame_id'])
def last_deltas(g):
    g = g.tail(last_n)
    out = {'dx_last': np.nan, 'dy_last': np.nan, 'ds_last': np.nan, 'ddir_last': np.nan}
    if len(g) >= 2:
        out['dx_last'] = g['x'].iloc[-1] - g['x'].iloc[0]
        out['dy_last'] = g['y'].iloc[-1] - g['y'].iloc[0]
        out['ds_last'] = g['s'].iloc[-1] - g['s'].iloc[0]
        # wrap angle difference to [-180,180]
        out['ddir_last'] = ((g['dir'].iloc[-1] - g['dir'].iloc[0] + 180) % 360) - 180
    return pd.Series(out)
deltas = df_sorted.groupby(['game_id','play_id','nfl_id']).apply(last_deltas).reset_index()
release_df = release_df.merge(deltas, on=['game_id','play_id','nfl_id'], how='left')


# Player roles
release_df['is_targeted'] = (release_df['player_role'] == 'Targeted Receiver').astype(int)
release_df['is_passer'] = (release_df['player_role'] == 'Passer').astype(int)
release_df['is_defense'] = (release_df['player_side'] == 'Defense').astype(int)

# smallest distance from any defender to ball landing + count within radius
defenders = release_df[release_df['player_side'] == 'Defense'].copy()
if len(defenders) > 0:
    defenders['dist_def_to_ball'] = np.hypot(defenders['x'] - defenders['ball_land_x'], defenders['y'] - defenders['ball_land_y'])
    defence_min = defenders.groupby(['game_id','play_id'])['dist_def_to_ball'].min().rename('min_defender_dist_to_landing').reset_index()
    defence_count = defenders.assign(within_R=(defenders['dist_def_to_ball'] <= defenders_radius)).groupby(['game_id','play_id'])['within_R'].sum().rename(f'n_def_within_{int(defenders_radius)}').reset_index()
else:
    defence_min = pd.DataFrame(columns=['game_id','play_id','min_defender_dist_to_landing'])
    defence_count = pd.DataFrame(columns=['game_id','play_id', f'n_def_within_{int(defenders_radius)}'])

release_df = release_df.merge(defence_min, on=['game_id','play_id'], how='left')
release_df = release_df.merge(defence_count, on=['game_id','play_id'], how='left')
release_df[f'n_def_within_{int(defenders_radius)}'] = release_df[f'n_def_within_{int(defenders_radius)}'].fillna(0).astype(int)

# Nearest defender
min_defender_to_receiver_dist = []
min_defender_speed = []
relative_speed_nearest_defender = []
# Player stats (average speed, average acceleration)
avg_speed = []
avg_acceleration = []

# Group by play
for (gid, pid), grp in release_df.groupby(['game_id','play_id']):
    # players in this play
    players_idx = grp.index.values
    # defenders in this play
    defs = grp[grp['player_side']=='Defense']
    
    px = grp['x'].to_numpy()
    py = grp['y'].to_numpy()
    pv = grp['s'].to_numpy()
    
    if defs.shape[0] == 0:
        # no defenders
        min_defender_to_receiver_dist.extend([np.nan]*len(players_idx))
        min_defender_speed.extend([np.nan]*len(players_idx))
        relative_speed_nearest_defender.extend([np.nan]*len(players_idx))
    else:
        dx = defs['x'].to_numpy()
        dy = defs['y'].to_numpy()
        def_s = defs['s'].to_numpy()
        for i in range(len(px)):
            dists = np.hypot(px[i]-dx, py[i]-dy)
            min_idx = np.argmin(dists)
            min_defender_to_receiver_dist.append(dists[min_idx])
            min_defender_speed.append(def_s[min_idx])
            relative_speed_nearest_defender.append(pv[i] - def_s[min_idx])
    avg_speed.extend([grp['s'].mean()]*len(players_idx))
    avg_acceleration.extend([grp['a'].mean()]*len(players_idx))

release_df['min_defender_to_receiver_dist'] = min_defender_to_receiver_dist
release_df['min_defender_speed'] = min_defender_speed
release_df['relative_speed_nearest_defender'] = relative_speed_nearest_defender
release_df['player_avg_speed'] = avg_speed
release_df['player_avg_acceleration'] = avg_acceleration

release_df['velocity_x'] = release_df['s'] * np.cos(release_df['dir_rad'])
release_df['velocity_y'] = release_df['s'] * np.sin(release_df['dir_rad'])

release_df['momentum_x'] = (release_df['velocity_x'] * release_df['player_weight'])/1000
release_df['momentum_y'] = (release_df['velocity_y'] * release_df['player_weight'])/1000

release_df['left_distance'] = release_df['y']
release_df['right_distance'] = 53.3 - release_df['y']

# Feature evaluation
features = [
    'game_id','play_id','nfl_id','frame_id','x','y','s','a','dir','o','vx','vy',
    'player_position','player_role','player_side',
    'dir_sin','dir_cos','o_sin','o_cos','dir_rad',
    'ball_dx','ball_dy','ball_land_x','ball_land_y','distance_to_ball','ball_angle','angle_diff_ball','eta_to_ball',
    'heading_alignment','projection_x','projection_y','projection_distance_to_ball', 'endzone_distance',
    'dx_last','dy_last','ds_last','ddir_last',
    'min_defender_dist_to_landing', f'n_def_within_{int(defenders_radius)}',
    'min_defender_to_receiver_dist','min_defender_speed','relative_speed_nearest_defender',
    'player_height_inches','player_weight','player_birth_date','player_age','player_avg_speed','player_avg_acceleration',
    'velocity_x','velocity_y','momentum_x','momentum_y']

# check NaN % number
nan_percent = (release_df[expected].isna().mean() * 100).sort_values(ascending=False)
print(nan_percent)

# Summary statistics
print("\n--- Summary Stats ---")
print(release_df[expected[:7]].describe())
print(release_df[expected[7:14]].describe())
print(release_df[expected[14:21]].describe())
print(release_df[expected[21:27]].describe())
print(release_df[expected[27:]].describe())
release_df.dtypes

release_df.to_csv("features_data.csv", index=False)
