"""
File creating the final_training_model.csv file used to train the CatBoost model
for the Kaggle Big Data Bowl Competition.

Uses 2 main functions, 1 for feature engineering and 1 for merging both input and output (containing target) data.
"""

import pandas as pd
import joblib
import numpy as np

# Load in pre-made data files
input_data = pd.read_csv("training_data.csv")
output_data = pd.read_csv("output_data.csv")
output_data.head

model_features = [
    # Player-related and movement features
    'pre_pass_x', 'pre_pass_y', 's', 'a', 'vx', 'vy', 'dir', 'o', 'dir_rad',
    'dir_sin', 'dir_cos', 'o_sin', 'o_cos',
    # Player physical features
    'player_height_inches', 'player_weight_kg', 'player_age',
    'age_speed', 'weight_momentum_x', 'weight_momentum_y','player_bmi',
    # Movement deltas (for motion tracking)
    'dx_last', 'dy_last', 'ds_last', 'ddir_last','distance_moved',
    # Player roles and positioning (categorical features)
    'player_position', 'player_role', 'player_side', 
    'is_targeted', 'is_passer', 'is_defense', 'is_offense', 'is_coverage', 'defensive_coverage',
    'offensive_side', 'passing_role', 'role_targeted_receiver', 
    # Distance and angle to ball (important for ball tracking)
    'endzone_distance', 'time_in_air', 'ball_land_x', 'ball_land_y',
    'distance_to_ball', 'ball_dx', 'ball_dy', 'ball_angle', 'angle_diff_ball', 'eta_to_ball',
    # Features involving projection and heading alignment
    'heading_alignment', 'projection_x', 'projection_y', 'projection_distance_to_ball',
    # Defensive-related features
    'min_defender_dist_to_landing', 'n_def_within_5',
    'min_defender_to_receiver_dist', 'min_defender_speed', 'relative_speed_nearest_defender',
    # Player stats based on velocity and acceleration
    'player_avg_speed', 'player_avg_acceleration',
    # Momentum-related features (player's motion dynamics)
    'momentum_x', 'momentum_y',
    # New features from interaction terms and physics-based features
    'distance_nearest_offensive', 'mean_distance_offensive', 'number_offensive', 'angle_to_offensive',
    # Additional dynamics-based features
    'squared_speed', 'acceleration_x', 'acceleration_y', 
    'combined_acceleration', 'speed_m_s', 'kinetic_energy',
    # Field positioning and distance from endzones
    'left_distance', 'right_distance'
]

def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Preprocess the input test data with applied feature engineering"""
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    df = df.copy()

    # Normalizing play direction (otherwise model must learn symmetry) (move to the right)
    # This reduces variance and speeds up training
    rad = np.deg2rad(df['dir'].fillna(0).to_numpy())
    df['vx'] = df['s'].fillna(0).to_numpy() * np.cos(rad)
    df['vy'] = df['s'].fillna(0).to_numpy() * np.sin(rad)

    # Canonicalize: make offense always move to +x
    mask_left = df['play_direction'] == 'left'
    # Mirror x coordinates across field length = 120
    df.loc[mask_left, 'x'] = 120.0 - df.loc[mask_left, 'x']
    df.loc[mask_left, 'ball_land_x'] = 120.0 - df.loc[mask_left, 'ball_land_x']
    # Mirror horizontally longitudinal velocity
    df.loc[mask_left, 'vx'] = -df.loc[mask_left, 'vx']

    # Recalculate 'dir' after mirroring
    new_rad = np.arctan2(df['vy'], df['vx'])
    df['dir'] = (np.rad2deg(new_rad) % 360)

    # Feature engineering
    teammate_radius = 5.0
    linear_projection = 1.0
    eps = 1e-6
    fps = 10

    # Compute total height in inches + other player-based features
    height = df['player_height'].str.split('-', expand=True)
    feet = pd.to_numeric(height[0], errors='coerce')
    inches = pd.to_numeric(height[1], errors='coerce')

    df['player_height_inches'] = feet * 12 + inches
    df['player_birth_date'] = pd.to_datetime(df['player_birth_date'], errors='coerce')
    df['player_weight_kg'] = df['player_weight'] * 0.45359237
    df['player_age'] = pd.to_datetime('2023-01-01').year - df['player_birth_date'].dt.year
    df['player_height_meters'] = df['player_height_inches'] * 0.0254
    df['player_bmi'] = df['player_weight_kg'] / (df['player_height_meters'] ** 2)

    # Angles because models don't enjoy wrapping around angles
    rad = np.deg2rad(df['dir'])
    df['dir_sin'] = np.sin(rad)
    df['dir_cos'] = np.cos(rad)

    rad_o = np.deg2rad(df['o'].fillna(0))
    df['o_sin'] = np.sin(rad_o)
    df['o_cos'] = np.cos(rad_o)

    last_frame = df['frame_id'] == df.groupby(['game_id','play_id','nfl_id'])['frame_id'].transform('max')
    release_df = df[last_frame].copy().reset_index(drop=True)

    # Linear projection features 
    release_df['projection_x'] = release_df['x'] + release_df['vx'] * linear_projection
    release_df['projection_y'] = release_df['y'] + release_df['vy'] * linear_projection
    release_df['projection_x'] = release_df['projection_x'].clip(0, 120)
    release_df['projection_y'] = release_df['projection_y'].clip(0, 53.3)

    # Ball landing + features
    if 'ball_land_x' in release_df.columns:
        release_df['ball_dx'] = release_df['ball_land_x'] - release_df['x']
        release_df['ball_dy'] = release_df['ball_land_y'] - release_df['y']
        release_df['distance_to_ball'] = np.hypot(release_df['ball_dx'], release_df['ball_dy'])
        release_df['ball_angle'] = np.arctan2(release_df['ball_dy'], release_df['ball_dx'])   # radians
        # Signed angular difference in radians, in [-pi, pi]
        release_df['dir_rad'] = np.deg2rad(release_df['dir'])
        release_df['angle_diff_ball'] = ((release_df['ball_angle'] - release_df['dir_rad'] + np.pi) % (2*np.pi)) - np.pi
        release_df['eta_to_ball'] = release_df['distance_to_ball'] / release_df['s'].replace(0, eps)
        release_df['eta_to_ball'] = release_df['eta_to_ball'].clip(upper=15)
        speed_norm = np.hypot(release_df['vx'], release_df['vy']).replace(0, eps)
        release_df['heading_alignment'] = (release_df['vx'] * release_df['ball_dx'] + release_df['vy'] * release_df['ball_dy']) / (speed_norm * release_df['distance_to_ball'].replace(0, eps))
        release_df['projection_distance_to_ball'] = np.hypot(release_df['projection_x'] - release_df['ball_land_x'], 
                                                  release_df['projection_y'] - release_df['ball_land_y'])

    # Deltas from previous frames
    df_sorted = df.sort_values(['game_id','play_id','nfl_id','frame_id'])

    def last_deltas(g):
        minimum_frames = 3
        available_frames = len(g)
        last_n = min(available_frames, minimum_frames)
        g = g.tail(last_n)
        out = {'dx_last': np.nan, 'dy_last': np.nan, 'ds_last': np.nan, 'ddir_last': np.nan, 
               'distance_moved': np.nan, 'time_in_air': np.nan}
        if len(g) >= 2:
            out['dx_last'] = g['x'].iloc[-1] - g['x'].iloc[0]
            out['dy_last'] = g['y'].iloc[-1] - g['y'].iloc[0]
            out['ds_last'] = g['s'].iloc[-1] - g['s'].iloc[0]
            dx = g['x'].iloc[-1] - g['x'].iloc[0]
            dy = g['y'].iloc[-1] - g['y'].iloc[0]
            out['distance_moved'] = np.sqrt(dx**2 + dy**2)
            # wrap angle difference to [-180,180]
            out['ddir_last'] = ((g['dir'].iloc[-1] - g['dir'].iloc[0] + 180) % 360) - 180
            # number of frames the ball is in the air
            num_frames = g['num_frames_output'].iloc[0]
        return pd.Series(out)

    deltas = df_sorted.groupby(['game_id','play_id','nfl_id']).apply(last_deltas).reset_index()
    release_df = release_df.merge(deltas, on=['game_id','play_id','nfl_id'], how='left')

    # Player roles
    release_df['is_targeted'] = (release_df['player_role'] == 'Targeted Receiver').astype(int)
    release_df['is_passer'] = (release_df['player_role'] == 'Passer').astype(int)
    release_df['is_defense'] = (release_df['player_side'] == 'Defense').astype(int)
    release_df['role_targeted_receiver'] = release_df['is_targeted']
    release_df['is_offense'] = (release_df['player_side'] == 'Offense').astype(int)
    release_df['is_coverage'] = (release_df['player_role'] == 'Defensive Coverage').astype(int)
    release_df['defensive_coverage'] = release_df['is_coverage']
    release_df['offensive_side'] = release_df['is_offense']
    release_df['passing_role'] = release_df['is_passer']

    # Smallest distance from any defender to ball landing + count within radius
    defenders = release_df[release_df['player_side'] == 'Defense'].copy()
    if len(defenders) > 0:
        defenders['dist_def_to_ball'] = np.hypot(defenders['x'] - defenders['ball_land_x'], defenders['y'] - defenders['ball_land_y'])
        defence_min = defenders.groupby(['game_id','play_id'])['dist_def_to_ball'].min().rename('min_defender_dist_to_landing').reset_index()
        defence_count = defenders.assign(within_R=(defenders['dist_def_to_ball'] <= teammate_radius)).groupby(['game_id','play_id'])['within_R'].sum().rename(f'n_def_within_{int(teammate_radius)}').reset_index()
    else:
        defence_min = pd.DataFrame(columns=['game_id','play_id','min_defender_dist_to_landing'])
        defence_count = pd.DataFrame(columns=['game_id','play_id', f'n_def_within_{int(teammate_radius)}'])

    release_df = release_df.merge(defence_min, on=['game_id','play_id'], how='left')
    release_df = release_df.merge(defence_count, on=['game_id','play_id'], how='left')
    release_df[f'n_def_within_{int(teammate_radius)}'] = release_df[f'n_def_within_{int(teammate_radius)}'].fillna(0).astype(int)

    # Nearest defender
    min_defender_to_receiver_dist = []
    min_defender_speed = []
    relative_speed_nearest_defender = []
    avg_speed = []
    avg_acceleration = []

    # Group by play
    for (gid, pid), grp in release_df.groupby(['game_id','play_id']):
        players_idx = grp.index.values
        defs = grp[grp['player_side']=='Defense']

        px = grp['x'].to_numpy()
        py = grp['y'].to_numpy()
        pv = grp['s'].to_numpy()

        if defs.shape[0] == 0:
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

    # Teammate-based offensive interactions features
    dist_nearest_off = []
    mean_dist_off = []
    n_off_within_R = []
    angle_to_off = []

    # Group by play to compute interactions within each play
    for (gid, pid), grp in release_df.groupby(['game_id', 'play_id']):
        players_idx = grp.index.values
        off_players = grp[grp['player_side'] == 'Offense']

        if len(off_players) == 0:
            dist_nearest_off.extend([np.nan] * len(grp))
            mean_dist_off.extend([np.nan] * len(grp))
            n_off_within_R.extend([0] * len(grp))
            angle_to_off.extend([np.nan] * len(grp))
            continue

        px = grp['x'].to_numpy()
        py = grp['y'].to_numpy()
        dir_rad = np.deg2rad(grp['dir'].to_numpy())

        off_x = off_players['x'].to_numpy()
        off_y = off_players['y'].to_numpy()

        for i in range(len(px)):
            dists_off = np.hypot(px[i] - off_x, py[i] - off_y)
            if grp.iloc[i]['player_side'] == 'Offense':
                self_idx = np.where((off_x == px[i]) & (off_y == py[i]))[0][0]
                dists_off[self_idx] = np.inf
            nearest_dist = np.min(dists_off)
            if nearest_dist == np.inf:
                nearest_dist = np.nan
            dist_nearest_off.append(nearest_dist)

            valid_dists = dists_off[dists_off != np.inf]
            if len(valid_dists) > 0:
                mean_dist = np.mean(valid_dists)
            else:
                mean_dist = np.nan
            mean_dist_off.append(mean_dist)

            n_off_within_R.append(np.sum(dists_off <= teammate_radius))

            nearest_idx = np.argmin(dists_off)
            angle = np.arctan2(off_y[nearest_idx] - py[i], off_x[nearest_idx] - px[i]) - dir_rad[i]
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            angle_to_off.append(angle)

    release_df['distance_nearest_offensive'] = dist_nearest_off
    release_df['mean_distance_offensive'] = mean_dist_off
    release_df['number_offensive'] = n_off_within_R
    release_df['angle_to_offensive'] = angle_to_off

    # Physics-based calculated features
    release_df['squared_speed'] = release_df['s'] ** 2
    release_df['acceleration_x'] = release_df['a'] * np.cos(rad)
    release_df['acceleration_y'] = release_df['a'] * np.sin(rad)

    release_df['momentum_x'] = release_df['vx'] * 0.9144 * release_df['player_weight_kg']
    release_df['momentum_y'] = release_df['vy'] * 0.9144 * release_df['player_weight_kg']
    release_df['combined_acceleration'] = np.sqrt(release_df['acceleration_x']**2 + release_df['acceleration_y']**2)
    release_df['speed_m_s'] = release_df['s'] * 0.9144
    release_df['kinetic_energy'] = 0.5 * release_df['player_weight_kg'] * release_df['speed_m_s']**2

    release_df['age_speed'] = release_df['player_age'] * release_df['s']
    release_df['weight_momentum_x'] = release_df['player_weight_kg'] * release_df['momentum_x']
    release_df['weight_momentum_y'] = release_df['player_weight_kg'] * release_df['momentum_y']
    
    release_df['endzone_distance'] = 120.0 - release_df['x']
    release_df['left_distance'] = release_df['y']
    release_df['right_distance'] = 53.3 - release_df['y']

    release_df[['min_defender_dist_to_landing','min_defender_to_receiver_dist',
            'relative_speed_nearest_defender','min_defender_speed', 'mean_distance_offensive', 'distance_nearest_offensive']] = \
    release_df[['min_defender_dist_to_landing','min_defender_to_receiver_dist',
                'relative_speed_nearest_defender','min_defender_speed', 'mean_distance_offensive', 'distance_nearest_offensive']].fillna(0)

    if 'time_in_air' not in release_df.columns:
        release_df['time_in_air'] = 0.0    

    release_df.rename(columns={'x': 'pre_pass_x', 'y': 'pre_pass_y'}, inplace=True)

    print("Feature engineering has been performed!")
    return release_df


def training_data(test: pd.DataFrame, test_input: pd.DataFrame) -> pd.DataFrame:
    """
    Function merging both input and output data files into one.
    The output is the final final_training_model.csv file to train the CatBoost model.
    """
    if not isinstance(test, pd.DataFrame):
        test = test.to_pandas()
    if not isinstance(test_input, pd.DataFrame):
        test_input = test_input.to_pandas()
    release_df_test = preprocess_test_data(test_input)
    
    play_dirs = test_input[['game_id', 'play_id', 'play_direction']].drop_duplicates()
    test = test.merge(play_dirs, on=['game_id', 'play_id'], how='left')
    print(test)

    mask_left = test['play_direction'] == 'left'
    if mask_left.any():
        test.loc[mask_left, 'x'] = 120.0 - test.loc[mask_left, 'x']
    print(test)

    keys = ['game_id', 'play_id', 'nfl_id']
    merge_cols = list(dict.fromkeys(model_features))
    
    predictions_df = test.merge(
        release_df_test[keys + merge_cols],
        on=keys,
        how='left'
    ).copy()
    
    # remove any duplicate columns
    predictions_df = predictions_df.loc[:, ~predictions_df.columns.duplicated()]
  
    time_delta = predictions_df['frame_id'].values * 0.1
    predictions_df['time_in_air'] = time_delta

    predictions_df[model_features] = predictions_df[model_features].fillna(0)

    return predictions_df

final_training_data = training_data(output_data, input_data)
final_training_data.to_csv("final_training_model.csv", index=False)
