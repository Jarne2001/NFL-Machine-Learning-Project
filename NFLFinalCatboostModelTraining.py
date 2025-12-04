"""
This file saves 2 CatBoost models for both the target coordinates. It uses the final_training_model.csv data file
made before for training. Then, it checks the training and validation RMSE to control overfitting. 
Next, these 2 models are uploaded and used online in the Kaggle Notebook.
These models are then used on the unseen competition data.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings("ignore")

# Configuration
INPUT_CSV = "final_training_model.csv"
N_SPLITS = 5
RANDOM_SEED = 20

# CatBoost parameters
# Tweaked many times to control overfitting while still giving good results.
CB_PARAMS = dict(
    iterations=1500,
    depth=7,
    learning_rate=0.025,
    loss_function='RMSE',
    verbose=200,
    task_type='GPU',
    random_seed=RANDOM_SEED,
    l2_leaf_reg=25,
    random_strength=2,
    bagging_temperature=1,
    od_type='Iter',
    od_wait=200
)

# filenames to save
MODEL_X_PATH = "NFL_x_Catboost.cbm"
MODEL_Y_PATH = "NFL_y_Catboost.cbm"
FEATURES_PATH = "model_features.json"

# Loading in the data
df = pd.read_csv(INPUT_CSV)
print("Loaded dataframe:", df.shape)

# identify groups for GroupKFold
groups = (df['game_id'].astype(str) + '_' + df['play_id'].astype(str)).values

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

df['oof_pred_x'] = np.nan
df['oof_pred_y'] = np.nan

print("Auto-detected model_features ({}): {}".format(len(model_features), model_features[:10]))

# set categorical features
categorical_features = ['player_position', 'player_role', 'player_side']
categorical_features = [c for c in categorical_features if c in model_features]
print("Using categorical features:", categorical_features)

# numeric columns
num_cols = [c for c in model_features if c not in categorical_features]
print("Number numeric features:", len(num_cols))

# reproducibility
np.random.seed(RANDOM_SEED)

# CV with a fold-wise scaler
gkf = GroupKFold(n_splits=N_SPLITS)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[['x','y']], groups=groups)):
    print(f"\n--- Fold {fold} ---")
    # Select feature frames for this fold
    X_tr = df.iloc[train_idx].reset_index(drop=True)[model_features].copy()
    X_va = df.iloc[val_idx].reset_index(drop=True)[model_features].copy()
    y_tr = df.iloc[train_idx].reset_index(drop=True)[['x','y']].copy()
    y_va = df.iloc[val_idx].reset_index(drop=True)[['x','y']].copy()

    # Create Pools with categorical feature names
    pool_tr_x = Pool(X_tr, y_tr['x'], cat_features=categorical_features)
    pool_va_x = Pool(X_va, y_va['x'], cat_features=categorical_features)
    pool_tr_y = Pool(X_tr, y_tr['y'], cat_features=categorical_features)
    pool_va_y = Pool(X_va, y_va['y'], cat_features=categorical_features)

    # instantiate models for this fold
    m_x = CatBoostRegressor(**CB_PARAMS)
    m_y = CatBoostRegressor(**CB_PARAMS)

    # train (use_best_model=True requires early_stopping_rounds or od_type/od_wait to control overfitting)
    m_x.fit(pool_tr_x, eval_set=pool_va_x, use_best_model=True, early_stopping_rounds=100)
    m_y.fit(pool_tr_y, eval_set=pool_va_y, use_best_model=True, early_stopping_rounds=100)

    # predict on validation
    pred_x = m_x.predict(X_va)
    pred_y = m_y.predict(X_va)
    df.loc[val_idx, 'oof_pred_x'] = pred_x
    df.loc[val_idx, 'oof_pred_y'] = pred_y
    preds = np.vstack([pred_x, pred_y]).T

    # compute per-play RMSE then mean across plays (matches competition metric style)
    val_groups = (df.iloc[val_idx]['game_id'].astype(str) + '_' + df.iloc[val_idx]['play_id'].astype(str)).values
    val_df = pd.DataFrame({
        'group': val_groups,
        'true_x': y_va['x'].values, 'true_y': y_va['y'].values,
        'pred_x': preds[:,0], 'pred_y': preds[:,1]
    })
    per_play_rmse = val_df.groupby('group').apply(
        lambda g: mean_squared_error(g[['true_x','true_y']], g[['pred_x','pred_y']])**0.5
    )
    fold_rmse = per_play_rmse.mean()
    fold_metrics.append(fold_rmse)
    print(f"Fold {fold} mean per-play RMSE: {fold_rmse:.6f}")

print("\nCV mean per-play RMSE:", np.mean(fold_metrics))

df['group'] = df['game_id'].astype(str) + '_' + df['play_id'].astype(str)

oof_rmse = df.groupby('group').apply(
    lambda g: mean_squared_error(g[['x','y']], g[['oof_pred_x','oof_pred_y']])**0.5
).mean()

print("OOF per-play RMSE:", oof_rmse)

# Final training models on the full training data
print("\nRetraining on 90% training data with early stopping...")

# Split 90% train / 10% validation
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
train_idx, val_idx = next(gss.split(df, groups=groups))
df_train = df.iloc[train_idx]
df_val   = df.iloc[val_idx]
X_train = df_train[model_features]
X_val = df_val[model_features]
y_train = df_train[['x', 'y']]
y_val = df_val[['x', 'y']]

# CatBoost Pools
pool_train_x = Pool(X_train, y_train['x'], cat_features=categorical_features)
pool_val_x = Pool(X_val, y_val['x'], cat_features=categorical_features)

pool_train_y = Pool(X_train, y_train['y'], cat_features=categorical_features)
pool_val_y = Pool(X_val, y_val['y'], cat_features=categorical_features)

# Final models with early stopping
m_x_final = CatBoostRegressor(**CB_PARAMS)
m_y_final = CatBoostRegressor(**CB_PARAMS)

m_x_final.fit(pool_train_x, eval_set=pool_val_x, use_best_model=True)
m_y_final.fit(pool_train_y, eval_set=pool_val_y, use_best_model=True)

# Save models
m_x_final.save_model(MODEL_X_PATH)
m_y_final.save_model(MODEL_Y_PATH)
print("Saved models:", MODEL_X_PATH, MODEL_Y_PATH)

# Save model features
with open(FEATURES_PATH, "w") as f:
    json.dump(model_features, f)
print("Saved feature list:", FEATURES_PATH)

print("Done.")

# Overfitting
# Training RMSE
pred_x = m_x_final.predict(X_train)  # predictions for x
pred_y = m_y_final.predict(X_train)  # predictions for y

# Compute per-play RMSE (stack x and y)
train_rmse = np.sqrt(mean_squared_error(y_train[['x','y']], np.column_stack([pred_x, pred_y])))
print("Train RMSE:", train_rmse)

val_pred_x = m_x_final.predict(X_val)
val_pred_y = m_y_final.predict(X_val)

# Compute per-play RMSE (stack x and y)
val_rmse = np.sqrt(mean_squared_error(y_val[['x','y']], np.column_stack([val_pred_x, val_pred_y])))
print("Validation RMSE:", val_rmse)
