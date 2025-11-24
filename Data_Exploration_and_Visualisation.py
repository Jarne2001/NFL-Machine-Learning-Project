"""
NFL Data Exploration and Visualisation
"""
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data exploration
training_df = pd.read_csv("training_data.csv")
training_df.head
training_df.shape
training_df.info()
training_df.columns
# Data visualisation
plt.rcParams["figure.figsize"] = (10, 7)

# Field heatmap of player locations (density) — per position
# Purpose: see positional spatial patterns (where positions usually are on the field)
position = 'WR'  # change to any player_position
position_df = training_df[training_df['player_position'] == position]

plt.figure(figsize=(10,6))
sns.kdeplot(x=position_df['x'], y=position_df['y'], fill=True, cmap="magma", levels=10)
plt.title(f"Location density for player position {position}")
plt.xlabel('x'); plt.ylabel('y')
plt.show()

# smaller sample
position_df = training_df.query("player_position == 'WR'").sample(50_000, random_state=42)

sns.kdeplot(x=position_df['x'], y=position_df['y'], fill=True, cmap="magma", levels=10)
plt.title("WR Location Density (50K samples)")
plt.show()

position = 'FS'
position_df = training_df.query(f"player_position == {position}").sample(50_000, random_state=42)

sns.kdeplot(x=position_df['x'], y=position_df['y'], fill=True, cmap="magma", levels=10)
plt.title(f"{position} Location Density (50K samples)")
plt.show()

position = 'SS'
position_df = training_df.query(f"player_position == {position}").sample(50_000, random_state=42)

sns.kdeplot(x=position_df['x'], y=position_df['y'], fill=True, cmap="magma", levels=10)
plt.title(f"{position} Location Density (50K samples)")
plt.show()

position = 'CB'
position_df = training_df.query(f"player_position == {position}").sample(50_000, random_state=42)

sns.kdeplot(x=position_df['x'], y=position_df['y'], fill=True, cmap="magma", levels=10)
plt.title(f"{position} Location Density (50K samples)")
plt.show()

position = 'MLB'
position_df = training_df.query(f"player_position == {position}").sample(50_000, random_state=42)

sns.kdeplot(x=position_df['x'], y=position_df['y'], fill=True, cmap="magma", levels=10)
plt.title(f"{position} Location Density (50K samples)")
plt.show()

position = 'TE'
position_df = training_df.query(f"player_position == {position}").sample(50_000, random_state=42)

sns.kdeplot(x=position_df['x'], y=position_df['y'], fill=True, cmap="magma", levels=10)
plt.title(f"{position} Location Density (50K samples)")
plt.show()

position = 'QB'
position_df = training_df.query(f"player_position == {position}").sample(50_000, random_state=42)

sns.kdeplot(x=position_df['x'], y=position_df['y'], fill=True, cmap="magma", levels=10)
plt.title(f"{position} Location Density (50K samples)")
plt.show()

position = 'OLB'
position_df = training_df.query(f"player_position == {position}").sample(50_000, random_state=42)

sns.kdeplot(x=position_df['x'], y=position_df['y'], fill=True, cmap="magma", levels=10)
plt.title(f"{position} Location Density (50K samples)")
plt.show()

# distributions, outliers (scaling needed?)
plt.figure()
sns.histplot(training_df['s'].dropna(), bins=50, kde=True)
plt.title("Speed distribution (s)")
plt.show()

plt.figure()
sns.histplot(training_df['player_height'].dropna(), bins=50, kde=False)
plt.title("Player height distribution (s)")
plt.show()

plt.figure()
sns.histplot(training_df['player_weight'].dropna(), bins=50, kde=False)
plt.title("Player weight distribution (s)")
plt.show()

plt.figure()
sns.histplot(training_df['a'].dropna(), bins=50, kde=True)
plt.title("a distribution (a)")
plt.show()

plt.figure()
sns.histplot(training_df['dir'].dropna(), bins=50, kde=True)
plt.title("dir distribution")
plt.show()

plt.figure()
sns.boxplot(x='player_position', y='s', data=training_df.sample(20000, random_state=1))  # sample for speed
plt.title("Speed by position (sampled)")
plt.xticks(rotation=45)
plt.show()

# linear correlations and collinearity
num_cols = ['x','y','s','a','dir','o','player_weight', 'player_height', 'absolute_yardline_number', 'ball_land_x' ,'ball_land_y']  # pick numeric cols you care about
corr = training_df[num_cols].corr(method='spearman').round(2)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Spearman correlation (selected numeric features)")
plt.show() # no strong correlation
# per-play aggregates and inspect distributions
agg = training_df.groupby(['game_id','play_id']).agg({
    's': ['mean','max'],
    'a': ['mean','max'],
    'x': ['min','max']
})
agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
agg = agg.reset_index()
plt.figure()
sns.histplot(agg['s_mean'], bins=50, kde=True)
plt.title("Distribution of mean speed per play")
plt.show()
