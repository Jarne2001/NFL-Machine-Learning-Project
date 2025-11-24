"""
NFL Data Feature Retention
"""

import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

release_df = pd.read_csv("release_df.csv")

# Which features are useful?
correlation = release_df[['x','y','vx','vy','dir_sin','dir_cos','o_sin','o_cos',
                       'distance_to_ball','eta_to_ball','heading_alignment',
                       'projection_distance_to_ball',
                       'min_defender_to_receiver_dist','relative_speed_nearest_defender',
                       'player_avg_speed','player_avg_acceleration']].corr()

plt.figure(figsize=(14,11))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='magma')
plt.title("Feature Correlations")
plt.show()

# Basic model testing
