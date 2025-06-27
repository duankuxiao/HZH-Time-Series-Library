import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("./electricity/tokyo.csv", parse_dates=True, index_col=0)

# Check for necessary columns
demand_col = "Electricity"
fossil_col = "Coal"

# Resample daily to find peak demand day in the last year (test set)
n = len(df)
train_end = 8760*2 + 24
val_end = 8760*2+24 + 8760
test_df = df.iloc[val_end:].copy()

## Create copy for the extreme weather disturbance
test_df_spike = test_df.copy()

# Define summer window: July 1 to September 30
summer_df = test_df_spike[(test_df_spike.index.month >= 7) & (test_df_spike.index.month <= 9)]

# Group by day and apply peak-based disturbance
for date, group in summer_df.groupby(summer_df.index.date):
    day_df = test_df_spike.loc[str(date)]
    if len(day_df) == 0:
        continue

    peak_idx = day_df[demand_col].idxmax()
    peak_loc = test_df_spike.index.get_loc(peak_idx)
    start_loc = max(0, peak_loc - 6)
    end_loc = min(len(test_df_spike), peak_loc + 7)

    increase_ratio = np.random.uniform(0.2, 0.5)
    test_df_spike.iloc[start_loc:end_loc, test_df_spike.columns.get_loc(demand_col)] *= (1 + increase_ratio)
    test_df_spike.iloc[start_loc:end_loc, test_df_spike.columns.get_loc(fossil_col)] += (
        test_df_spike.iloc[start_loc:end_loc][demand_col] - test_df.iloc[start_loc:end_loc][demand_col]
    )

# Pattern shift: uniformly increase entire test set
increase_ratio_2 = np.random.uniform(0.1, 0.2)
test_df_shift = test_df.copy()
test_df_shift[demand_col] *= (1 + increase_ratio_2)
test_df_shift[fossil_col] += (test_df_shift[demand_col] - test_df[demand_col])

# Save results
test_df_spike.to_csv("./electricity/test_data_extreme_weather.csv")
test_df_shift.to_csv("./electricity/test_data_pattern_shift.csv")