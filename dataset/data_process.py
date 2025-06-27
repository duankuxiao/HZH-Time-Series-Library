import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("/electricity/tokyo.csv", parse_dates=True, index_col=0)

# Check for necessary columns
required_columns = ["Electricity", "Coal"]
df.columns = [col.strip().lower() for col in df.columns]  # Normalize column names
demand_col = [col for col in df.columns if "Electricity" in col][0]
fossil_col = [col for col in df.columns if "Coal" in col][0]

# Resample daily to find peak demand day in the last year (test set)
n = len(df)
train_end = 8760*2+24
val_end = 8760*2+24 + 8760
test_df = df.iloc[val_end:].copy()

# Find the day with the highest total demand
daily_demand = test_df[demand_col].resample('D').sum()
peak_day = daily_demand.idxmax()

# Get the full day data (24 hours) for that peak day
peak_day_data = test_df.loc[peak_day.strftime('%Y-%m-%d')]
increase_ratio_1 = np.random.uniform(0.2, 0.5)

# Apply the increase
test_df_spike = test_df.copy()
test_df_spike.loc[peak_day_data.index, demand_col] *= (1 + increase_ratio_1)
test_df_spike.loc[peak_day_data.index, fossil_col] += (
    test_df_spike.loc[peak_day_data.index, demand_col] - test_df.loc[peak_day_data.index, demand_col]
)

# Pattern shift: uniformly increase entire test set
increase_ratio_2 = np.random.uniform(0.1, 0.2)
test_df_shift = test_df.copy()
test_df_shift[demand_col] *= (1 + increase_ratio_2)
test_df_shift[fossil_col] += (test_df_shift[demand_col] - test_df[demand_col])

# Save results
test_df_spike.to_csv("/mnt/data/test_data_extreme_weather.csv")
test_df_shift.to_csv("/mnt/data/test_data_pattern_shift.csv")