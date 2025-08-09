import pandas as pd
import numpy as np


def extreme_weather():
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
    test_df_spike = summer_df.copy()

    # Group by day and apply peak-based disturbance
    for date, group in summer_df.groupby(summer_df.index.date):
        day_df = test_df_spike.loc[str(date)]
        if len(day_df) == 0:
            continue

        peak_idx = day_df[demand_col].idxmax()
        peak_loc = test_df_spike.index.get_loc(peak_idx)
        start_loc = max(0, peak_loc - 6)
        end_loc = min(len(test_df_spike), peak_loc + 7)

        n_points = end_loc - start_loc
        individual_ratios = np.random.uniform(0.2, 0.5, size=n_points)

        demand_idx = test_df_spike.columns.get_loc(demand_col)
        fossil_idx = test_df_spike.columns.get_loc(fossil_col)

        original_demand = test_df_spike.iloc[start_loc:end_loc, demand_idx].copy()
        test_df_spike.iloc[start_loc:end_loc, demand_idx] *= (1 + individual_ratios)
        test_df_spike.iloc[start_loc:end_loc, fossil_idx] += (
                test_df_spike.iloc[start_loc:end_loc, demand_idx] - original_demand
        )

    # Pattern shift: uniformly increase entire test set
    test_df_shift = test_df.copy()
    individual_ratios_2 = np.random.uniform(0.1, 0.2, size=len(test_df))
    original_demand = test_df_shift[demand_col].copy()
    test_df_shift[demand_col] *= (1 + individual_ratios_2)
    test_df_shift[fossil_col] += (test_df_shift[demand_col] - original_demand)

    # Save results
    test_df_spike.to_csv("./electricity/test_data_extreme_weather_.csv")
    test_df_shift.to_csv("./electricity/test_data_pattern_shift_.csv")


def get_daily_weather():
    # 读取CSV，假设index是Datetime，列中有 "Temperature" 和 "SolarRadiation"
    df = pd.read_csv(r'D:\Time-LLM-main\dataset\electricity\kansai.csv', parse_dates=True, index_col=0)

    # 确保 index 是 Datetime 类型
    df.index = pd.to_datetime(df.index)

    # 每日聚合
    daily_stats = pd.DataFrame()
    daily_stats['Max_Temperature'] = df['Temperature'].resample('D').max()
    daily_stats['Min_Temperature'] = df['Temperature'].resample('D').min()
    daily_stats['Max_SolarRadiation'] = df['Global_horizontal_irradiance'].resample('D').max()

    # 保存为新的CSV
    daily_stats.to_csv(r'D:\Time-LLM-main\dataset\electricity\kansai_daily_summary.csv')


if __name__ == '__main__':
    input_file = './Qingdao.csv'  # 输入文件路径
    output_file = './Qingdao_15min.csv'  # 输出文件路径
    columns_to_round = ['temp', 'dewPt']  # 指定需要保留一位小数的列名

    # === 读取并设置时间为索引 ===
    df = pd.read_csv(input_file)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')  # 第一列为时间
    df = df.dropna(subset=[df.columns[0]])  # 删除无法解析为时间的行
    df = df.set_index(df.columns[0])
    df = df.sort_index()

    # === 重新采样为15分钟 ===
    df_15min = df.resample('15T').asfreq()  # 保留空行，用于补值

    # === 数值型数据：线性插值 ===
    numeric_cols = df_15min.select_dtypes(include='number').columns
    df_15min[numeric_cols] = df_15min[numeric_cols].interpolate(method='linear')

    # === 非数值型数据：向前填充 ===
    non_numeric_cols = df_15min.select_dtypes(exclude='number').columns
    df_15min[non_numeric_cols] = df_15min[non_numeric_cols].ffill()

    # === 保留一位小数 ===
    for col in columns_to_round:
        if col in df_15min.columns:
            df_15min[col] = df_15min[col].round(1)
        else:
            print(f"⚠️ 列 {col} 不存在，跳过保留小数处理。")

    # === 输出结果 ===
    df_15min.to_csv(output_file, encoding='utf-8-sig')
    print(f"✅ 已保存插值结果到：{output_file}")

