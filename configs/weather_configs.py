from configs.common_configs import args

args.model_comment = 'weather forecast'
args.data = 'Tokyo'
args.root_path = './dataset/weather'
args.data_path = 'Tokyo_10min.csv'
args.source_data_path = 'Tokyo_10min.csv'

args.features = 'M'

args.target = ['Temperature','Global_horizontal_irradiance']
args.num_train = (8760 * 2 + 24) * 6
args.num_test = 8760 * 6
args.seq_len = 6 * 24
args.pred_len = 6 * 24 * 3
args.label_len = args.seq_len
args.forecast_dim = 2
args.features_cols = ['Year', 'Month', 'Day', 'Hour', 'Min',
       'Atmospheric_pressure_local', 'Atmospheric_pressure_sea_level', 'Temperature', 'Relative_humidity', 'Wind_speed_mean', 'Wind_speed_max','Sunshine_duration']

args.enc_in = 9
args.dec_in = 9
args.c_out = 1
args.scale = True


if __name__ == '__main__':
    import pandas as pd
    import os
    df = pd.read_csv(os.path.join('D:\Time-LLM-main\dataset\electricity', 'tokyo_2016.csv'),encoding='SHIFT-JIS',low_memory=False)
    df.index = pd.to_datetime(df.index)
    df_clean = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    df_filled = df_clean.interpolate(method='time')
    df_filled.to_csv(os.path.join('D:\Time-LLM-main\dataset\electricity', 'tokyo_2016_.csv'),encoding='SHIFT-JIS')

    # df_filled = df_filled.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    #
    # # 再找出哪些位置原始不是 NaN 但转换后是 NaN
    # mask_bad = df_filled.isna() & df.notna()
    #
    # # 将这些位置（行, 列）打印出来
    # bad_locations = [(idx, col, df.loc[idx, col])
    #                  for col in df.columns
    #                  for idx in df.index
    #                  if mask_bad.loc[idx, col]]
    # print("这些单元格无法转为 float：")
    # for loc in bad_locations:
    #     print(f"  行 {loc[0]!r}，列 {loc[1]!r}，值 = {loc[2]!r}")

