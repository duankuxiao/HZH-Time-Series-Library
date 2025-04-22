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
    df = pd.read_csv(os.path.join('../dataset/weather', 'Tokyo_10min.csv'),encoding='SHIFT-JIS',low_memory=False)
    df['Sunshine_duration'] = df['Sunshine_duration'].fillna(0)
    df.to_csv(os.path.join('../dataset/weather', 'Tokyo_10min.csv'),encoding='SHIFT-JIS')
