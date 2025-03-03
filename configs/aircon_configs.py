import os

from configs.common_configs import args

args.model_comment = 'air conditioning measured data'
args.data = 'aircon'
args.root_path = './dataset/aircon'
args.data_path = 'aircon.csv'
args.source_data_path = 'aircon.csv'

args.features = 'MS'
args.target = ['Global_horizontal_irradiance']

args.num_train = 30*24*6*2
args.num_test = 30*24*6

args.seq_len = 3 * 6
args.pred_len = 1
args.label_len = args.seq_len
args.forecast_dim = 6
args.features_cols = ['ac3_fanspeed', 'ac4_fanspeed','ac3_temp', 'ac4_temp','Temperature', 'Relative_humidity','ac3_rh_ra', 'ac4_rh_ra',
       'ac3_rh_sa', 'ac4_rh_sa', '2F', '1F_room1', '1F_room2', 'ac3_ra', 'ac4_ra', 'ac3_sa', 'ac4_sa',  'ac3_power', 'ac4_power', 'Sunshine_duration']

# ['ac3_fanspeed', 'ac4_fanspeed','ac3_temp', 'ac4_temp','Temperature', 'Relative_humidity','ac3_rh_ra', 'ac4_rh_ra',
#        'ac3_rh_sa', 'ac4_rh_sa', '2F', '1F_room1', '1F_room2', 'ac3_ra', 'ac4_ra', 'ac3_sa', 'ac4_sa',  'ac3_power', 'ac4_power','PV',  'Wind_speed_mean','Wind_speed_max',  'Sunshine_duration']

# 'Year', 'Month', 'Day', 'Hour', 'Min', 'Atmospheric_pressure_local',
#        'Atmospheric_pressure_sea_level', 'Precipitation', 'Temperature',
#        'Relative_humidity', 'Wind_speed_mean', 'Wind_direction_mean',
#        'Wind_speed_max', 'Wind_direction_max', 'Sunshine_duration'

args.target = ['2F', '1F_room1', '1F_room2','ac3_power', 'ac4_power',]

args.enc_in = int(len(args.features_cols))
args.dec_in = int(len(args.features_cols))
args.c_out = int(len(args.target))
args.scale = True


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv(os.path.join(r'D:\Time-LLM-main\dataset\aircon', 'aircon.csv'),index_col=0,encoding='SHIFT-JIS')
    print(data.columns)