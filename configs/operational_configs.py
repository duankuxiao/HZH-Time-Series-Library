import os

from configs.common_configs import args

args.model_comment = 'office building operational data'
args.data = 'aircon'
args.root_path = './dataset/aircon'
args.data_path = 'operational_data_5min.csv'
args.source_data_path = 'operational_data_5min.csv'
args.freq = 't'
args.features = 'M'

args.num_train = 88693 - 31 * 24 * 12 * 2 - 30 * 24 * 12  # 71113
args.num_test = 30 * 24 * 12 + 31 * 24 * 12  # 1465  # 4320   val 5622

args.seq_len = 36
args.pred_len = 12
args.label_len = args.seq_len
args.forecast_dim = 1
args.feature_cols = ['RoomA_Control__setpoint_temperature_global', 'RoomA_Inlet_Flow', 'Solar_east', 'Solar_south', 'Solar_west', 'Temperature_air',
                     'Wind__velocity', 'RoomA_Control__setpoint_temperature_global', 'RoomA:Damper__position', 'RoomB:Damper__position', 'RoomA:Window__opened_closed',
                     'RoomA:Sensor__CO2', 'RoomB:Sensor__CO2', 'RoomB:Window__opened_closed', 'Ti_B']

args.target = ['Ti_A', 'Ti_B']

if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv(os.path.join(r'D:\Time-LLM-main\dataset\aircon', 'aircon.csv'), index_col=0, encoding='SHIFT-JIS')
    print(data.columns)
