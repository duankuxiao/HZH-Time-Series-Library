import os

from configs.common_configs import args

args.model_comment = 'office building operational data'
args.data = 'aircon'
args.root_path = './dataset/aircon'
args.data_path = 'operational_data_5min.csv'
args.source_data_path = 'operational_data_5min.csv'
args.freq = 't'
args.features = 'M'

# args.num_train = 88693 - 31 * 24 * 12 * 2 - 30 * 24 * 12 * 2
# args.num_test = 30 * 24 * 12 + 31 * 24 * 12
args.num_train = 35688   # 7test  35688  11test  71124
args.num_test = 17569   # 7test 53005  11test  17569
args.val = False
args.use_norm = False
args.seq_len = 24  # dx 24
args.pred_len = 1  # dx 1
args.label_len = args.seq_len
args.forecast_dim = 1
args.feature_cols = args.feature_cols = ['Solar_east', 'Solar_south', 'Solar_west', 'Temperature_air',
       'Wind__velocity', 'RoomA_Control__setpoint_temperature_global', 'Ti_A',
       'RoomA:Damper__position', 'RoomB:Damper__position',
       'RoomA:Window__opened_closed', 'RoomA:Sensor__CO2', 'RoomB:Sensor__CO2',
       'RoomB:Window__opened_closed', 'Ti_B', 'RoomA_Inlet_Flow',
       'RoomB:Inlet_Flow']

# args.target = ['Ti_A', 'Ti_B']
args.target = ['Ti_A']


if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv(os.path.join(r'D:\Time-LLM-main\dataset\aircon', 'aircon.csv'), index_col=0, encoding='SHIFT-JIS')
    print(data.columns)
