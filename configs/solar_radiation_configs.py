from configs.common_configs import args

args.model_comment = 'solar radiation forecast'
args.data = 'Tokyo'
args.root_path = './dataset/solar_radiation'
args.data_path = 'Tokyo.csv'
args.source_data_path = 'Tokyo.csv'

args.features = 'M'
args.target = ['Temperature','Global_horizontal_irradiance']
args.num_train = 8760+24
args.num_test = 8760
args.seq_len = 24 * 3
args.pred_len = 24 * 3
args.label_len = args.seq_len
args.forecast_dim = 2
args.features_cols = None
# ['Temperature','Relative_humidity','Precipitation','Dew_point','Vapor_pressure','Wind_speed','Sunshine_duration','Snowfall','Global_horizontal_irradiance']
args.enc_in = 9
args.dec_in = 9
args.c_out = 1
args.scale = True
