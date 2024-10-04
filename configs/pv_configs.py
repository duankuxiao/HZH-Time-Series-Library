from common_configs import args

args.model_comment = 'PV forecast'
args.data = 'PV'
args.root_path = './dataset/PV'
args.data_path = 'PV_hour.csv'
args.features = 'MS'
args.target = ['DA-24','DA-DT','DA-PT','TTV']
args.freq = 'd'
args.num_train = 8760*2+24
args.num_test = 8760
args.seq_len = 72
args.pred_len = 72
args.label_len = 24
args.forecast_dim = 2
args.features_cols = None
# ['Temperature','Relative_humidity','Sun','Dew_point','Global_horizontal_irradiance','Sunshine_duration','PV']
args.enc_in = 11
args.dec_in = 11
args.c_out = 1