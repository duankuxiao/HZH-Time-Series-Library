from common_configs import args

args.model_comment = 'price'
args.data = 'Price'
args.root_path = './dataset/price'
args.data_path = 'Tokyo.csv'
args.features = 'MS'
args.target = 'Price'
args.num_train = 8760*2+24
args.num_test = 8760
args.seq_len = 24 * 3
args.pred_len = 24 * 3
args.label_len = 24
args.forecast_dim = 1
args.features_cols = None
args.enc_in = 29
args.dec_in = 29
args.c_out = 1


