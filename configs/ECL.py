from configs.common_configs import args

args.model_comment = 'ECL'
args.data = 'ECL'
args.root_path = './dataset'
args.data_path = 'ECL.csv'
args.source_data_path = 'ECL.csv'

args.features = 'M'

args.num_train = 8760+24
args.num_test = 8760
args.seq_len = 96
args.pred_len = 96
args.label_len = args.seq_len
args.forecast_dim = 2
args.feature_cols = None
args.target = None
args.enc_in = 321
args.dec_in = 321
args.c_out = 321
args.scale = True