from configs.common_configs import args

args.model_comment = 'jepx_index'
args.data = 'Index'
args.root_path = './dataset/price'
args.data_path = 'spot_index.csv'
args.source_data_path = 'spot_index.csv'

args.features = 'M'
args.target = ['DA-24','DA-DT','DA-PT','TTV']
args.freq = 'd'
args.num_train = 8760*2+24
args.num_test = 8760
args.seq_len = 15
args.pred_len = 30
args.label_len = 15
args.forecast_dim = 0
args.features_cols = None
args.moving_avg = 31
args.enc_in = 4
args.dec_in = 4
args.c_out = 4
args.scale = True

# ['DA-24','DA-DT', 'DA-PT','TTV']

