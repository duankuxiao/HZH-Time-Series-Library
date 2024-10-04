from common_configs import args

args.model_comment = 'electricity'
args.data = 'electricity'
args.root_path = './dataset/electricity'
args.data_path = 'kyushu.csv'
args.features = 'MS'
args.target = 'Electricity'
args.num_train = 8760*2+24
args.num_test = 8760
args.seq_len = 24 * 3
args.pred_len = 24 * 7
args.label_len = 24
args.forecast_dim = 1
args.features_cols = None
args.enc_in = 22
args.dec_in = 22
args.c_out = 1
# ['Electricity', 'Nuclear', 'Coal', 'Hydro', 'Geothermal', 'Biomass',
#        'Solar', 'Solar_curtailment', 'Wind', 'Wind_ccurtailment',
#        'Water_pumping', 'Interconnection', 'Total', 'Temperature',
#        'Relative_humidity', 'Precipitation', 'Dew_point', 'Vapor_pressure',
#        'Wind_speed', 'Sunshine_duration', 'Snowfall',
#        'Global_horizontal_irradiance']



