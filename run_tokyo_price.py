import torch
import os
from exp.exp_forecasting import Exp_Forecast
from utils.print_args import print_args
from utils.tools import load_content
import random
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 4213
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def get_setting(args,ii):
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_dropout{}_eb{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.enc_in,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.dropout,
        args.embed,
        args.des, ii)

    if 'TimeLLM' in args.model:
        setting += '_{}_llmd{}_llmf{}_tk{}'.format(args.llm_model, args.llm_dim, args.llm_layers, args.top_k)
        if args.use_prompt:
            setting += '_prompt'
    if 'RNN' in args.model:
        setting += '_{}_llmd{}_llmf{}'.format(args.rnn_model, args.rnn_dim, args.rnn_layers, )

    if args.use_forecast:
        setting += '_forecast'
    if args.percent != 100:
        setting = 'few-shot{}_'.format(args.percent) + setting
    if args.scale:
        setting += '_scale'
    return setting


def main(args):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    args.inverse = True
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    if args.feature_cols is not None:
        args.enc_in = len(args.feature_cols)
        args.dec_in = len(args.feature_cols)

    if 'TimeLLM' in args.model:
        args.batch_size = 24
        args.learning_rate = 0.01
        args.content = load_content(args)
        if 'LLAMA' in args.llm_model:
            args.llm_dim = 4096
        elif 'BERT' in args.llm_model:
            args.llm_dim = 768
        else:
            raise ValueError('Unknown llm model')

    if args.feature_cols is not None:
        args.enc_in = len(args.feature_cols)
        args.dec_in = len(args.feature_cols)

    if args.features == 'S':
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            # setting record of experiments
            setting = get_setting(args,ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = get_setting(args,ii)

        exp = Exp(args)  # set experiments
        print(' >>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # from pv_configs import args
    # from solar_radiation_configs import args
    from configs.price_configs import args
    args.scale = True
    args.data = 'Tokyo'
    args.target = 'Price'
    args.data_path = 'Tokyo.csv'
    args.seq_len = 24 * 3
    args.pred_len = 24 * 3
    args.label_len = 24

    args.model_id = 'price'

    args.model = 'RNN'  # [Autoformer, TimeLLM, TimeLLMX, TimeLLMformer, TransformerTimeLLM, RNNTimeLLM, TimesNet, DLinear, Informer, Transformer, TimeMixer, iTransformer, TransformerForecast, RNN, PatchTST,TimeLLMDLinear, DLinearTimeLLM]
    args.is_training = 1
    args.features = 'MS'
    args.use_prompt = False
    args.use_forecast = False  # False
    args.forecast_dim = 4

    if 'TimeLLM' in args.model:
        args.d_model = 64
        args.d_ff = 128
        args.e_layers = 4
        args.d_layers = 1
        args.llm_layers = 6

    if args.model == 'Transformer':
        args.d_model = 64
        args.d_ff = 256
        args.e_layers = 4
        args.d_layers = 1

    if 'RNN' in args.model:
        args.rnn_dim = 256
        args.rnn_layers = 3
    args.feature_cols = ['System_price', 'Sell_volume', 'Buy_volume', 'Total_volume',
                         'Sell_volume_block_orders', 'Sell_volume_contracted_block_orders', 'Buy_volume_block_orders', 'Buy_volume_contracted_block_orders',  'Price']

    # ['date', 'Temperature', 'Relative_humidity', 'Precipitation',
    #  'Dew_point', 'Vapor_pressure', 'Wind_speed', 'Sunshine_duration',
    #  'Snowfall', 'Global_horizontal_irradiance', 'System_price',
    #  'Sell_volume', 'Buy_volume', 'Total_volume', 'Sell_volume_block_orders',
    #  'Sell_volume_contracted_block_orders', 'Buy_volume_block_orders',
    #  'Buy_volume_contracted_block_orders', 'Volumes', 'Number_of_contracts',
    #  'Opening_price', 'Highest_price', 'Lowest_price', 'Last_price',
    #  'Average_price', 'DA-24', 'DA-DT', 'DA-PT', 'TTV', 'Price']


    main(args)


