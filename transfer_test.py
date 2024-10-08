from utils.tools import load_config,save_config
import os
import torch
import os
from exp.exp_forecasting import Exp_Forecast
from utils.print_args import print_args
from utils.tools import load_content
import random
import numpy as np
from pred_results import res_evaluation
from run_main import get_setting


def transfer_test(args, path):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    args.inverse = True
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Forecast

    ii = 0
    setting = get_setting(args,ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1, path=path)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    path = r'D:\Time-LLM-main\results\res\Electricity_Transformer_electricity_ftM_sl72_ll24_pl168_sd22_td3_dm128_nh8_el6_dl2_df512_fc3_dropout0.1_ebtimeF_test_0_scale'
    args = load_config(os.path.join(path,'checkpoints','configs.pkl'))

    # for city in ['tokyo','hokkaido','tohoku']:
    for city in ['hokkaido']:

        # args.target = ['Coal']  # ['Electricity', 'Renewable_energy', 'Coal']  _Electricity  _Renewable_energy _Coal
        args.target = ['Electricity', 'Renewable_energy', 'Coal']
        args.data_path = '{}.csv'.format(city)
        args.source_data_path = args.data_path
        args.features = 'M'
        transfer_test(args, path)
