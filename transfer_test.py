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
from run_tokyo_price import get_setting

def transfer_test(args,path):
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

    path = r'D:\Time-LLM-main\results\price\price_RNN_Tokyo_ftS_sl72_ll24_pl24_sd1_dm512_nh8_el8_dl1_df2048_fc3_dropout0.1_ebtimeF_test_0_LSTM_llmd256_llmf3_scale'
    args = load_config(os.path.join(path,'checkpoints','configs.pkl'))

    for city in ['Sapporo','Fukuoka']:
        args.data = city  # Sapporo Sendai Tokyo Osaka Fukuoka Naha
        args.data_path = '{}.csv'.format(args.data)

        transfer_test(args, path)
