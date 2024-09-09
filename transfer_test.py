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
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_dropout{}_eb{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.seq_dim,
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
    if 'RNN' in args.model:
        setting += '_{}_llmd{}_llmf{}'.format(args.rnn_model, args.rnn_dim, args.rnn_layers, )

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1, path=path)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    path = r'D:\Time-LLM-main\results\sr_RNN_Tokyo_ftM_sl72_ll24_pl24_sd9_dm512_nh8_el8_dl1_df2048_fc3_dropout0.1_ebtimeF_test_0_LSTM_llmd256_llmf3'
    args = load_config(os.path.join(path,'checkpoints','configs.pkl'))

    for city in ['Sapporo','Naha']:
        args.data = city  # Sapporo Sendai Tokyo Osaka Fukuoka Naha
        args.data_path = '{}.csv'.format(args.data)

        transfer_test(args, path)
