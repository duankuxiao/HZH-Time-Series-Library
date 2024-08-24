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

if __name__ == '__main__':
    # from pv_configs import args
    from solar_radiation_configs import args

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.inverse = True
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        args.learning_rate = 0.01

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Forecast

    if 'TimeLLM' in args.model:
        args.content = load_content(args)
        if 'LLAMA' in args.llm_model:
            args.d_model = 16
            args.d_ff = 32
            args.llm_layers = 32
        elif 'BERT' in args.llm_model:
            args.d_model = 32
            args.d_ff = 128
            args.llm_layers = 6
        else:
            raise ValueError('Unknown llm model')

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            # setting record of experiments
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
                setting += '_{}_llmd{}_llmf{}'.format(args.llm_model, args.llm_dim, args.llm_layers, )
            if 'RNN' in args.model:
                setting += '_{}_llmd{}_llmf{}'.format(args.rnn_model, args.rnn_dim, args.rnn_layers, )

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()