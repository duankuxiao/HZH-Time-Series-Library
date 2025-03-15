import pandas as pd
import torch
import os
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
    setting = 'if_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_td{}_dm{}_df{}_el{}_dl{}_nh{}_ma{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.enc_in,
        args.c_out,
        args.d_model,
        args.d_ff,
        args.e_layers,
        args.d_layers,
        args.n_heads,
        args.moving_avg,
        args.des, args.likelihood)

    if 'TimeLLM' in args.model:
        setting += '_{}_llmd{}_llmf{}_tk{}'.format(args.llm_model, args.llm_dim, args.llm_layers, args.top_k)
        if args.use_prompt:
            setting += '_prompt'
    if 'RNN' in args.model:
        setting += '_{}_rnnd{}_rnnf{}'.format(args.rnn_model, args.rnn_dim, args.rnn_layers, )

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

    args.content = load_content(args)

    if 'TimeLLM' in args.model:
        args.batch_size = 24
        args.learning_rate = 0.01
        args.content = load_content(args)
        if args.llm_model == 'LLAMA8b':
            args.llm_dim = 4096
        elif args.llm_model == 'LLAMA3b':
            args.llm_dim = 3072
        elif args.llm_model == 'LLAMA1b':
            args.llm_dim = 2048
        elif 'BERT' in args.llm_model:
            args.llm_dim = 768
        elif 'GPT2' in args.llm_model:
            args.llm_dim = 768
        else:
            raise ValueError('Unknown llm model')

    if args.feature_cols is not None:
        args.enc_in = len(args.feature_cols)
        args.dec_in = len(args.feature_cols)
    args.c_out = len(args.target)

    if args.features == 'S':
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1

    print('Args in experiment:')
    print_args(args)
    if args.task_name == 'interval_forecast':
        from exp.exp_interval_forecasting import Exp_Forecast
    else:
        from exp.exp_forecasting import Exp_Forecast

    Exp = Exp_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            # setting record of experiments
            setting = get_setting(args,ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            res_df, res_metrics_df = exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = get_setting(args,ii)

        exp = Exp(args)  # set experiments
        print(' >>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        res_df, res_metrics_df = exp.test(setting, test=1)
        torch.cuda.empty_cache()
    return res_df, res_metrics_df



if __name__ == '__main__':
    # from pv_configs import args
    from configs.solar_radiation_configs import args
    # from configs.price_configs import args
    # from configs.electricity_configs import args
    args.target = ['Temperature', 'Global_horizontal_irradiance']
    # args.target = ['Global_horizontal_irradiance']

    args.task_name = 'interval_forecast'
    args.likelihood = 'g'
    args.seq_len = 168
    args.pred_len = 24
    args.label_len = args.seq_len
    args.is_training = 1
    args.accelerate = False
    args.use_prompt = True
    all_results = []

    # for model in ['RNN', 'Transformer','DLinear','Informer','Autoformer', 'iTransformer', 'TimesNet','PatchTST','TimeLLM', 'TimeLLMformer']:
    # for model in ['RNN', 'Transformer', 'DLinear', 'Informer', 'Autoformer', 'iTransformer', 'TimesNet', 'PatchTST', 'TimeLLMformer']:
    for model in ['TimeLLMformer']:
        args.lradj = 'PEMS'

        args.model_id = '1'

        args.model = model  # [Autoformer, TimeLLM, TimeLLMX, TimeLLMformer, TimesNet, DLinear, Informer, Transformer, TimeMixer, iTransformer, TransformerForecast, RNN, PatchTST,]
        args.llm_model = 'LLAMA1b'
        args.d_model = 32
        args.d_ff = 64
        args.e_layers = 4
        args.d_layers = 4
        args.llm_layers = 32

        if args.model == 'TimeLLMformer':
            args.lradj = 'PEMS'
            args.d_model = 32
            args.d_ff = 64
            args.e_layers = 4
            args.d_layers = 4
            args.llm_layers = 6

        if args.model == 'Transformer':
            args.d_model = 512
            args.d_ff = 2048
            args.e_layers = 8
            args.d_layers = 3

        if 'RNN' in args.model:
            args.rnn_dim = 512
            args.rnn_layers = 2

        if args.model == 'DLinear':
            args.moving_avg = 25

        if args.model == 'iTransformer':
            args.d_model = 512
            args.d_ff = 2048
            args.e_layers = 4
            args.d_layers = 1

        if args.model == 'TimesNet':
            args.d_model = 64
            args.d_ff = 256
            args.e_layers = 2
            args.d_layers = 1

        if args.model == 'TimeLLM':
            args.feature_cols = args.target
            args.d_model = 16
            args.d_ff = 32
            args.e_layers = 1
            args.d_layers = 1
            args.llm_layers = 6

        _, res_metrics_df = main(args)
        res_metrics_df.insert(0, 'model', model)
        all_results.append(res_metrics_df)
        final_metrics_df = pd.concat(all_results, axis=0, ignore_index=False)
        final_metrics_df.to_csv('./results/all_models_comparison.csv')


