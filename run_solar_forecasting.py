import pandas as pd
import torch
import os
import random
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 4213
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def get_setting(args,ii):
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_td{}_dm{}_df{}_nh{}_el{}_dl{}_ma{}_factor{}_dropout{}_eb{}_{}'.format(
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
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.moving_avg,
        args.factor,
        args.dropout,
        args.embed, ii)

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
    # from configs.solar_radiation_configs import args as default_args
    from configs.operational_configs import args as op_configs
    from copy import deepcopy
    from utils.hyparameter_setup import model_hyparameter_setup

    all_results = []
    # for model in ['RNN', 'Transformer','DLinear','Informer','Autoformer', 'iTransformer', 'TimesNet','PatchTST','TimeLLM', 'TimeLLMformer']:
    for model in ['RNN', 'Transformer', 'DLinear', 'Informer', 'Autoformer', 'iTransformer', 'TimesNet', 'PatchTST']:
    # for model in ['TimesNet']:

        args = deepcopy(op_configs)

        args.is_training = 1

        args.model_id = 'dx_opdata11'
        args.model = model  # [Autoformer, TimeLLM, TimeLLMX, TimeLLMformer, TimesNet, DLinear, Informer, Transformer, TimeMixer, iTransformer, TransformerForecast, RNN, PatchTST,]
        args = model_hyparameter_setup(args)
        args.num_train = 71124   # 35688   71124
        args.num_test = 17569   # 53005   17569
        args.val = False
        args.use_norm = False
        args.patience = 6
        args.learning_rate = 0.0001

        _, res_metrics_df = main(args)
        res_metrics_df.insert(0, 'model', model)
        all_results.append(res_metrics_df)
        final_metrics_df = pd.concat(all_results, axis=0, ignore_index=False)
        final_metrics_df.to_csv('./results/{}_all_models_comparison.csv'.format(args.model_id))



