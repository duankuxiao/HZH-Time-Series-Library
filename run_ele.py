import pandas as pd
import torch
import os
from exp.exp_forecasting import Exp_Forecast
import random
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 4213
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def get_setting(args,ii):
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_td{}_lr{}_dm{}_df{}_nh{}_el{}_dl{}_ma{}_factor{}_dropout{}_eb{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.enc_in,
        args.c_out,
        args.learning_rate,
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
    from utils.hyparam_setup_forele import model_hyparameter_setup
    from configs.electricity_configs import args as default_args
    from copy import deepcopy

    # for model in ['RNN', 'DLinear', 'Informer', 'Autoformer', 'iTransformer', 'TimesNet', 'TimeLLM', 'TimeLLMformer']:
    for model in ['Informer']:

        all_results = []
        # for pred_len in [1, 24, 72, 168, 336, 720]:
        # for data in ['hokkaido','tohoku','kyushu']:
        for data in ['tokyo']:

        # for model in ['Informer', 'Autoformer']:
            args = deepcopy(default_args)
            args.model_id = 'test'
            args.data_path = '{}.csv'.format(data)
            args.source_data_path = '{}.csv'.format(data)
            args.model = model  # [Autoformer, TimeLLM, TimeLLMX, TimeLLMformer, TimesNet, DLinear, Informer, Transformer, TimeMixer, iTransformer, TransformerForecast, RNN, PatchTST,]
            args.is_training = 1
            args.accelerate = False
            args.use_prompt = True

            args.feature_cols = ['Electricity','Renewable_energy', 'Nuclear', 'Coal', 'Hydro', 'Geothermal', 'Biomass','Solar', 'Solar_curtailment', 'Wind', 'Wind_ccurtailment','Water_pumping',
                                 'Interconnection', 'Temperature', 'Relative_humidity', 'Precipitation', 'Dew_point', 'Vapor_pressure', 'Wind_speed', 'Sunshine_duration',
                                  'Global_horizontal_irradiance']
            # args.feature_cols = ['Electricity', 'Renewable_energy','Coal', ]
            args.target = ['Electricity', 'Renewable_energy', 'Coal']
            # args.target = ['Renewable_energy']

            args = model_hyparameter_setup(args)


            _, res_metrics_df = main(args)
            # res_metrics_df.insert(0, 'pred_len', pred_len)
            # all_results.append(res_metrics_df)
            # final_metrics_df = pd.concat(all_results, axis=0, ignore_index=False)
            # final_metrics_df.to_csv('./results/ele_{}_all_models_comparison.csv'.format(model))


