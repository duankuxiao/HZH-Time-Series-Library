import pandas as pd
import torch
import os
from exp.exp_imputation import Exp_Imputation
from exp.exp_imputation_forecast import Exp_Imputation_Forecast
import random
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def get_setting(args, ii):
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_sd{}_td{}_dm{}_df{}_nh{}_el{}_dl{}_factor{}_loss{}_{}_{}{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.enc_in,
        args.c_out,
        args.d_model,
        args.d_ff,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.factor, args.loss,args.loss_method,args.mask_method,args.mask_rate)

    if 'LLM' in args.model:
        setting += '_{}_llmd{}_llmf{}_tk{}'.format(args.llm_model, args.llm_dim, args.llm_layers, args.top_k)
        if args.use_prompt:
            setting += '_prompt'

    if args.use_forecast:
        setting += '_forecast'
    if args.percent != 100:
        setting = 'few-shot{}_'.format(args.percent) + setting
    if args.scale:
        setting += '_scale'
    if args.mask_target_only:
        setting += '_target'
    if args.output_ori:
        setting += '_ori'
    if args.input_inter:
        setting += '_int'
    return setting

def main(args):
    torch.cuda.empty_cache()
    if args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'imputation_forecast':
        Exp = Exp_Imputation_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            # setting record of experiments
            setting = get_setting(args, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            res_df, metrics_df, imputation_metrics_df  = exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = get_setting(args, ii)

        exp = Exp(args)  # set experiments
        print(' >>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        res_df, metrics_df, imputation_metrics_df = exp.test(setting, test=1)
    torch.cuda.empty_cache()
    return res_df, metrics_df, imputation_metrics_df


def xgboost(args):
    from utils.setup_imputation import model_hyparameter_setup
    from exp.exp_imputation import Exp_Imputation

    args = model_hyparameter_setup(args)

    exp = Exp_Imputation(args)
    df = exp.xgboost_imputation()
    return df


if __name__ == '__main__':
    # from configs.operational_configs import args as default_args
    # from configs.electricity_configs import args as default_args
    from configs.HVAC_configs import args as default_args
    from copy import deepcopy
    from utils.setup_imputation import model_hyparameter_setup

    all_results_imputation = []
    all_results = []
    for model in ['RNN','DLinear', 'Transformer', 'Informer', 'iTransformer', 'PatchTST', 'TimesNet', 'LLMformer']:  # AutoTimes
    # for model in ['LSTM','DLinear', 'Transformer', 'Informer', 'iTransformer', 'PatchTST', 'TimesNet', 'LLMformer']:
        for mask_rate in [0.3]:
            # for seq_len in [288,480,672]:
        # for mask_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
            # all_results_imputation = []
            # all_results = []
            # for seed in [4, 42, 421, 4212, 94213, 974213, 9974213, 19974213, 199714213,1997715213]:
            # for seed in [6, 46, 426, 42126, 942136, 9742136, 99742136, 199742136, 1997142136]:
            # for seed in [1]:
                # for model in ['RNN', 'DLinear', 'Transformer', 'Informer', 'Autoformer', 'iTransformer', 'PatchTST', 'TimesNet', 'LLMformer']:
                # for model in ['RNN']:

            args = deepcopy(default_args)
            args.mask_method = 'mcar'
            if args.mask_method == 'rdo':
                if mask_rate == 0.1:
                    args.fix_seed = 19974213
                elif mask_rate == 0.2 or mask_rate == 0.4 or mask_rate == 0.5:
                    args.fix_seed = 42
                elif mask_rate == 0.3:
                    args.fix_seed = 421
                elif mask_rate == 0.6:
                    args.fix_seed = 199714213
                elif mask_rate == 0.7:
                    args.fix_seed = 9974213
                elif mask_rate == 0.8:
                    args.fix_seed = 974213
                elif mask_rate == 0.9:
                    args.fix_seed = 421
            if args.mask_method == 'mcar':
                if mask_rate == 0.1:
                    args.fix_seed = 199714213
                elif mask_rate == 0.2:
                    args.fix_seed = 974213
                elif mask_rate == 0.3 or mask_rate == 0.5:
                    args.fix_seed = 19974213
                elif mask_rate == 0.4:
                    args.fix_seed = 421
                elif mask_rate == 0.6:
                    args.fix_seed = 9974213
                elif mask_rate == 0.7:
                    args.fix_seed = 199714213
                elif mask_rate == 0.8:
                    args.fix_seed = 974213
                elif mask_rate == 0.9:
                    args.fix_seed = 1997715213
            if args.mask_method == 'mar':
                if mask_rate == 0.1:
                    args.fix_seed = 19974213
                elif mask_rate == 0.2:
                    args.fix_seed = 9974213
                elif mask_rate == 0.3 or mask_rate == 0.4 or mask_rate == 0.5:
                    args.fix_seed = 421
                elif mask_rate == 0.6 or mask_rate == 0.9:
                    args.fix_seed = 421
                elif mask_rate == 0.7:
                    args.fix_seed = 974213
                elif mask_rate == 0.8:
                    args.fix_seed = 9974213

            # args.fix_seed = seed
            random.seed(args.fix_seed)
            torch.manual_seed(args.fix_seed)
            np.random.seed(args.fix_seed)
            args.mask_rate = mask_rate

            args.model_id = 'withoutORI'
            args.model = model
            # 1 MSE True fix  2 MAE True fix  3 MAE False fix  4 MAE False  adaptive  5 MAE False ori
            args.loss = 'MAE'
            args.output_ori = False
            args.loss_method = "adaptive"  # missing  fix  adaptive  ori

            args.task_name = 'imputation'
            args.mask_target_only = False

            args.input_inter = True
            args = model_hyparameter_setup(args)
            args.is_training = 1
            # args.seq_len = seq_len
            # args.label_len = seq_len

            # args.patience = 1
            # args.train_epochs = 1
            # res_df, metrics_df, imputation_metrics_df = xgboost(args)
            res_df, metrics_df, imputation_metrics_df = main(args)

            imputation_metrics_df.insert(0, 'model', model)
            imputation_metrics_df.insert(1, 'mask_rate', mask_rate)
            # imputation_metrics_df.insert(2, 'seq_len', seq_len)

            metrics_df.insert(0, 'model', model)
            metrics_df.insert(1, 'mask_rate', mask_rate)
            # metrics_df.insert(2, 'seed', seed)

            all_results_imputation.append(imputation_metrics_df.iloc[-1:])
            all_results.append(metrics_df.iloc[-1:])

            final_metrics_df_imputation = pd.concat(all_results_imputation, axis=0, ignore_index=False)
            final_metrics_df_imputation.to_csv('./results/{}_{}_benchmark_imputation_metrics.csv'.format(args.model_id,args.mask_method))

            final_metrics_df = pd.concat(all_results, axis=0, ignore_index=False)
            final_metrics_df.to_csv('./results/{}_{}_benchmark_metrics.csv'.format(args.model_id,args.mask_method))

            # avg_row = final_metrics_df_imputation.mean(numeric_only=True)
            # avg_row.name = "average"  # 新行的索引，可以是字符串、时间戳等
            # # 将这一行追加到原 DataFrame 的末尾
            # df_with_avg = pd.concat([final_metrics_df_imputation, pd.DataFrame([avg_row])])
            # df_with_avg.to_csv('./results/{}_all_models_comparison_mr{}.csv'.format(args.model_id,mask_rate))


