import pandas as pd
import torch
import os
from exp.exp_classification import Exp_Classification
import random
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def get_setting(args, ii):
    setting = '{}_{}_{}_ft{}_sl{}_sd{}_td{}_dm{}_df{}_nh{}_el{}_dl{}_ma{}_factor{}_dropout{}_loss{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.enc_in,
        args.c_out,
        args.d_model,
        args.d_ff,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.moving_avg,
        args.factor,
        args.dropout, args.loss,args.loss_method)

    if 'LLM' in args.model:
        setting += '_{}_llmd{}_llmf{}_tk{}'.format(args.llm_model, args.llm_dim, args.llm_layers, args.top_k)
        if args.use_prompt:
            setting += '_prompt'
    if 'RNN' in args.model:
        setting += '_{}_rnnd{}_rnnf{}'.format(args.rnn_model, args.rnn_dim, args.rnn_layers, )

    if args.percent != 100:
        setting = 'few-shot{}_'.format(args.percent) + setting
    if args.scale:
        setting += '_scale'
    return setting

def main(args):
    torch.cuda.empty_cache()
    Exp = Exp_Classification


    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            # setting record of experiments
            setting = get_setting(args, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            metrics_df,pred_res = exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = get_setting(args, ii)

        exp = Exp(args)  # set experiments
        print(' >>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        metrics_df,pred_res = exp.test(setting, test=1)
    torch.cuda.empty_cache()
    return metrics_df,pred_res


if __name__ == '__main__':
    # from configs.operational_configs import args as default_args
    # from configs.electricity_configs import args as default_args
    from configs.HVAC_configs import args as default_args
    from copy import deepcopy
    from utils.hyparameter_setup import model_hyparameter_setup

    all_results = []
    for model in ['RNN', 'DLinear', 'Transformer', 'Informer', 'Autoformer', 'iTransformer','PatchTST', 'TimesNet']:
    # for model in ['Autoformer']:

        args = deepcopy(default_args)
        fix_seed = 1234
        args.fix_seed = fix_seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        args.is_training = 0

        args.model_id = 'train80'
        args.model = model
        args.task_name = 'classification'
        args.pred_len = 0
        args.label_len = 0

        args.data_path = 'dataset_splits_train80_test20.npz'  # dataset_splits_train40_test60
        args = model_hyparameter_setup(args)
        args.seq_len = 6
        args.enc_in = 16
        # args.patience = 2
        args.train_epochs = 50

        metrics_df,_ = main(args)
        metrics_df.insert(0, 'model', model)

        all_results.append(metrics_df.iloc[-1:])

        final_metrics_df = pd.concat(all_results, axis=0, ignore_index=False)
        # final_metrics_df.to_csv('./results/{}_all_models_comparison.csv'.format(args.model_id))


