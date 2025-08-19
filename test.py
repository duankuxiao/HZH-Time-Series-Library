from utils.tools import load_config,save_config
import os
import torch
import os
from utils.print_args import print_args
import pandas as pd


def get_setting(args, ii):
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_td{}_dm{}_df{}_nh{}_el{}_dl{}_ma{}_factor{}_dropout{}_loss{}_{}_mr{}'.format(
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
        args.dropout, args.loss,args.loss_method,args.mask_rate)

    if 'LLM' in args.model:
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


def transfer_test(args, path):
    if args.task_name == 'interval_forecast':
        from exp.exp_interval_forecasting import Exp_Forecast
    else:
        from exp.exp_forecasting import Exp_Forecast

    ii = 0
    setting = get_setting(args,ii)

    exp = Exp_Forecast(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    pred_res,metrics_df = exp.test(setting, test_only=1, path=path)
    torch.cuda.empty_cache()
    return pred_res,metrics_df


def get_direct_subfolders(root_folder):
    """
    获取指定文件夹下的直接子文件夹（不包括嵌套子文件夹和文件）。

    Args:
        root_folder (str): 要遍历的根文件夹路径。

    Returns:
        list: 直接子文件夹的完整路径列表。
    """
    subfolders = [
        os.path.join(root_folder, item)
        for item in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, item))
    ]
    return subfolders


def regions_test(root_path):
    subfolders = get_direct_subfolders(root_path)
    all_results = []

    for path in subfolders:
        renamed_dfs = []
        for city in ['tokyo','hokkaido','tohoku','kyushu','kansai']:
        # for city in ['tokyo']:
            args = load_config(os.path.join(path, 'checkpoints', 'configs.pkl'))
            args.data_path = '{}.csv'.format(city)
            args.source_data_path = args.data_path
            print(args)
            args.loss_method = 'adaptive'
            pred_res, metrics_df = transfer_test(args, path)

            metrics_df.index = metrics_df.index + '_' + city
            renamed_dfs.append(metrics_df)
            combined_df = pd.concat(renamed_dfs, axis=0)
            combined_df.to_csv(os.path.join(path, 'zero-shot_res_metrics.csv'))

            metrics_df.insert(0, 'model', args.model)
            # all_results.append(metrics_df.iloc[-1:])
            all_results.append(metrics_df)
            final_metrics_df = pd.concat(all_results, axis=0, ignore_index=False)
            final_metrics_df.to_csv(os.path.join(root_path, 'pf_{}_all_models_comparison.csv'.format(args.model_id)))


def extreme_weather_test(root_path, pattern='shift'):
    assert pattern in ['shift','extreme','extreme_clean']
    subfolders = get_direct_subfolders(root_path)
    all_results = []

    for path in subfolders:
        args = load_config(os.path.join(path, 'checkpoints', 'configs.pkl'))
        if pattern == 'shift':
            args.data_path = 'test_data_pattern_shift.csv'  # test_data_extreme_weather  test_data_extreme_weather_clean  test_data_pattern_shift
        elif pattern == 'extreme':
            args.data_path = 'test_data_extreme_weather.csv'
            args.num_test = 2208
        elif pattern == 'extreme_clean':
            args.data_path = 'test_data_extreme_weather_clean.csv'
            args.num_test = 2208
        args.model_id = pattern

        args.source_data_path = args.data_path
        print(args)

        args.loss_method = 'adaptive'
        pred_res, metrics_df = transfer_test(args, path)
        metrics_df.insert(0, 'model', args.model)
        all_results.append(metrics_df)
        final_metrics_df = pd.concat(all_results, axis=0, ignore_index=False)
        final_metrics_df.to_csv(os.path.join(root_path, 'pf_{}_all_models_comparison.csv'.format(args.model_id)))


if __name__ == '__main__':
    # root_path = r"D:\results\pf_ver2\zero-shot\kansai"  # 替换为实际路径
    # root_path = r'D:\results\pf_ver2\patchTST\tohoku'
    # regions_test(root_path)

    root_path = r'D:\results\pf_ver2\Robustness\extreme_clean'  # extreme extreme_clean  shift
    extreme_weather_test(root_path,pattern='extreme_clean')


