from utils.tools import load_config,save_config
import os
import torch
import os
from utils.print_args import print_args

def get_setting(args,ii):
    setting = 'if_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_td{}_dm{}_df{}_el{}_dl{}_nh{}_ma{}_factor{}_{}'.format(
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
        args.factor, args.loss)

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


def transfer_test(args, path):
    if args.task_name == 'interval_forecast':
        from exp.exp_interval_forecasting import Exp_Forecast
    else:
        from exp.exp_forecasting import Exp_Forecast

    ii = 0
    setting = get_setting(args,ii)

    exp = Exp_Forecast(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    pred_res, metrics_df = exp.test(setting, test=1, path=path)
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


if __name__ == '__main__':
    import pandas as pd

    root_path = r"D:\Time-LLM-main\results\test7"  # 替换为实际路径
    subfolders = get_direct_subfolders(root_path)
    # print(subfolders)

    # path = r'D:\Time-LLM-main\results\res\kyushu\kyushu_TimesNet_electricity_ftM_sl72_ll24_pl168_sd22_td3_dm64_nh8_el2_dl1_df256_fc3_dropout0.1_ebtimeF_test_0_scale'
    for path in subfolders:
        renamed_dfs = []
        # path = r'D:\Time-LLM-main\results\zero-shot-kyushu_Informer_electricity_ftM_sl72_ll24_pl168_sd8_td3_lr0.001_dm64_df256_nh8_el4_dl2_ma25_factor5_dropout0.1_ebtimeF_0_scale'

        args = load_config(os.path.join(path,'checkpoints','configs.pkl'))
        print(args)
        # for city in ['tokyo','hokkaido','tohoku','kyushu']:
        # for city in ['Sapporo','Sendai','Tokyo','Fukuoka']:
        for time in ['30min','1hour']:
        #     args.feature_cols = ['Electricity', 'Renewable_energy', 'Coal']  # ['Electricity', 'Renewable_energy', 'Coal']
            # args.feature_cols = ['Coal']  # ['Electricity', 'Renewable_energy', 'Coal']

        # args.target = ['Electricity', 'Renewable_energy', 'Coal']  # ['Electricity', 'Renewable_energy', 'Coal']  _Electricity  _Renewable_energy _Coal
        # args.target = ['Electricity', 'Renewable_energy', 'Coal']
            args.data_path = 'operational_data_{}.csv'.format(time)
            args.source_data_path = args.data_path
            args.num_train = 2000
            if time == '30min':
                args.num_test = 8833  # 2929
            elif time == '1hour':
                args.num_test = 4417  # 1465
            pred_res, metrics_df = transfer_test(args, path)
            metrics_df.index = metrics_df.index + '_' + time
            renamed_dfs.append(metrics_df)
            combined_df = pd.concat(renamed_dfs, axis=0)
            combined_df.to_csv(os.path.join(path,'zero-shot_res_metrics.csv'.format()))