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
from run_ele import get_setting


def transfer_test(args, path):
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
    pred_res,metrics_df = exp.test(setting, test=1, path=path)
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

    root_path = r"D:\Time-LLM-main\results\res\tokyo"  # 替换为实际路径
    subfolders = get_direct_subfolders(root_path)
    print(subfolders)

    # path = r'D:\Time-LLM-main\results\res\kyushu\kyushu_TimesNet_electricity_ftM_sl72_ll24_pl168_sd22_td3_dm64_nh8_el2_dl1_df256_fc3_dropout0.1_ebtimeF_test_0_scale'
    for path in subfolders:
        args = load_config(os.path.join(path,'checkpoints','configs.pkl'))

        renamed_dfs = []
        for city in ['tokyo','hokkaido','tohoku','kyushu']:
        # for city in ['kyushu']:
        #     args.feature_cols = ['Electricity', 'Renewable_energy', 'Coal']  # ['Electricity', 'Renewable_energy', 'Coal']
            # args.feature_cols = ['Coal']  # ['Electricity', 'Renewable_energy', 'Coal']

            # args.target = ['Electricity', 'Renewable_energy', 'Coal']  # ['Electricity', 'Renewable_energy', 'Coal']  _Electricity  _Renewable_energy _Coal
            # args.target = ['Electricity', 'Renewable_energy', 'Coal']
            args.data_path = '{}.csv'.format(city)
            args.source_data_path = args.data_path
            args.features = 'M'
            pred_res, metrics_df = transfer_test(args, path)
            metrics_df.index = metrics_df.index +'_'+ city
            renamed_dfs.append(metrics_df)
            combined_df = pd.concat(renamed_dfs, axis=0)
            combined_df.to_csv(os.path.join(path,'zero-shot_res_metrics.csv'.format()))

