import numpy as np
import pandas as pd
import os
from utils.tools import results_evaluation

def res_evaluation(res_path,data=None):
    if data is None:
        pred = np.load(os.path.join(res_path, 'pred.npy'))
        true = np.load(os.path.join(res_path, 'true.npy'))
    else:
        pred = np.load(os.path.join(res_path, 'pred_{}.npy'.format(data)))
        true = np.load(os.path.join(res_path, 'true_{}.npy'.format(data)))
    pred_output = pred.squeeze()[::24,:].reshape(-1,1)
    true_output = true.squeeze()[::24,:].reshape(-1,1)
    pred_res = pd.DataFrame({'pred':pred_output.flatten(), 'true':true_output.flatten()})
    pred_res.loc[pred_res['true'] < 0.001, 'true'] = 0
    pred_res.loc[pred_res['true'] == 0, 'pred'] = 0
    [mse, rmse, mae, r2,corr] = results_evaluation(pred_res['true'].values, pred_res['pred'].values)
    if data is None:
        pred_res.to_csv(os.path.join(res_path, 'pred_results.csv'))
        metrics_df = pd.DataFrame({'mae':mae,'rmse':rmse,'r2':r2},index=[0])
        metrics_df.to_csv(os.path.join(res_path, 'metrics_results.csv'))
    else:
        pred_res.to_csv(os.path.join(res_path, 'pred_results_{}.csv'.format(data)))
        metrics_df = pd.DataFrame({'mae': mae, 'rmse': rmse, 'r2': r2}, index=[0])
        metrics_df.to_csv(os.path.join(res_path, 'metrics_results_{}.csv'.format(data)))
    print('RMSE: {} MAE: {} R2: {}'.format(rmse,mae,r2))

if __name__ == '__main__':
    # data = 'Sapporo'
    data = None
    res_path = r'D:\Time-LLM-main\results\sr_Transformer_Tokyo_ftM_sl72_ll24_pl24_sd9_dm256_nh8_el4_dl2_df1024_fc3_dropout0.1_ebtimeF_test_0'
    res_evaluation(res_path,data)