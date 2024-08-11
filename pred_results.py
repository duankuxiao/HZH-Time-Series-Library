import numpy as np
import pandas as pd
import os
from utils.tools import results_evaluation

def res_evaluation(res_path):
    pred = np.load(os.path.join(res_path, 'pred.npy'))
    true = np.load(os.path.join(res_path, 'true.npy'))
    pred_output = pred.squeeze()[::24,:].reshape(-1,1)
    true_output = true.squeeze()[::24,:].reshape(-1,1)
    pred_res = pd.DataFrame({'pred':pred_output.flatten(), 'true':true_output.flatten()})
    pred_res.loc[pred_res['true'] < 0.001, 'true'] = 0
    pred_res.loc[pred_res['true'] == 0, 'pred'] = 0
    [mse, rmse, mae, r2,corr] = results_evaluation(pred_res['true'].values, pred_res['pred'].values)
    pred_res.to_csv(os.path.join(res_path, 'pred_results.csv'))
    metrics_df = pd.DataFrame({'mae':mae,'rmse':rmse,'r2':r2},index=[0])
    metrics_df.to_csv(os.path.join(res_path, 'metrics_results.csv'))
    print('RMSE: {} MAE: {} R2: {}'.format(rmse,mae,r2))

if __name__ == '__main__':
    res_path = r'D:\Time-LLM-main\results\long_term_forecast_PV_Transformer_PV_ftM_sl72_ll24_pl24_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_pv_0'
    res_evaluation(res_path)