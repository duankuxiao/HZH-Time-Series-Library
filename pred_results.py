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
    pred_res.loc[pred_res['true'] < 0.01, 'true'] = 0
    pred_res.loc[pred_res['true'] == 0, 'pred'] = 0
    pred_res.loc[pred_res['pred'] < 0, 'pred'] = 0

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

import os
import pandas as pd




if __name__ == '__main__':
    # 设置主目录路径
    import os
    import pandas as pd

    main_folder = r'D:\Time-LLM-main\results\interval_forecast\pred_len'

    # 存储处理后的所有模型的 mean 数据
    all_mean_rows = []

    for filename in os.listdir(main_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(main_folder, filename)
            df = pd.read_csv(file_path)

            # 提取预测长度（例如从 'result_24.csv' 中提取 24）
            pred_len = int("".join(filter(str.isdigit, filename)))

            # 筛选 mean 行
            mean_rows = df[df.iloc[:, 0] == 'mean']

            for _, row in mean_rows.iterrows():
                model = row['model']
                metrics = row.iloc[2:]  # 除去前两列（预测目标和模型名）
                metrics.index = [f"{col}_{model}" for col in metrics.index]  # 指标_模型名
                metrics_df = metrics.to_frame().T
                metrics_df.insert(0, 'pred_len', pred_len)
                all_mean_rows.append(metrics_df)

    # 合并所有 mean 行
    final_df = pd.concat(all_mean_rows, ignore_index=True)

    # 将 pred_len 作为 index，按列合并所有模型
    final_df = final_df.groupby('pred_len').first()
    final_df.to_csv("summary_metrics_pred_len.csv")

    print("✅ 所有模型 mean 行合并完成！")
