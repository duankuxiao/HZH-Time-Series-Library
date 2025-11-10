from configs.common_configs import args

args.model_comment = 'HVAC'  # 能源供应
args.features = 'M'

# args.data = 'hvac-onsite'
# args.root_path = './dataset/hvac-onsite'
# args.data_path = 'aeb_21-23_clean.csv'
# args.feature_cols = ['Cooling', 'Heating', 'Electricity','Drybulb', 'Dew point', 'Humidity',
#                      'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'Global Horizontal Illuminance',
#                      'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance', 'Global Horizontal Infrared Radiation', 'Direct Normal Infrared Radiation',
#                      'Diffuse Horizontal Infrared Radiation', 'UV Index', 'WindSpeed', 'Aerosol Optical Depth', 'Atmospheric Pressure']  #  'Cooling simulation', 'Electricity simulation', 'Heating simulation',
# args.target = ['Cooling', 'Heating', 'Electricity']
# args.num_train = int(26279 - 8760)  # - 8760 * 0.75)
# args.num_test = 8760
# args.seq_len = 48
# args.pred_len = 24
# args.label_len = args.seq_len
# args.forecast_dim = 1

'''
'Cumulative_Chiller_Energy_Consumption','Cumulative_Primary_Chilled_Water_Pump_Energy', 'Cumulative_Secondary_Chilled_Water_Pump_Energy', 'Cumulative_Cooling_Water_Pump_Energy ','Cumulative_Cooling_Tower_Energy', 
'Process_Chilled_Water_Pressure_Difference', 'Process_Chilled_Water_Supply_Pressure', 'Process_Chilled_Water_Supply_Temperature','Chilled_Water_Bypass_Temperature',
'AC_Chilled_Water_Supply_Temperature',  'AC_Chilled_Water_Bypass_Valve_Opening', 'Cooling_Water_Bypass_Valve_Opening', 'AC_Chilled_Water_Supply_Pressure', 'AC_Chilled_Water_Pressure_Difference',
 'Chilled_Water_Return_Pressure',  'Chilled_Water_Temperature_Difference', 'Chilled_Water_Distribution_Coefficient', 'Humidity','Process_Chilled_Water_Bypass_Valve_Opening',
'''
args.data = 'hvac'
args.root_path = './dataset/HVAC'
args.data_path = 'summary_2.csv'
args.feature_cols = ['Temperature', 'Dewpoint','Dry_Bulb_Temperature', 'Wet_Bulb_Temperature', 'Total_Chiller_Power', 'Cooling_Tower_Total_Power', 'Cooling_Water_Pump_Total_Power',
                     'Total_Chilled_Water_Flowrate', 'Process_Chilled_Water_Flowrate',  'AC_Chilled_Water_Flowrate', 'Cooling_Water_Return_Temperature', 'Chiller_COP',
                     'Chilled_Water_Supply_Pressure', 'Chilled_Water_Pressure_Difference', 'Cooling_Water_Supply_Temperature',  'Secondary_Chilled_Water_Pump_Total_Power_Process',
                     'Primary_Chilled_Water_Pump_Total_Power', 'Total_Cooling_Capacity',
                     'Chilled_Water_Pump_Efficiency',   'Chilled_Water_Bypass_Temperature', 'Chilled_Water_Return_Temperature', 'Chilled_Water_Supply_Temperature',
                     'Total_Power', 'System_COP', 'System_Energy_Efficiency']


args.target = ['Total_Power','Total_Chiller_Power','System_Energy_Efficiency','Total_Cooling_Capacity']
args.num_train = 35136  # 2023/03/08 - 2024/03/07
args.num_test = 15840  # 2024/03/08 - 2024/08/19
args.seq_len = 12
args.pred_len = args.seq_len
args.label_len = args.seq_len

args.c_out = len(args.target)
args.forecast_dim = 2
args.source_data_path = args.data_path
args.enc_in = len(args.feature_cols)
args.dec_in = len(args.feature_cols)

args.scale = True
args.val = False


if __name__ == '__main__':
    import pandas as pd
    import os
    import numpy as np
    from scipy.stats import zscore

    df = pd.read_csv(os.path.join(r'D:\Time-LLM-main\dataset\HVAC', 'summary_2.csv'))
    # === 参数设置 ===
    z_threshold = 10  # 超过这个z-score就算异常

    for col in df.select_dtypes(include=[np.number]).columns:  # 仅处理数值列
        series = df[col]
        z_scores = zscore(series, nan_policy='omit')  # 计算 z-score（忽略 NaN）

        # 标记异常值（绝对 z-score > 阈值）
        outliers = np.abs(z_scores) > z_threshold

        # 输出异常信息
        if outliers.any():
            print(f"列：{col} 发现 {outliers.sum()} 个异常值")
            print("异常行索引和值如下：")
            print(df.loc[outliers, [col]])

            # 将异常值替换为 NaN
            df.loc[outliers, col] = np.nan

            # 使用线性插值补全异常值
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
        else:
            print(f"列：{col} 未发现异常值")

    # === 对所有列执行插值补全 ===
    # df.to_csv(os.path.join(r'D:\Time-LLM-main\dataset\HVAC', 'summary_2.csv'))
