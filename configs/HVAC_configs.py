from configs.common_configs import args

args.model_comment = 'HVAC'  # 能源供应
args.data = 'hvac'
args.root_path = './dataset/HVAC'
args.data_path = 'summary.csv'
# args.data_path = 'kyushu.csv'
# args.data_path = 'Texas.csv'

args.source_data_path = args.data_path

args.features = 'M'
# args.feature_cols = ['Dry_Bulb_Temperature', 'Wet_Bulb_Temperature', 'Relative_Humidity', 'Total_Cooling_Capacity', 'Total_Power', 'System_COP', 'Cumulative_Total_Cooling',
#                      'Cumulative_Total_Energy_Consumption', 'System_Energy_Efficiency', 'Chiller_Efficiency', 'Chilled_Water_Pump_Efficiency', 'Cooling_Water_Pump_Efficiency',
#                      'Cooling_Tower_Efficiency', 'Total_Chiller_Power', 'Primary_Chilled_Water_Pump_Total_Power', 'Secondary_Chilled_Water_Pump_Total_Power_Process',
#                      'Secondary_Chilled_Water_Pump_Total_Power_AC', 'Cooling_Water_Pump_Total_Power', 'Cooling_Tower_Total_Power', 'Chilled_Water_Distribution_Coefficient',
#                      'Total_Chilled_Water_Flowrate', 'Chilled_Water_Supply_Temperature', 'Chilled_Water_Return_Temperature', 'Chilled_Water_Temperature_Difference',
#                      'Chilled_Water_Supply_Pressure', 'Chilled_Water_Return_Pressure', 'Chilled_Water_Pressure_Difference', 'Chilled_Water_Bypass_Temperature',
#                      'Cooling_Water_Supply_Temperature', 'Cooling_Water_Return_Temperature', 'Cooling_Water_Temperature_Difference', 'Cooling_Water_Bypass_Valve_Opening',
#                      'Process_Chilled_Water_Flowrate', 'Process_Chilled_Water_Supply_Temperature', 'Process_Chilled_Water_Return_Temperature',
#                      'Process_Chilled_Water_Supply_Pressure', 'Process_Chilled_Water_Pressure_Difference', 'Process_Chilled_Water_Bypass_Valve_Opening',
#                      'AC_Chilled_Water_Flowrate', 'AC_Chilled_Water_Supply_Temperature', 'AC_Chilled_Water_Return_Temperature', 'AC_Chilled_Water_Supply_Pressure',
#                      'AC_Chilled_Water_Pressure_Difference', 'AC_Chilled_Water_Bypass_Valve_Opening']  # 'Temperature', 'Dewpoint', 'Humidity'


'''
Cumulative_Chiller_Energy_Consumption','Cumulative_Primary_Chilled_Water_Pump_Energy', 'Cumulative_Secondary_Chilled_Water_Pump_Energy', 'Cumulative_Cooling_Water_Pump_Energy ',
'Cumulative_Cooling_Tower_Energy', 'Chiller_COP', 'Free_Cooling_Condition_Satisfied', 'Plate_Heat_Exchanger_Supply_Temperature', 'Plate_Heat_Exchanger_Return_Temperature',
'Cooling_Tower_Discharge_Temperature_Setpoint', 'Chilled_Water_Pump_Hydraulic_Balance_Setpoint', 'Process_Chilled_Water_Bypass_Valve_Differential_Pressure_Setpoint', 
'AC_Chilled_Water_Bypass_Valve_Differential_Pressure_Setpoint', 'System_Chilled_Water_Supply_Temperature_Setpoint', 'CTO_Optimal_Discharge_Temperature_Setpoint',
'Free_Cooling_Max_Wet_Bulb_Temperature_Limit'
'''

args.feature_cols = ['Total_Chiller_Power', 'Total_Chilled_Water_Flowrate', 'Process_Chilled_Water_Flowrate', 'Wet_Bulb_Temperature', 'AC_Chilled_Water_Flowrate',
                     'Dry_Bulb_Temperature', 'Chilled_Water_Distribution_Coefficient', 'Process_Chilled_Water_Pressure_Difference', 'Cooling_Water_Return_Temperature',
                     'Cooling_Tower_Total_Power', 'Chilled_Water_Temperature_Difference', 'Cooling_Water_Supply_Temperature', 'Secondary_Chilled_Water_Pump_Total_Power_Process',
                     'Process_Chilled_Water_Supply_Pressure', 'Chilled_Water_Supply_Pressure', 'Cooling_Water_Pump_Total_Power', 'Primary_Chilled_Water_Pump_Total_Power',
                     'Chilled_Water_Pump_Efficiency', 'Chilled_Water_Pressure_Difference', 'AC_Chilled_Water_Supply_Temperature', 'Process_Chilled_Water_Bypass_Valve_Opening',
                     'Process_Chilled_Water_Supply_Temperature', 'Chilled_Water_Return_Temperature', 'Chilled_Water_Supply_Temperature', 'AC_Chilled_Water_Return_Temperature',
                     'Chilled_Water_Return_Pressure', 'Total_Power', 'System_COP', 'System_Energy_Efficiency']

args.target = ['Total_Power', 'System_COP', 'System_Energy_Efficiency']


args.num_train = 35136  # 2023/03/06 - 2024/03/05
args.num_test = 16032
args.seq_len = 4 * 24
args.pred_len = 4 * 24
args.label_len = args.seq_len
args.forecast_dim = 1
args.enc_in = len(args.feature_cols)
args.dec_in = len(args.feature_cols)
args.c_out = 3
args.scale = True
args.val = False



if __name__ == '__main__':
    import pandas as pd
    import os
    data = pd.read_csv(os.path.join(r'D:\Time-LLM-main\dataset\HVAC', 'summary_cleaned.csv'))
    print(data.columns)

