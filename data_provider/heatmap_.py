from utils.tools import heatmap
import pandas as pd
import os

folder = r'D:\Time-LLM-main\dataset\HVAC'
data_file = 'summary.csv'
data = pd.read_csv(os.path.join(folder, data_file),index_col=0)
data = data[['Dry_Bulb_Temperature', 'Wet_Bulb_Temperature', 'Relative_Humidity', 'Total_Cooling_Capacity', 'Total_Power', 'System_COP', 'Cumulative_Total_Cooling',
                     'Cumulative_Total_Energy_Consumption', 'System_Energy_Efficiency', 'Chiller_Efficiency', 'Chilled_Water_Pump_Efficiency', 'Cooling_Water_Pump_Efficiency',
                     'Cooling_Tower_Efficiency', 'Total_Chiller_Power', 'Primary_Chilled_Water_Pump_Total_Power', 'Secondary_Chilled_Water_Pump_Total_Power_Process',
                     'Secondary_Chilled_Water_Pump_Total_Power_AC', 'Cooling_Water_Pump_Total_Power', 'Cooling_Tower_Total_Power', 'Chilled_Water_Distribution_Coefficient',
                     'Total_Chilled_Water_Flowrate', 'Chilled_Water_Supply_Temperature', 'Chilled_Water_Return_Temperature', 'Chilled_Water_Temperature_Difference',
                     'Chilled_Water_Supply_Pressure', 'Chilled_Water_Return_Pressure', 'Chilled_Water_Pressure_Difference', 'Chilled_Water_Bypass_Temperature',
                     'Cooling_Water_Supply_Temperature', 'Cooling_Water_Return_Temperature', 'Cooling_Water_Temperature_Difference', 'Cooling_Water_Bypass_Valve_Opening',
                     'Process_Chilled_Water_Flowrate', 'Process_Chilled_Water_Supply_Temperature', 'Process_Chilled_Water_Return_Temperature',
                     'Process_Chilled_Water_Supply_Pressure', 'Process_Chilled_Water_Pressure_Difference', 'Process_Chilled_Water_Bypass_Valve_Opening',
                     'AC_Chilled_Water_Flowrate', 'AC_Chilled_Water_Supply_Temperature', 'AC_Chilled_Water_Return_Temperature', 'AC_Chilled_Water_Supply_Pressure',
                     'AC_Chilled_Water_Pressure_Difference', 'AC_Chilled_Water_Bypass_Valve_Opening'] ]
output_file = 'summary_heatmap'
heatmap(data,output_file)