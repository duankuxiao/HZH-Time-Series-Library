from utils.tools import heatmap
import pandas as pd
import os

folder = r'G:\我的云端硬盘\☆_論文\sci\19_\新建文件夹'
data_file = 'interpolated_damper_data_5min.csv'
data = pd.read_csv(os.path.join(folder, data_file),index_col=0,encoding='SHIFT-JIS')
data = data[['Ti_A', 'Ti_B', 'Temperature_air', 'Solar_east', 'Solar_south',
       'Solar_west', 'WindVelocity', 'Setpoint_A', 'Damper_A', 'Damper_B',
       'Window_A', 'Window_B', 'CO2_A', 'CO2_B', 'InletFlow_A', 'InletFlow_B']]
output_file = 'interpolated_damper_data_5min'
heatmap(data,output_file)