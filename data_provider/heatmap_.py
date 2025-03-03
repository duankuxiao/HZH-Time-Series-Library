from utils.tools import heatmap
import pandas as pd
import os

folder = r'D:\Time-LLM-main\dataset\aircon'
data_file = 'aircon.csv'
data = pd.read_csv(os.path.join(folder, data_file),index_col=0,encoding='SHIFT-JIS')
data = data[['ac3_fanspeed', 'ac4_fanspeed','ac3_temp', 'ac4_temp','Temperature', 'Relative_humidity','ac3_rh_ra', 'ac4_rh_ra',
       'ac3_rh_sa', 'ac4_rh_sa', '2F', '1F_room1', '1F_room2', 'ac3_ra', 'ac4_ra', 'ac3_sa', 'ac4_sa',  'ac3_power', 'ac4_power','PV',  'Wind_speed_mean','Wind_speed_max',  'Sunshine_duration']]
output_file = 'aicron'
heatmap(data,output_file)