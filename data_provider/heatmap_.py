from utils.tools import heatmap
import pandas as pd
import os

folder = '../dataset/HVAC'
data_file = 'summary_.csv'
data = pd.read_csv(os.path.join(folder, data_file),index_col=0)
data = data[['Total_Chilled_Water_Flowrate', 'Process_Chilled_Water_Flowrate', 'Wet_Bulb_Temperature', 'AC_Chilled_Water_Flowrate',
                     'Dry_Bulb_Temperature',   'Cooling_Water_Return_Temperature', 'Chiller_COP', 'Chilled_Water_Supply_Pressure', 'Chilled_Water_Pressure_Difference',
                     'Cooling_Tower_Total_Power',  'Cooling_Water_Supply_Temperature',  'Secondary_Chilled_Water_Pump_Total_Power_Process',
                       'Cooling_Water_Pump_Total_Power', 'Primary_Chilled_Water_Pump_Total_Power', 'Temperature', 'Dewpoint',
                     'Chilled_Water_Pump_Efficiency', 'Chilled_Water_Bypass_Temperature',
                     'Chilled_Water_Return_Temperature', 'Chilled_Water_Supply_Temperature','System_COP',
                      'Total_Power','Total_Chiller_Power','System_Energy_Efficiency','Total_Cooling_Capacity']]
# data = data[[ 'Chilled_Water_Supply_Pressure', 'Chilled_Water_Return_Pressure','Process_Chilled_Water_Supply_Pressure','Chilled_Water_Pressure_Difference', 'Process_Chilled_Water_Pressure_Difference','Temperature', 'Dewpoint', 'Humidity','Total_Cooling_Capacity','Total_Power', 'System_COP', 'System_Energy_Efficiency']]
output_file = 'hvac_2'
heatmap(data,output_file)


# args.root_path = './dataset/hvac-onsite'
# args.data_path = 'aeb_21-23_clean.csv'
# args.feature_cols = ['Cooling', 'Heating', 'Electricity', 'Cooling simulation', 'Electricity simulation', 'Heating simulation', 'Drybulb', 'Dew point', 'Humidity',
#                      'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'Global Horizontal Illuminance',
#                      'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance', 'Global Horizontal Infrared Radiation', 'Direct Normal Infrared Radiation',
#                      'Diffuse Horizontal Infrared Radiation', 'UV Index', 'WindSpeed']  # , 'Atmospheric Pressure', 'Aerosol Optical Depth'