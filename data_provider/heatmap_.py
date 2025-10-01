from utils.tools import heatmap
import pandas as pd
import os

folder = '../dataset/hvac-onsite'
data_file = 'aeb_21-23_clean.csv'
data = pd.read_csv(os.path.join(folder, data_file),index_col=0)
data = data[['Cooling', 'Heating', 'Electricity', 'Cooling simulation', 'Electricity simulation', 'Heating simulation', 'Drybulb', 'Dew point', 'Humidity',
                     'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'Global Horizontal Illuminance',
                     'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance', 'Global Horizontal Infrared Radiation', 'Direct Normal Infrared Radiation',
                     'Diffuse Horizontal Infrared Radiation', 'UV Index', 'WindSpeed', 'Atmospheric Pressure', 'Aerosol Optical Depth']]
output_file = 'summary_heatmap-hvac-onsite'
heatmap(data,output_file)


# args.root_path = './dataset/hvac-onsite'
# args.data_path = 'aeb_21-23_clean.csv'
# args.feature_cols = ['Cooling', 'Heating', 'Electricity', 'Cooling simulation', 'Electricity simulation', 'Heating simulation', 'Drybulb', 'Dew point', 'Humidity',
#                      'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation', 'Global Horizontal Illuminance',
#                      'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance', 'Global Horizontal Infrared Radiation', 'Direct Normal Infrared Radiation',
#                      'Diffuse Horizontal Infrared Radiation', 'UV Index', 'WindSpeed']  # , 'Atmospheric Pressure', 'Aerosol Optical Depth'