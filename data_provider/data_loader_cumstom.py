import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import torch

warnings.filterwarnings('ignore')

class Dataset_cumstom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='PV_power.csv',
                 target='PV', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None,forecast_dim=2,feature_cols=['Temperature']):
        self.forecast_dim = forecast_dim
        self.feature_cols = feature_cols

        if size == None:
            self.seq_len = 24 * 3
            self.label_len = 24
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, :]
        seq_y = self.data_y[r_begin:r_end, :]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        x_forecast = self.data_forecast[r_begin:r_end, :self.forecast_dim]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, x_forecast

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        raise NotImplementedError

    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)


class Dataset_PV_hour(Dataset_cumstom):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='PV_power.csv',
                 target='PV', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None,forecast_dim=2,feature_cols=['Temperature']):
        super(Dataset_PV_hour, self).__init__(root_path, flag, size, features,data_path,target,scale,timeenc,freq,percent,seasonal_patterns,forecast_dim,feature_cols)

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.feature_cols is None:
            self.feature_cols = df_raw.columns[1:]
        cols = list(self.feature_cols.copy())
        if self.target in cols:
            cols.remove(self.target)
        if 'date' in cols:
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = 8760
        num_test = 8760
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # if self.features == 'M' or self.features == 'MS':
        #     cols_data = df_raw.columns[1:]
        #     df_data = df_raw[cols_data]
        #     df_target = df_raw[[self.target]]
        # elif self.features == 'S':
        #     df_data = df_raw[[self.target]]
        #     df_target = df_raw[[self.target]]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_target = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.target_scaler.fit(df_target.values)
            df_target = self.target_scaler.transform(df_target.values)
        else:
            data = df_data.values
            df_target = df_target.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.features == 'M' or self.features == 'MS':
            self.data_x = data[border1:border2,:len(self.feature_cols)]
            self.data_y = data[border1:border2,:len(self.feature_cols)]
        elif self.features == 'S':
            self.data_x = data[border1:border2, -1:]
            self.data_y = data[border1:border2, -1:]

        self.data_forecast = data[border1:border2, :self.forecast_dim]
        self.data_stamp = data_stamp


class Dataset_solar_radiation(Dataset_cumstom):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='Tokyo.csv',
                 target='Global_horizontal_irradiance', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None,forecast_dim=2,feature_cols=['Temperature']):
        super(Dataset_solar_radiation, self).__init__(root_path, flag, size, features,data_path,target,scale,timeenc,freq,percent,seasonal_patterns,forecast_dim,feature_cols)

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        df_source_domain = pd.read_csv(os.path.join(self.root_path, 'Tokyo.csv'))
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.feature_cols is None:
            self.feature_cols = df_raw.columns[1:]
        cols = list(self.feature_cols.copy())
        if self.target in cols:
            cols.remove(self.target)
        if 'date' in cols:
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        df_source_domain = df_source_domain[['date'] + cols + [self.target]]

        num_train = 8760 * 2 + 24  # int(len(df_raw) * 0.7)
        num_test = 8760  # int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        # num_train = int(8760 * 0.3)
        # num_test = int(8760 * 0.7)
        # num_vali = len(df_raw) - num_train - num_test
        # border1s = [0, 8760 - num_test - self.seq_len, 8760 - num_test - self.seq_len]
        # border2s = [num_train, 8760, 8760]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_target = df_raw[[self.target]]

        cols_data_source_domain = df_source_domain.columns[1:]
        df_data_source_domain = df_source_domain[cols_data_source_domain]
        df_target_source_domain = df_source_domain[[self.target]]

        if self.scale:
            '''
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.target_scaler.fit(df_target.values)
            '''

            train_data = df_data_source_domain[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.target_scaler.fit(df_target_source_domain.values)

        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.features == 'M' or self.features == 'MS':
            self.data_x = data[border1:border2, :len(self.feature_cols)]
            self.data_y = data[border1:border2, :len(self.feature_cols)]
        elif self.features == 'S':
            self.data_x = data[border1:border2, -1:]
            self.data_y = data[border1:border2, -1:]

        self.data_forecast = data[border1:border2, :2]
        self.data_stamp = data_stamp


if __name__ == '__main__':
    from utils.tools import heatmap
    folder = r'D:\Time-LLM-main\dataset\price'
    data_file = 'Tokyo.csv'
    data = pd.read_csv(os.path.join(folder, data_file),index_col=0)
    heatmap(data)
