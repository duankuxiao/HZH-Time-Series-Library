import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.interval_forecasting_tools import gaussian_sample, negative_binomial_sample
from .Distribution import Gaussian,NegativeBinomial

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.c_out = configs.c_out
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.use_forecast = configs.use_forecast
        self.likelihood = configs.likelihood
        self.use_norm = configs.use_norm
        if self.use_forecast:
            self.forecast_projection = nn.Linear(configs.forecast_dim, configs.enc_in)
        if configs.rnn_model == 'GRU':
            self.rnn_layer = nn.GRU(configs.enc_in, configs.rnn_dim, configs.rnn_layers, batch_first=True)
        elif configs.rnn_model == 'LSTM':
            self.rnn_layer = nn.LSTM(configs.enc_in, configs.rnn_dim, configs.rnn_layers, batch_first=True)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'interval_forecast':
            self.linear_predict = nn.Linear(configs.seq_len, configs.pred_len)
            if self.use_forecast:
                self.linear_predict = nn.Linear(configs.seq_len + configs.pred_len, configs.pred_len)
        if self.task_name == 'imputation_forecast':
            self.linear_predict = nn.Linear(configs.seq_len, configs.seq_len + configs.pred_len)

        if self.task_name == 'interval_forecast':
            if configs.likelihood == "g":
                self.likelihood_layer = Gaussian(configs.rnn_dim, configs.c_out)
            elif configs.likelihood == "nb":
                self.likelihood_layer = NegativeBinomial(configs.rnn_dim, configs.c_out)
            else:
                self.likelihood_layer = Gaussian(configs.rnn_dim, configs.c_out)

        self.output_projection = nn.Linear(configs.rnn_dim, configs.c_out)
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.output_projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x_enc,x_forecast=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if self.use_forecast:
            if self.use_norm:
                means_forecast = x_forecast.mean(1, keepdim=True).detach()
                x_enc_forecast = x_forecast - means_forecast
                stdev_forecast = torch.sqrt(torch.var(x_enc_forecast, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x_forecast /= stdev_forecast
            x_forecast_ = self.forecast_projection(x_forecast[:,-self.pred_len:,:])
            x_enc = torch.cat((x_enc, x_forecast_), dim=1)

        x,_ = self.rnn_layer(x_enc)
        if 'forecast' in self.task_name:
            x = self.linear_predict(x.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = self.output_projection(x)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1))
        return dec_out

    def forecast(self, x_enc,x_forecast=None):
        return self.encoder(x_enc,x_forecast)

    def interval_forecast(self, x_enc, x_forecast=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if self.use_forecast:
            # means_forecast = x_forecast.mean(1, keepdim=True).detach()
            # x_enc_forecast = x_forecast - means_forecast
            # stdev_forecast = torch.sqrt(torch.var(x_enc_forecast, dim=1, keepdim=True, unbiased=False) + 1e-5)
            # x_forecast /= stdev_forecast
            x_forecast_ = self.forecast_projection(x_forecast[:, -self.pred_len:, :])
            x_enc = torch.cat((x_enc, x_forecast_), dim=1)

        x, _ = self.rnn_layer(x_enc)
        x = self.linear_predict(x.permute(0, 2, 1)).permute(0, 2, 1)

        mu, sigma = self.likelihood_layer(x)

        dec_out = self.output_projection(x)
        if self.use_norm:
            # # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -self.c_out:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out, mu, sigma

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,x_forecast=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc,x_forecast)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation_forecast':
            dec_out = self.forecast(x_enc,x_forecast)
            dec_out[:, :self.seq_len, -self.c_out:] = mask[:, :, -self.c_out:] * x_enc[:, :self.seq_len, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * dec_out[:, :self.seq_len, -self.c_out:]
            return dec_out  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            dec_out = mask[:, :, -self.c_out:] * x_enc[:, :, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * dec_out
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        if self.task_name == 'interval_forecast':
            dec_out, mu, sigama = self.interval_forecast(x_enc, x_forecast)
            return dec_out, mu, sigama
        return None
