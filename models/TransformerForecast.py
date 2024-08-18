import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self,args,in_size,out_size):
        super(MLP,self).__init__()

        self.hidden_layers = nn.ModuleList()
        in_features = in_size
        for hidden_size in args.hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_features, hidden_size))
            self.hidden_layers.append(nn.ReLU())
            in_features = hidden_size

        self.fc_output = nn.Linear(args.hidden_sizes[-1],out_size)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self,x):
        for layer in self.hidden_layers:
            x = self.dropout(x)
            x = layer(x)

        output = self.fc_output(x)
        output = self.dropout(output)
        return output


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # [5000,7]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [5000, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [4]
        pe[:, 0::2] = torch.sin(position * div_term)  # [5000,4]
        pe[:, 1::2] = torch.cos(position * div_term)  # [5000,3]
        pe = pe.unsqueeze(0)  # [1,5000,16]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]  # [32, 24, 7] [1, 24, 16]
        # pos_output = input_seq + Variable(self.pe[:, :input_seq.size(1), :], requires_grad=False)  # [32, 24, 7] [1, 24, 16]
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.config = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.pos_emb = PositionalEmbedding(configs.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=4 * configs.d_model,
            batch_first=True,
            dropout=configs.dropout,
            device=configs.device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dropout=configs.dropout,
            dim_feedforward=4 * configs.d_model,
            batch_first=True,
            device=configs.device
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=configs.e_layers)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=configs.d_layers)

        self.fc_enc_out = nn.Linear(self.seq_len, self.pred_len)
        self.fc_dec_input = nn.Linear(configs.d_model + 3, configs.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len)

            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

        self.projection = nn.Linear(configs.d_model, 1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_forecast):
        enc_out = self.encoder(self.pos_emb(x_enc))
        enc_out = self.fc_enc_out(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_in = self.fc_dec_input(torch.cat((enc_out, x_forecast[:, :, :3]), dim=2))
        dec_out = self.decoder(self.pos_emb(dec_in), enc_out)
        dec_out = self.projection(dec_out[:,-1:,:])
        return dec_out.permute(0, 2, 1)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,x_forecast=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,x_forecast)
            return dec_out[:, :, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
