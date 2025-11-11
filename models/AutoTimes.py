import datetime
import os

import pandas as pd
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, Qwen2Tokenizer, Qwen2Model, Qwen2Config
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=256,
                 hidden_layers=2,
                 dropout=0.1,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.patch_len
        self.use_norm = configs.use_norm
        self.output_ori = configs.output_ori
        self.task_name = configs.task_name
        self.gpt2_config = GPT2Config.from_pretrained(r'D:\LLM\gpt2')

        self.gpt2 = GPT2Model.from_pretrained(
                    r'D:\LLM\gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
        self.hidden_dim_of_gpt2 = 768
        self.mix = True

        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        if configs.hidden_size is None:
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec,mask=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, _, n_vars = x_enc.shape

        x_enc = x_enc.permute(0, 2, 1)  # [B, n_vars, len]

        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)  # [B * nvars, seq_len]
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)  # [B * nvars, token_num, token_len]
        token_num = fold_out.shape[1]
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_gpt2]
        times_embeds = self.encoder(fold_out)
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        # outputs: [bs * n_vars x token_num x hidden_dim_of_gpt2]
        outputs = self.gpt2(
            inputs_embeds=times_embeds).last_hidden_state
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(B, n_vars, -1)
        # dec_out: [bs x token_num * token_len x n_vars]
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,x_forecast=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            if self.output_ori:
                dec_out[:, :self.seq_len, -self.c_out:] = mask[:, :, -self.c_out:] * x_enc[:, :self.seq_len, -self.c_out:] + (1 - mask[:, :, -self.c_out:]) * dec_out[:, :self.seq_len,
                                                                                                                                                           -self.c_out:]
            return dec_out


class Dataset_Custom(Dataset):
    def __init__(self, configs,root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True,features='M',target=None,freq=None,percent=None,timeenc=None, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.num_train = configs.num_train
        self.num_test = configs.num_test
        self.task_name = configs.task_name
        self.feature_cols = configs.feature_cols
        self.c_out = configs.c_out
        self.features = features
        self.target = target
        self.data_name = configs.data
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len

        self.flag = flag

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        df_source_domain = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.feature_cols is None:
            self.feature_cols = df_raw.columns[1:]
        cols = list(self.feature_cols.copy())

        if 'date' in cols:
            cols.remove('date')

        if self.features == 'M' and self.target is None:
            self.target = self.feature_cols
        if self.target is not None:
            for s in self.target:
                if s in cols:
                    cols.remove(s)
        df_raw = df_raw[['date'] + cols + self.target]
        df_source_domain = df_source_domain[['date'] + cols + self.target]

        num_vali = len(df_raw) - self.num_train - self.num_test
        if 'forecast' in self.task_name:
            border1s = [0, self.num_train - self.seq_len, len(df_raw) - self.num_test - self.seq_len]
            border2s = [self.num_train, self.num_train + num_vali, len(df_raw)]
        else:
            border1s = [0, self.num_train, len(df_raw) - self.num_test]
            border2s = [self.num_train, self.num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_target = df_raw[self.target]

        cols_data_source_domain = df_source_domain.columns[1:]
        df_data_source_domain = df_source_domain[cols_data_source_domain]
        df_target_source_domain = df_source_domain[self.target]

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
            df_target_data = df_target_source_domain[border1s[0]:border2s[0]]
            self.target_scaler.fit(df_target_data.values)

        else:
            data = df_data.values

        self.data_stamp = torch.load(os.path.join(self.root_path, f'{self.data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)


class Preprocess_forAutoTimes(nn.Module):
    def __init__(self, configs):
        super(Preprocess_forAutoTimes, self).__init__()
        self.device = configs.gpu
        self.gpt2_config = GPT2Config.from_pretrained(r'D:\LLM\gpt2')
        self.gpt2 = GPT2Model.from_pretrained(
            r'D:\LLM\gpt2',
            trust_remote_code=True,
            local_files_only=True,
            config=self.gpt2_config,
        )
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(r'D:\LLM\gpt2',)
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
        self.vocab_size = self.gpt_tokenizer.vocab_size
        self.hidden_dim_of_llama = 768

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False
        self.gpt2.to(device=self.device)

    def tokenizer(self, x_list):
        output = self.gpt_tokenizer(
            x_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )['input_ids'].to(self.device)
        result = self.gpt2.get_input_embeddings()(output)
        return result

    def forecast(self, x_mark_enc):
        x_mark_enc = self.tokenizer(x_mark_enc)
        text_outputs = self.gpt2(inputs_embeds=x_mark_enc)[0]
        text_outputs = text_outputs[:, -1, :]
        return text_outputs

    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)


class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True,data='hvac', seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_type = data
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        if self.data_type in ['price', 'electricity', 'weather']:
            end = (start + datetime.timedelta(hours=self.token_len - 1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_type in ['hvac']:
            end = (start + datetime.timedelta(minutes=15 * (self.token_len - 1))).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = f"This is Time Series from {self.data_stamp[s_begin]} to {end}"
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)


if __name__ == '__main__':
    from configs.HVAC_configs import args as default_args
    from copy import deepcopy
    from torch.utils.data import DataLoader
    args = deepcopy(default_args)

    model = Preprocess_forAutoTimes(args)

    data_set = Dataset_Preprocess(
        root_path=r'D:\Time-LLM-main\dataset\HVAC',
        data_path=args.data_path,
        data=args.data,
        size=[args.seq_len, args.label_len, args.pred_len])

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    from tqdm import tqdm

    print(len(data_set.data_stamp))
    print(data_set.tot_len)
    save_dir_path = r'D:\Time-LLM-main\dataset\HVAC'
    output_list = []
    for idx, data in tqdm(enumerate(data_loader)):
        output = model(data)
        output_list.append(output.detach().cpu())
    result = torch.cat(output_list, dim=0)
    print(result.shape)
    torch.save(result, save_dir_path + f'/{args.data}.pt')
