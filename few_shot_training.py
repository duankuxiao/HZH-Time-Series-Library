from utils.tools import load_config,save_config
import torch
import os
from exp.exp_few_shot import Exp_FewShot
from utils.print_args import print_args


def few_shot_training(args,shot_num,path):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    args.inverse = True
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_FewShot

    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_sd{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_dropout{}_eb{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.seq_dim,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.dropout,
        args.embed,
        args.des, shot_num)

    if 'TimeLLM' in args.model:
        setting += '_{}_llmd{}_llmf{}_tk{}'.format(args.llm_model, args.llm_dim, args.llm_layers, args.top_k)
    if 'RNN' in args.model:
        setting += '_{}_llmd{}_llmf{}'.format(args.rnn_model, args.rnn_dim, args.rnn_layers, )

    exp = Exp(args)  # set experiments
    print('>>>>>>>start few-shot training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.few_shot_train(setting, shot_num=shot_num, path=path)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    path = r'D:\Time-LLM-main\results\Tokyo\sr_TimeLLM_Tokyo_ftM_sl72_ll24_pl24_sd9_dm32_nh8_el2_dl1_df32_fc3_dropout0.1_ebtimeF_test_0_BERT_llmd768_llmf32_tk5'
    args = load_config(os.path.join(path, 'checkpoints', 'configs.pkl'))

    args.model_id = 'fs'
    args.data = 'Sapporo'
    args.data_path = 'Sapporo.csv'

    few_shot_training(args, 64, path)