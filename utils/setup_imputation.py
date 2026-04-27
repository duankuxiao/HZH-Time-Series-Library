from utils.print_args import print_args
from utils.tools import load_content
import torch


def model_hyparameter_setup(args,d_mode=None,e_layers=None):
    args.use_norm = True

    # default
    if args.task_name == 'imputation':
        args.learning_rate = 0.001
        args.patience = 5
        args.train_epochs = 50
    elif args.task_name == 'classification':
        args.learning_rate = 0.01
        args.patience = 5
        args.train_epochs = 50

    else:
        args.learning_rate = 0.0001
    args.d_model = 512 if d_mode is None else d_mode
    args.d_ff = int(4 * args.d_model)
    args.e_layers = 4 if e_layers is None else e_layers
    args.d_layers = 1
    args.factor = 3
    args.moving_avg = 25
    if args.task_name == 'classification':
        args.e_layers = 4 if e_layers is None else e_layers
        args.d_model = 256 if d_mode is None else d_mode  # 128 default
        args.d_ff = int(2 * args.d_model)  # 256 default
        args.top_k = 3
        args.patch_len = 2
        args.stride = 1

    if args.model == 'LLMformer':
        args.train_epochs = 30
        args.batch_size = 8
        if args.task_name == 'imputation':
            args.use_norm = True
            args.learning_rate = 0.001
        else:
            args.learning_rate = 0.001
        args.patience = 6
        args.llm_model = 'BERT'  #GPT2
        args.d_model = 16
        args.d_ff = 64
        args.e_layers = 2
        args.d_layers = 2  # 3
        args.llm_layers = 2  # 6

    if 'RNN' in args.model:
        args.use_norm = False
        args.d_model = 256 if d_mode is None else d_mode
        args.e_layers = 2 if e_layers is None else e_layers

    if args.model == 'Transformer':
        args.use_norm = False
        # args.e_layers = 2
        # args.d_model = 256
        # args.d_ff = 512
        if args.task_name == 'imputation':
            args.d_model = 256
            args.d_ff = 512
        if args.task_name == 'classification':
            args.d_model = 128 if d_mode is None else d_mode
            args.d_ff = int(args.d_model * 2)

    if args.model == 'DLinear':
        args.use_norm = False

    if args.model == 'Informer':
        args.train_epochs = 20
        args.use_norm = False
        args.factor = 5
        args.d_layers = 2  # default 2  imputatiaon 1
        if args.task_name == 'imputation':
            args.d_model = 128 if d_mode is None else d_mode
            args.d_ff = int(args.d_model)
        if args.task_name == 'classification':
            args.d_model = 256 if d_mode is None else d_mode
            args.d_ff = int(args.d_model * 2)
            args.e_layers = 4 if e_layers is None else e_layers

    if args.model == 'Autoformer':
        args.e_layers = 2
        if args.task_name == 'imputation':
            args.d_model = 512
            args.d_ff = 2048
        if args.task_name == 'classification':
            args.d_model = 512 if d_mode is None else d_mode
            args.d_ff = int(args.d_model * 4)
            args.e_layers = 4 if e_layers is None else e_layers

        args.use_norm = False

    if args.model == 'iTransformer':  # default
        args.train_epochs = 20
        args.e_layers = 3 if e_layers is None else e_layers
        args.d_model = 512
        args.d_ff = 512
        if args.task_name == 'imputation':
            # args.use_norm = False
            args.d_model = 512 if d_mode is None else d_mode
            args.d_ff = int(args.d_model)
        if args.task_name == 'classification':
            args.d_model = 256 if d_mode is None else d_mode
            args.d_ff = int(args.d_model)

    if args.model == 'TimeLLM':
        args.llm_model = 'GPT2'
        args.patience = 3
        args.batch_size = 12
        args.learning_rate = 0.01
        args.train_epochs = 20
        args.patience = 3
        args.feature_cols = args.target
        args.top_k = 5
        args.d_model = 16
        args.d_ff = 32
        args.llm_layers = 12

    if args.model == 'PatchTST':
        args.train_epochs = 20
        args.use_norm = True
        args.d_model = 128 if d_mode is None else d_mode
        args.d_ff = int(args.d_model * 2)
        args.e_layers = 2 if e_layers is None else e_layers
        args.n_heads = 8
        if args.task_name == 'imputation':
            args.d_model = 512 if d_mode is None else d_mode
            args.d_ff = int(args.d_model)
            args.e_layers = 4 if e_layers is None else e_layers
        elif args.task_name == 'classification':
            args.e_layers = 4 if e_layers is None else e_layers
            args.d_model = 512 if d_mode is None else d_mode
            args.d_ff = int(args.d_model)

    if args.model == 'TimesNet':
        args.e_layers = 2 if e_layers is None else e_layers
        args.use_norm = True
        args.train_epochs = 10
        if args.task_name == 'imputation':
            args.d_model = 64 if d_mode is None else d_mode  # min{max[2**log(seq_dim),32],512} for forecast   min{max[2**log(seq_dim),64],128} for imputation
            args.d_ff = int(args.d_model)
            args.top_k = 3  # 5 for forecast   3 for imputation, classification, anomaly detection
        elif args.task_name == 'classification':
            args.d_model = 256 if d_mode is None else d_mode
            args.d_ff = int(args.d_model)
            args.e_layers = 4 if e_layers is None else e_layers
            args.top_k = 3
            args.num_kernels = 4
        else:
            args.d_model = 32 if d_mode is None else d_mode  # min{max[2**log(seq_dim),32],512} for forecast   min{max[2**log(seq_dim),64],128} for imputation
            args.d_ff = int(args.d_model)
            args.top_k = 5  # 5 for forecast   3 for imputation, classification, anomaly detection

    if args.model == 'SAITS':
        args.learning_rate = 0.001
        args.e_layers = 2
        args.d_model = 256
        args.d_ff = 128
        args.n_heads = 4
        args.d_v = 64
        args.d_k = 64

    if 'LLM' in args.model:
        args.content = load_content(args)
        if args.llm_model == 'LLAMA8b':
            args.llm_dim = 4096
        elif args.llm_model == 'LLAMA3b':
            args.llm_dim = 3072
        elif args.llm_model == 'LLAMA1b':
            args.llm_dim = 2048
        elif 'BERT' in args.llm_model:
            args.llm_dim = 768
        elif 'GPT2' in args.llm_model:
            args.llm_dim = 768
        else:
            raise ValueError('Unknown llm model')

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    args.inverse = True
    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.feature_cols is not None:
        args.enc_in = len(args.feature_cols)
        args.dec_in = len(args.feature_cols)
    if args.target is not None:
        args.c_out = len(args.target)

    if args.features == 'S':
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1

    print('Args in experiment:')
    print_args(args)
    return args