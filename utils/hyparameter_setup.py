def model_hyparameter_setup(args):
    if args.model == 'TimeLLMformer':
        args.lradj = 'PEMS'
        args.llm_model = 'BERT'
        args.d_model = 32
        args.d_ff = 64
        args.e_layers = 4
        args.d_layers = 4
        args.llm_layers = 6

    if 'RNN' in args.model:
        args.rnn_dim = 512
        args.rnn_layers = 2

    if args.model == 'DLinear':
        args.moving_avg = 25

    if args.model == 'Informer':  # default
        args.factor = 5
        args.e_layers = 4
        args.d_layers = 2

    if args.model == 'Autoformer':  # default
        args.dropout = 0.05

    if args.model == 'TimeLLM':  # default
        args.feature_cols = args.target
        args.top_k = 5
        args.d_model = 16
        args.d_ff = 64
        args.llm_layers = 32

    if args.model == 'PatchTST':  # default
        args.dropout = 0.2
        args.d_model = 128  # for small dataset 16
        args.d_ff = 256  # for small dataset 128
        args.e_layers = 3
        args.d_layers = 1
        args.n_heads = 16  # for small dataset 4

    if args.model == 'TimesNet':
        args.learning_rate = 0.001  # 0.001 for imputation 0.0001 for forecast
        args.d_model = 64  # 64 for imputation 32 for forecast
        args.d_ff = 4 * args.d_model
        args.e_layers = 2
        args.top_k = 5

    if args.model == 'SAITS':
        args.e_layers = 2
        args.d_model = 256
        args.d_ff = 128
        args.n_heads = 4
        args.d_v = 64
        args.d_k = 64
    return args