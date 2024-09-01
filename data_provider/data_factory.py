from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from torch.utils.data import DataLoader
from data_provider.data_loader_cumstom import Dataset_PV_hour,Dataset_solar_radiation
from data_provider.data_loader_LLM import Dataset_PV_hour_llm,Dataset_solar_radiation_llm

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'PV': Dataset_PV_hour,
    'PV7f': Dataset_PV_hour,
    'Tokyo':Dataset_solar_radiation,  # Dataset_solar_radiation
    'Naha':Dataset_solar_radiation,  # Dataset_solar_radiation
    'Sapporo': Dataset_solar_radiation,  # Dataset_solar_radiation
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    if 'TimeLLM' in args.model and args.data == 'PV':
        Data = Dataset_PV_hour_llm  # Dataset_PV_hour_llm
    if 'TimeLLM' in args.model and args.data == 'Tokyo':
        Data = Dataset_solar_radiation_llm  # Dataset_solar_radiation_llm

    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
