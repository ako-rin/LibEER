import numpy as np

from .config.setting import Setting, preset_setting, set_setting_by_args
from .models.Models import Model

from .utils.utils import state_log, result_log, setup_seed, sub_result_log
from .utils.args import get_args_parser
from .data_utils.load_data import get_data
from .data_utils.split import merge_to_part, index_to_data, get_split_index
from .utils.store import save_res


def main(args):
    # set the dataset and data preprocess parameters by args
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setup_seed(args.seed)

    # get all data set by setting, obtain the number of channels and feature dim
    data, label, channels, feature_dim, num_classes = get_data(setting)

    # if setting.save_data:

    # merge the data to corresponding part by experiment mode
    data, label = merge_to_part(data, label, setting)
    # get train and test indexes and split type

    best_metrics = []
    subjects_metrics = []
    if setting.experiment_mode == "subject-dependent":
        subjects_metrics = [[]for _ in range(len(data))]
    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)
            if val_indexes[0] == -1:
                print(f"train indexes:{train_indexes}, test indexes:{test_indexes}")
            else:
                print(f"train indexes:{train_indexes}, val indexes:{val_indexes}, test indexes:{test_indexes}")

            # split train and test data by specified experiment mode
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, args.keep_dim)
            # print(len(train_data))
            # model to train
            if args.sample_length == 1 or args.only_seg:
                model = Model[args.model](channels, feature_dim, num_classes)
            else:
                model = Model[args.model](args.sample_length, channels, feature_dim, num_classes)
            # Train one round using the train one round function defined in the model
            round_metric = model.train_one_round(args, ridx, rridx, train_data, train_label, val_data, val_label, test_data, test_label)
            best_metrics.append(round_metric)
            save_res(args, round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx-1].append(round_metric)
    # best metrics: every round metrics dict
    # subjects metrics: (subject, sub_round_metric)
    if setting.experiment_mode == "subject-dependent":
        sub_result_log(args, subjects_metrics)
    else:
        result_log(args, best_metrics)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # log out train state
    state_log(args)
    main(args)
