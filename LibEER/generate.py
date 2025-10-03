

server = 127
device = 3
model = "GCBNet"
epochs = 150
lr = 0.001
batch = 16
dataset = "seediv_raw"
dataset_path = ""
label_used = "both"
setting = ""
process = "de_lds"
# process = "raw"
# process = "psd"
isIn = False


if dataset.startswith("seediv"):
    if server == 128:
        dataset_path = "/date1/yss/data/SEED数据集/SEED_IV"
    else:
        dataset_path = "/data1/cxx/SEED数据集/SEED_IV"
    if isIn:
        setting = "seediv_sub_independent_train_val_test_setting"
    else:
        setting = "seediv_sub_dependent_train_val_test_setting"
elif dataset.startswith("seed"):
    if server == 128:
        dataset_path = "/date1/yss/data/SEED数据集/SEED"
    else:
        dataset_path = "/data1/cxx/SEED数据集/SEED/"
    if isIn:
        setting = "seed_sub_independent_train_val_test_setting"
    else:
        setting = "seed_sub_dependent_train_val_test_setting"
elif dataset.startswith("deap"):
    if server == 128:
        dataset_path = "/date1/yss/data/DEAP数据集/data_preprocessed_python"
    else:
        dataset_path = "/data1/cxx/DEAP/data_preprocessed_python"
    if isIn:
        setting = ""
    else:
        setting = "deap_sub_dependent_train_val_test_setting"




command = (f"CUDA_VISIBLE_DEVICES={device} nohup python main.py -metrics 'acc' 'macro-f1' -model {model} "
           f"-metric_choose 'macro-f1' -setting {setting} "
           f"-dataset_path {dataset_path} -dataset {dataset} "
           f"-batch_size {batch} -epochs {epochs} -lr {lr}")
if dataset.endswith("raw") or dataset.startswith("deap"):
    if process == "de_lds":
        command += " -time_window 1 -feature_type de_lds"
    elif process == "raw":
        command += " -only_seg"
        if dataset.startswith("seed"):
            command += " -sample_length 200 -stride 200"
        else:
            command += " -sample_length 128 -stride 128"
    elif process == 'psd':
        command += " -time_window 1 -feature_type psd"

if dataset.startswith("seediv"):
    cs = command + f" -seed 2024 >{model}/s4_b{batch}e{epochs}lr{lr}.log\nwait\n"
    # cs = cs + "wait\n" + command + f" >{model}/s4_b{batch}e{epochs}_seed0lr{lr}.log\n wait\n"
elif dataset.startswith("seed"):
    cs = command + f" -seed 2024 >{model}/b{batch}e{epochs}lr{lr}.log\nwait\n"
    # cs = cs + "wait\n" + command + f" >{model}/b{batch}e{epochs}_seed0lr{lr}.log\n wait\n"
elif dataset.startswith("deap"):
    if label_used=="both":
        cs = command + f" -bounds 5 5 -label_used valence arousal -seed 2024 >{model}/deap_{label_used}_b{batch}e{epochs}lr{lr}.log\n"
        cs = (cs + "wait\n")
              # + command + f" -bounds 5 5 -label_used valence arousal >{model}/deap_{label_used}_b{batch}e{epochs}_seed0lr{lr}.log\n wait\n")
    else:
        cs = command + f" -bounds 5 5 -label_used {label_used} -seed 2024 >{model}/deap_{label_used}_b{batch}e{epochs}lr{lr}.log\n"
        cs = (cs + "wait\n")
              # + command + f" -bounds 5 5 -label_used {label_used} >{model}/deap_{label_used}_b{batch}e{epochs}_seed0lr{lr}.log\n wait\n")


print(cs)
