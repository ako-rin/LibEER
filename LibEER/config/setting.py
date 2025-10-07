
class Setting:
    """
    训练/数据配置对象：集中管理数据集、预处理、划分与实验模式等参数。

    Args:
        dataset (str): 数据集标识，例如 'seed', 'seedv_de', 'deap', 'dreamer', 'hci' 等。

        dataset_path (str): 数据集根目录的绝对/相对路径。

        pass_band (list[float, float] | None): 带通滤波的下/上截止频率 [low, high]（单位 Hz）。若不需要带通或使用已提取特征，可设为 None。

        extract_bands (list[list[float, float]] | None): 要提取的频带范围列表，例如 [[4, 7], [8, 13], [14, 30]]。不同数据集/特征类型可能有默认值；传 None 表示走默认或忽略。

        time_window (int | float): 预处理/特征提取的窗口大小。单位依数据而定：
            - 原始 EEG（按采样点处理）时表示采样点数；
            - 已对齐到秒/帧的场景（如部分 DE/DE_LDS）可用秒或步长表示。

        overlap (int | float): 相邻窗口的重叠长度，单位与 time_window 保持一致（点数/秒/步）。

        sample_length (int): 输入到模型的一次序列长度（时间步/片段数）。

        stride (int): 滑窗抽样步长（窗口移动的步数/点数）。

        seed (int): 随机种子，保证结果可复现。

        feature_type (str): 特征类型，常见有 'raw', 'de', 'de_lds' 等。不同类型会影响是否需要提取频带、窗口单位含义等。

        only_seg (bool, optional): 仅做分段/缓存而不进入训练流程（若上游支持）。默认 False。

        cross_trail (bool | str, optional): 是否采用“跨 trial”的划分/采样策略。部分数据管线允许传布尔或字符串 'true'/'false'。默认 'true'。

        experiment_mode (str, optional): 实验模式：'subject-dependent'（被试内）、'subject-independent'（跨被试）、'cross-session'（跨会话）。默认 'subject-dependent'。

        train_part (Any, optional): 可选的训练子集/分片指定（若有数据接口支持）。

        eog_clean (bool, optional): 是否进行眼动/眼电（EOG）相关的伪迹去除或抑制。默认 True。

        metrics (list[str] | None, optional): 评估指标列表（若管线支持自定义度量）。

        normalize (bool, optional): 是否对数据/特征进行归一化。默认 False。

        save_data (bool, optional): 是否缓存/保存中间数据（如分段结果或特征）。默认 True。

        split_type (str, optional): 数据划分方式。可选：'kfold'、'leave-one-out'、'front-back'、'early-stop' 等。默认 'kfold'。

        fold_num (int, optional): K 折交叉验证的折数。默认 5。

        fold_shuffle (bool, optional): K 折划分前是否打乱顺序。默认 True。

        front (int, optional): 在 'front-back' 策略下，前多少个 trial 作为训练，其余作为测试。默认 9。

        test_size (float, optional): 在早停/随机划分中，测试集比例。默认 0.2。

        val_size (float, optional): 在早停/随机划分中，验证集比例。默认 0.2。

        sessions (list[int] | None, optional): 会话选择（跨会话实验时指定使用哪些会话）。None 表示默认策略。

        pr (Any | None, optional): 预处理/数据接口的可选参数（例如预处理相关的比率/标识等），保持 None 使用默认。

        sr (Any | None, optional): 采样率/重采样率等可选参数（具体取决于数据加载实现），保持 None 使用默认。

        bounds (list[float] | list[float, float] | None, optional): 连续标签数据集（如 DEAP、DREAMER）离散化的阈值；例如：
            - 二分类单标签: [5]
            - 二分类双标签（valence + arousal）: [5, 5]
            - 三分类: [low_thr, high_thr]（如 [3, 6]）
            SEED/SEED-V 等离散标签数据集会忽略该参数。

        onehot (bool, optional): 是否将标签转换为 one-hot 编码。默认 False。

        label_used (list[str] | None, optional): 指定使用哪些标签维度（如 ['valence'], ['valence','arousal']）。连续标签数据集常用；离散标签数据集可忽略。

    Notes:
        - 本类仅承载配置，不主动执行逻辑；不同数据集/管线会读取并解释这些字段。
        - 各参数的可用性取决于你选择的数据集与特征类型，未被使用的字段可以保留默认值。
    """
    def __init__(self, dataset, dataset_path, pass_band, extract_bands, time_window, overlap, sample_length, stride, seed,
                 feature_type, only_seg=False, cross_trail='true', experiment_mode="subject-dependent", train_part=None, eog_clean=True,
                 metrics=None, normalize=False, save_data=True, split_type="kfold", fold_num=5, fold_shuffle=True, front=9, test_size=0.2, val_size = 0.2, sessions=None, pr=None, sr=None, bounds=None,
                 onehot=False, label_used=None):
        # random seed
        self.seed = seed

        # dataset setting

        self.dataset = dataset
        self.dataset_path = dataset_path

        # preprocess setting

        # Data at indices 0 and 1 represent the lower and higher thresholds of bandpass filtering
        self.pass_band = pass_band
        # Two-dimensional array, with each element at an index representing the range of each frequency band
        self.extract_bands = extract_bands if extract_bands is None else extract_bands
        # The size of the time window during preprocessing, in num of data points
        self.time_window = time_window
        # the length of overlap for each preprocessing window
        self.overlap = overlap
        # The length of sample sequences input to the model at once
        self.sample_length = sample_length
        # the stride of a sliding window for data extraction
        self.stride = stride
        # Feature type of EEG signals
        self.feature_type = feature_type
        # whether remove the eye movement interference
        self.eog_clean = eog_clean
        # whether normalize
        self.normalize = normalize
        # whether save_data
        self.save_data = save_data

        self.only_seg = only_seg

        # train_test_setting

        # whether use cross trial setting
        self.cross_trail = cross_trail
        # subject-dependent or subject-independent or cross-session
        self.experiment_mode = experiment_mode
        # how to partition a dataset
        self.split_type = split_type
        # according to the split type, choose which part is used as the training set or testing set
        self.fold_num = fold_num
        self.fold_shuffle = fold_shuffle
        self.front = front
        self.test_size = test_size
        self.val_size = val_size
        self.sessions = sessions
        self.pr = pr
        self.sr = sr
        
        # 连续标签数据集的离散化阈值；SEED/SEED-V 等离散标签将忽略
        self.bounds = bounds
        # 是否将标签转换为 one-hot 表示
        self.onehot = onehot
        # 指定使用的标签维度名称（如 ['valence']）；离散标签数据集可忽略
        self.label_used = label_used



def set_setting_by_args(args):
    if args.dataset_path is None:
        print("Please set the dataset path")
    if args.dataset is None:
        print("Please select the dataset to train")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, cross_trail=args.cross_trail, experiment_mode=args.experiment_mode,
                   metrics=args.metrics, normalize=args.normalize, split_type=args.split_type, fold_num=args.fold_num,
                   fold_shuffle=args.fold_shuffle, front=args.front, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   bounds=args.bounds, onehot=args.onehot, label_used=args.label_used)


def seed_sub_dependent_front_back_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED subject dependent experiment mode,\n"
          "the first 9 trails for each subject were used as a training set and the last 6 as a test set")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='front-back', front=9, sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)

def seed_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Seed subject dependent train val test experiment mode, \n"
          "For each subject, nine random trails were used as training set, three random trails were used as verification"
          " set, last three trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)
def seediv_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('seediv'):
        print('not using SEED IV dataset, please check your setting')
        exit(1)
    print("Using SeedIV subject dependent early stopping experiment mode, \n"
          "For each subject, sixteen random trails were used as training set, four random trails were used as verification"
          " set, last four trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)

def seed_sub_dependent_5fold_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED subject dependent experiment mode,\n"
          "Using a 5-fold cross-validation, three test sets are grouped in the Order of trail")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, cross_trail=args.cross_trail, experiment_mode="subject-dependent",
                   normalize=args.normalize, split_type='kfold', fold_num=5, fold_shuffle=False, sessions=args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)


def seed_sub_independent_leave_one_out_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED subject independent early stopping experiment mode,\n"
          "Using the leave one out method, all samples of 15 trails for 1 subject were split "
          "into all samples as a test set, and all samples of 15 trails for 14 other round "
          "were split into all samples as a training set, cycle 15 times to report average results")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='leave-one-out', sessions=[1] if args.sessions is None else args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)

def seed_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using early stopping SEED subject independent experiment mode,\n"
          "The random nine subjects' data are taken as training set, random three subjects' data are taken as "
          "validation set, random three subject's data are taken as test set. We choose the best results in validation set,"
          "and test it in test set"
          )
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=[1] if args.sessions is None else args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)

def hci_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('hci'):
        print('not using Hci dataset, please check your setting')
        exit(1)
    print("Using hci subject dependent early stopping experiment mode, \n"
          ""
          )
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot, bounds=args.bounds,
                   label_used=args.label_used)

def seediv_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('seediv'):
        print('not using SEED IV dataset, please check your setting')
        exit(1)
    print("Using SeedIV subject dependent early stopping experiment mode, \n"
          "For each subject, sixteen random trails were used as training set, four random trails were used as verification"
          " set, last four trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2,
                   sessions=[1] if args.sessions is None else args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)

def deap_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using deap dataset, please check your setting')
        exit(1)
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot, bounds=args.bounds,
                   label_used=args.label_used)
def hci_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('hci'):
        print('not using Hci dataset, please check your setting')
        exit(1)
    print("Using hci subject dependent early stopping experiment mode, \n"
          ""
          )
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-independent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot, bounds=args.bounds,
                   label_used=args.label_used)
def deap_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using deap dataset, please check your setting')
        exit(1)
    print("Using deap subject dependent early stopping experiment mode")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="subject-dependent", normalize=args.normalize,
                   split_type='early-stop', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot,bounds=args.bounds,
                   label_used=args.label_used)



def seed_cross_session_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED cross session experiment mode,\n"
          "Three sessions of data, one as the test dataset")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="cross-session", normalize=args.normalize,
                   split_type='leave-one-out', sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)

def deap_sub_independent_leave_one_out_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using DEAP dataset, please check your setting')
        exit(1)
    print("Using Default DEAP sub independent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 10], [8, 12], [13, 30], [30, 47]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="subject-independent",
                   normalize=args.normalize, split_type='leave-one-out', pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

def deap_sub_dependent_10fold_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using DEAP dataset, please check your setting')
        exit(1)
    print("Using Default DEAP sub independent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 10], [8, 12], [13, 30], [30, 47]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="subject-dependent",
                   normalize=args.normalize, cross_trail=args.cross_trail, split_type='kfold', fold_num=10, pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

def dreamer_sub_independent_setting(args):
    if not args.dataset.startswith('dreamer'):
        print('not using Dreamer dataset, please check your setting')
        exit(1)
    print("Using Default Dreamer sub independent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 13], [14, 30]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="subject-independent",
                   normalize=args.normalize, split_type='leave-one-out', pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

def dreamer_sub_dependent_setting(args):
    if not args.dataset.startswith('dreamer'):
        print('not using Dreamer dataset, please check your setting')
        exit(1)
    print("Using Default Dreamer sub dependent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 13], [14, 30]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="subject-dependent",
                   normalize=args.normalize, cross_trail=args.cross_trail, split_type='leave-one-out', pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

preset_setting = {
    # 记得改一下格式
    "seed_sub_dependent_train_val_test_setting": seed_sub_dependent_train_val_test_setting,
    "seediv_sub_dependent_train_val_test_setting": seediv_sub_dependent_train_val_test_setting,
    "seed_sub_independent_train_val_test_setting": seed_sub_independent_train_val_test_setting,
    "seediv_sub_independent_train_val_test_setting": seediv_sub_independent_train_val_test_setting,
    "deap_sub_dependent_train_val_test_setting" : deap_sub_dependent_train_val_test_setting,
    "hci_sub_dependent_train_val_test_setting" : hci_sub_dependent_train_val_test_setting,
    "deap_sub_independent_train_val_test_setting" : deap_sub_independent_train_val_test_setting,
    "hci_sub_independent_train_val_test_setting" : hci_sub_independent_train_val_test_setting,
    # ***********************************************************************
    "seed_sub_dependent_5fold_setting": seed_sub_dependent_5fold_setting,
    "seed_sub_dependent_front_back_setting": seed_sub_dependent_front_back_setting,
    "seed_sub_independent_leave_one_out_setting": seed_sub_independent_leave_one_out_setting,
    "seed_cross_session_setting": seed_cross_session_setting,
    "deap_sub_independent_leave_one_out_setting": deap_sub_independent_leave_one_out_setting,
    "deap_sub_dependent_10fold_setting": deap_sub_dependent_10fold_setting,
    "dreamer_sub_independent_setting": dreamer_sub_independent_setting,
    "dreamer_sub_dependent_setting": dreamer_sub_dependent_setting,

    None: set_setting_by_args
}
