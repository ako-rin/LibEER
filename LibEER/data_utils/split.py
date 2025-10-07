import numpy as np
from sklearn.preprocessing import StandardScaler
from ..utils.store import save_data
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split
import random

# def train_test_split(data, label, setting):
#     """
#     Provides division of training set and test set under various experimental settings No matter how the experimental
#     settings are, they are all based on trail division, so trail is a basic division unit. For the three typical
#     experimental settings on a dataset, subject-dependent, subject-independent, cross-session,they can be operated
#     based on each subject’s trail, each subject, each session
#           input : all the eeg data and label which can directly be taken as an input
#           output : data and label that make up the training set or test set
#           input shape -> data :   (session, subject, trail, sample, sample_length, time_window, channel, band_feature)
#                          label :  (session, subject, trail, sample, label)
#           output shape -> data :  (sample, sample_length, time_window, channel, band_feature)
#                           label : (sample, label)
#     """
#     train_data = []
#     train_label = []
#     test_data = []
#     test_label = []
#     if setting.experiment_mode == "subject-dependent":
#         # reshape to (sample, sample_length, time_window, channel, band_feature)
#         train_data = [sample for session in data for subject in session for i in setting.train_part for sample in
#                       subject[i - 1]]
#         train_label = [sample for session in label for subject in session for i in setting.train_part for sample in
#                        subject[i - 1]]
#
#         test_part = list(set(range(1, len(data[0][0]) + 1)) - set(setting.train_part))
#
#         test_data = [sample for session in data for subject in session for i in test_part for sample in
#                      subject[i - 1]]
#         test_label = [sample for session in label for subject in session for i in test_part for sample in
#                       subject[i - 1]]
#
#     elif setting.experiment_mode == "subject-independent":
#
#         # reshape to (sample, sample_length, time_window, channel, band_feature)
#         train_data = [sample for session in data for i in setting.train_part for trail in session[i-1]
#                       for sample in trail]
#         train_label = [sample for session in label for i in setting.train_part for trail in session[i-1]
#                        for sample in trail]
#
#         test_part = list(set(range(1, len(data[0]) + 1)) - set(setting.train_part))
#
#         test_data = [sample for session in data for i in test_part for trail in session[i-1] for sample in trail]
#         test_label = [sample for session in label for i in test_part for trail in session[i-1] for sample in trail]
#
#     elif setting.experiment_mode == "cross-session":
#
#         # reshape to (sample, sample_length, time_window, channel, band_feature)
#         train_data = [sample for i in setting.train_part for session in data[i-1] for subject in session for trail in
#                       subject for sample in trail]
#         train_label = [sample for i in setting.train_part for session in label[i-1] for subject in session for trail in
#                        subject for sample in trail]
#
#         test_part = list(set(range(1, len(data) + 1)) - set(setting.train_part))
#
#         test_data = [sample for i in test_part for subject in data[i-1] for trail in subject for sample in trail]
#         test_label = [sample for i in test_part for subject in label[i-1] for trail in subject for sample in trail]
#
#     train_data = np.asarray(train_data)
#     train_label = np.asarray(train_label)
#     test_data = np.asarray(test_data)
#     test_label = np.asarray(test_label)
#     if setting.normalize:
#         for i in range(len(train_data[0][0])):
#             scaler = StandardScaler()
#             train_data[:, :, i] = scaler.fit_transform(train_data[:, :, i])
#             test_data[:, :, i] = scaler.transform(test_data[:, :, i])
#     # if setting.save_data:
#     #     save_data(train_data, train_label, test_data, test_label)
#     return train_data, train_label, test_data, test_label

def index_to_data(data, label, train_indexes, test_indexes, val_indexes, keep_dim=False):
    """
    将 get_split_index 产生的“基于 part 的索引”映射为具体的训练/验证/测试数据与标签。

    Args:
        data: “part 级”数据结构，通常来自 merge_to_part 的输出。
        label: 与 data 对齐的“part 级”标签结构。
        train_indexes (List[int]): 训练集使用的 part 索引列表。
        test_indexes (List[int]): 测试集使用的 part 索引列表。
        val_indexes (List[int]): 验证集使用的 part 索引列表；若为 [-1] 表示当前划分无验证集。
        keep_dim (bool):
            - False（默认）：将所选 part 内部的数据“打平”到样本级，并返回 numpy.ndarray；
            - True：保持原有层级（不打平），直接按索引挑选并返回列表结构（便于保留 trial/subject 维度）。

    Returns:
        Tuple[train_data, train_label, val_data, val_label, test_data, test_label]
            - 当 keep_dim=False：返回的 *_data 与 *_label 为 numpy 数组（打平后的样本集合）。
            - 当 keep_dim=True：返回的为列表，结构与 data/label 的对应维度保持一致（不打平）。

    Notes:
        - 当 val_indexes 为 [-1] 时，表示无验证集；此时返回的 val_data/val_label 为空数组（keep_dim=False）或空列表（keep_dim=True）。
        - “part”的粒度由 merge_to_part 决定（trial/subject/session 等），本函数不改变 part 的定义，只负责索引映射与可选打平。
        - merge_to_part 会先把数据按 part（trial/subject/session，依实验模式而定）组织起来。
        - index_to_data 在 keep_dim=False 时，会把这些选中的 part 里的样本全部取出来，合成形如 (N, ...) 的 numpy 数组；这就叫“打平”，即取消了每个 part 的"分界线"。
        - keep_dim=True 则不打平，保留原来的层级，比如返回 [part0的样本列表, part1的样本列表, ...] 这样的列表结构。

    """
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    if keep_dim:
        for train_index in train_indexes:
            train_data.append(data[train_index])
            train_label.append(label[train_index])
        for test_index in test_indexes:
            test_data.append(data[test_index])
            test_label.append(label[test_index])
        if val_indexes[0] != -1:
            for val_index in val_indexes:
                val_data.append(data[val_index])
                val_label.append(label[val_index])
    else:
        for train_index in train_indexes:
            train_data.extend(data[train_index])
            train_label.extend(label[train_index])
        for test_index in test_indexes:
            test_data.extend(data[test_index])
            test_label.extend(label[test_index])
        if val_indexes[0] != -1:
            for val_index in val_indexes:
                val_data.extend(data[val_index])
                val_label.extend(label[val_index])
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_label = np.array(train_label)
        test_label = np.array(test_label)
        val_data = np.array(val_data)
        val_label = np.array(val_label)
    return train_data, train_label, val_data, val_label, test_data, test_label


def get_split_index(data, label, setting=None):
    """
    根据设置生成基于“part”的训练/验证/测试索引列表，从 merge_to_part 的输出里取数据。

    支持的分割策略（setting.split_type）：
        - 'kfold': K 折交叉验证。按 label 长度做 KFold，是否打乱由 fold_shuffle 控制；
                   当 fold_shuffle 为字符串 'true' 或 'True' 时启用打乱，并使用 setting.seed 作为 random_state。
        - 'leave-one-out': 留一法。每次留 1 个 part 作为测试，其余为训练。
        - 'front-back': 前后切分。前 setting.front 个 part 为训练，其余为测试；要求 front < len(label)。
        - 'early-stop': 早停划分。产生一次划分，含 train/val/test 三组：
            * subject-dependent: 近似分层（按每个 part 的首个标签值聚类），
              对每个标签组内按比例分配 test/val/train，其余样本再按整体比例补齐。
            * 其他实验模式: 直接对 part 索引整体随机打乱后按比例切分。

    二次选择（setting.sr）：
        - 若设置了 setting.sr（1-based），会在生成的多个 rounds 中选取这些轮次对应的索引子集。

    参数:
        data: 与 label 对齐的“part 级”数据列表（只用长度，不参与划分计算）。
        label: “part 级”标签列表；长度决定可划分的 part 数。
        setting: Setting 配置对象，需包含 split_type、fold_num、fold_shuffle、seed、front、test_size、val_size、sr 等。

    返回:
        tts: dict，包含：
            - 'train': List[List[int]]
            - 'test':  List[List[int]]
            - 'val':   List[List[int]]（若某些策略不产生验证集，则对应位置为 [-1] 以占位）

    备注:
        - 本函数返回的是“基于 part 的索引”，需配合 index_to_data 将索引映射为具体样本集合。
        - subject-dependent 下的 early-stop 会做标签组内的随机划分，尽量保持各组比例，
          但由于取整可能产生余数，会在 others 中再按全局比例补齐。
    """
    tts = {}
    if setting.split_type == "kfold":
        kf = KFold(setting.fold_num, shuffle=True if setting.fold_shuffle == 'true' or setting.fold_shuffle == 'True' else False,
                   random_state=setting.seed if setting.fold_shuffle == 'true' else None)
        tts['train'] = [list(train_index) for train_index, _ in kf.split(label)]
        tts['test'] = [list(test_index) for _, test_index in kf.split(label)]
    elif setting.split_type == "leave-one-out":
        loo = LeaveOneOut()
        tts['train'] = [list(train_index) for train_index, _ in loo.split(label)]
        tts['test'] = [list(test_index) for _, test_index in loo.split(label)]
    elif setting.split_type == "front-back":
        if setting.front >= len(label):
            print(f"using front-back split type and {setting.experiment_mode} experiment mode")
            print(f"front size {setting.front} > split part num {len(label)}")
            print("please check your experiment mode or split type")
            exit(1)
        tts['train'] = [[i for i in range(setting.front)]]
        tts['test'] = [[setting.front + i for i in range(len(label) - setting.front)]]
    elif setting.split_type == "early-stop":
        if setting.experiment_mode == "subject-dependent":
            # data need to be split balanced
            # input data : [[not-repetitive] * trails], label : [[repetitive] * trails]
            # output : split index
            tts['test'] = [[]]
            tts['train'] = [[]]
            tts['val'] = [[]]
            groups = {}
            for index, value in enumerate(label):
                if isinstance(value[0], np.ndarray):
                    value_key = tuple(value[0])
                else:
                    value_key = value[0]
                if value_key in groups:
                    groups[value_key].append(index)
                else:
                    groups[value_key] = [index]
            # print(groups)
            others = []
            for indexes in groups.values():
                random.shuffle(indexes)
                total_length = len(indexes)
                test_num = int(setting.test_size * total_length)
                val_num = int(setting.val_size * total_length)
                train_num = int((1-setting.test_size-setting.val_size)*total_length)
                tts['test'][0].extend(indexes[:test_num])
                tts['val'][0].extend(indexes[test_num:test_num+val_num])
                tts['train'][0].extend(indexes[test_num+val_num:test_num+val_num+train_num])
                others.extend(indexes[test_num+val_num+train_num:])
            if len(others) != 0:
                random.shuffle(others)
                expect_test_num = int(len(label) * setting.test_size)
                expect_val_num = int(len(label) * setting.val_size)
                test_num = expect_test_num - len(tts['test'][0])
                val_num = expect_val_num - len(tts['val'][0])
                tts['test'][0].extend(others[:test_num])
                tts['val'][0].extend(others[test_num:test_num+val_num])
                tts['train'][0].extend(others[test_num+val_num:])
        else:
            tts['test'] = [[]]
            tts['train'] = [[]]
            tts['val'] = [[]]
            indexes = [i for i in range(len(label))]
            random.shuffle(indexes)
            total_length = len(indexes)
            test_num = int(setting.test_size * total_length)
            val_num = int(setting.val_size * total_length)
            train_num = total_length - test_num - val_num
            tts['test'][0].extend(indexes[:test_num])
            tts['val'][0].extend(indexes[test_num:test_num + val_num])
            tts['train'][0].extend(indexes[test_num + val_num:])
    else:
        print("wrong split type, please check out")
        exit(1)
    assert setting.sr is None or (max(setting.sr)<=len(label) and min(setting.sr) > 0), \
        "secondary rounds out of limit or secondary rounds set less than 0"
    if setting.sr is not None:
        tts['train'] = [tts['train'][i-1] for i in setting.sr]
        tts['test'] = [tts['test'][i-1] for i in setting.sr]
        if 'val' in tts:
            tts['val'] = [tts['val'][i-1] for i in setting.sr]
    if 'val' not in tts:
        tts['val'] = [[-1] for _ in tts['train']]
    return tts


def merge_to_part(data, label, setting=None):
    """
    按实验模式把原始层级 [session][subject][trial][sample] 重组为“按 part 组织”的列表，
    以便后续基于 part 做训练/验证/测试划分，避免同一 part 被拆到不同集合导致的数据泄漏。

    Args:
        data: 形如 data[session][subject][trial][sample, ...] 的嵌套列表/数组。
        label: 与 data 对齐的标签层级，label[session][subject][trial][sample, ...]。
        setting: 配置对象，关键字段：
            - experiment_mode: 'subject-dependent' | 'subject-independent' | 'cross-session'
            - cross_trail: 'true' | 'false'（仅在 subject-dependent 下生效）
            - sessions: 选用的会话编号，1-based；None 表示使用全部会话
            - pr: 可选，primary rounds 过滤索引（1-based），用于选择部分 part

    Returns:
        (m_data, m_label):
            - subject-dependent 且 cross_trail='true' 时：
                m_data -> [session×subject][trial][sample,...]（trial 保留为独立单元）
                m_label -> 与 m_data 同结构
            - subject-dependent 且 cross_trail='false' 时：
                m_data -> [subject][sample组...]（将该被试的所有 trial 打平成样本组）
                m_label -> 与 m_data 同结构
            - subject-independent 时：
                m_data -> [[subject][sample...]]（外层 1 个分组，内层按被试聚合跨会话样本）
                m_label -> 与 m_data 同结构
            - cross-session 时：
                m_data -> [[session][sample...]]（外层 1 个分组，内层按会话聚合样本）
                m_label -> 与 m_data 同结构

    Notes:
        - 本函数只做“重组/聚合”，不执行具体的随机划分；后续通过 get_split_index 等函数基于 part 划分。
        - sessions 是 1-based 输入，这里会转换为 0-based 索引。
        - 为了保持索引连续紧凑，subject-dependent + cross_trail='true' 使用枚举到的会话索引 s_idx 构造桶编号：
          k = s_idx * num_subjects + subject_idx。
    """
    assert setting.sessions is None or (max(setting.sessions)<=len(label) and min(setting.sessions) >= 0), \
        "sessions set fault, session not exist in dataset"
    # 1) 选择会话；外部传入的 sessions 为 1-based，这里统一转为 0-based
    if setting.sessions is None:
        sessions = range(len(data))
    else:
        sessions = [i - 1 for i in setting.sessions]
    m_data = []
    m_label = []
    if setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'true':

        # 2-a) 被试内 + 保留 trial 粒度：为每个“会话×被试”建立一个桶，trial 作为独立单元追加
        # 使用 s_idx（枚举会话索引）保证桶索引连续：k = s_idx * subjects_per_session + subject_idx
        m_data = [[] for _ in range(len(data[0]) * len(sessions))]  # 将要处理的数据：session x 被试
        m_label = [[] for _ in range(len(data[0]) * len(sessions))]
        
        # 将每个 trial 放进对应的桶中
        for s_idx, i in enumerate(sessions):
            for idx1, subject in enumerate(data[i]):
                for idx2, trail in enumerate(subject):
                    m_data[s_idx * len(data[i]) + idx1].append(trail)
        for s_idx, i in enumerate(sessions):
            for idx1, subject in enumerate(label[i]):
                for idx2, trail in enumerate(subject):
                    m_label[s_idx * len(data[i]) + idx1].append(trail)
    elif setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'false':
        # 2-b) 被试内 + 不保留 trial 粒度：同一被试跨会话的所有 trial 样本合并到同一桶（按样本级）
        m_data = [[] for _ in range(len(data[0]))]
        m_label = [[] for _ in range(len(data[0]))]
        for s_idx, i in enumerate(sessions):
            for idx1, subject in enumerate(data[i]):
                for idx2, trail in enumerate(subject):
                    for sample in trail:
                        m_data[idx1].append([sample])
        for s_idx, i in enumerate(sessions):
            for idx1, subject in enumerate(label[i]):
                for idx2, trail in enumerate(subject):
                    for sample in trail:
                        m_label[idx1].append([sample])
    elif setting.experiment_mode == "subject-independent":
        # 2-c) 跨被试：外层单组，内层按被试聚合（合并跨会话的样本）
        m_data = [[[] for _ in range(len(data[0]))]]
        m_label = [[[] for _ in range(len(data[0]))]]
        for i in sessions:
            for idx, subject in enumerate(data[i]):
                for trail in subject:
                    m_data[0][idx].extend(trail)
        for i in sessions:
            for idx, subject in enumerate(label[i]):
                for trail in subject:
                    m_label[0][idx].extend(trail)
    elif setting.experiment_mode == "cross-session":
        # 2-d) 跨会话：外层单组，内层按“会话”聚合样本；使用 s_idx 保持桶索引连续
        m_data = [[[] for _ in range(len(sessions))]]
        m_label = [[[] for _ in range(len(sessions))]]
        for s_idx, i in enumerate(sessions):
            for subject in data[i]:
                for trail in subject:
                    m_data[0][s_idx].extend(trail)
        for s_idx, i in enumerate(sessions):
            for subject in label[i]:
                for trail in subject:
                    m_label[0][s_idx].extend(trail)
    # 3) 可选：根据 primary rounds (pr, 1-based) 选择部分 part
    assert setting.pr is None or (max(setting.pr)<=len(m_label) and min(setting.pr) > 0), \
        "primary rounds out of limit or primary rounds set less than 0"
    if setting.pr is not None:
        m_data = [m_data[i-1] for i in setting.pr]
        m_label = [m_label[i-1] for i in setting.pr]
    return m_data, m_label
