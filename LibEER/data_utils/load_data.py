import fnmatch
import json
import os
import pickle

from scipy.io import loadmat
import numpy as np
import multiprocessing as mp
from functools import partial
import mne
import xmltodict
import warnings

from ..data_utils.preprocess import preprocess, label_process, lds


def get_data(setting=None):
    if setting is None:
        print(f"Error: Setting not set")

    # obtain data in the uniform formats, which load dataset and integrate into (session, subject, trail) format
    data, baseline, label, sample_rate, channels = get_uniform_data(setting.dataset, setting.dataset_path)
    # preprocess the eeg signal
    all_data, feature_dim = preprocess(data=data, baseline=baseline, sample_rate=sample_rate,
                                     pass_band=setting.pass_band, extract_bands=setting.extract_bands,
                                     sample_length=setting.sample_length, stride=setting.stride
                                     , time_window=setting.time_window, overlap=setting.overlap,
                                     # 这里进行判断，若数据集是经过特征提取了的，就设定为 True
                                     only_seg=setting.only_seg if setting.dataset not in extract_dataset else True,
                                     
                                     feature_type=setting.feature_type,
                                     eog_clean=setting.eog_clean)
    # bounds： 阈值设置，当二分类（低/高）双标签（valence + arousal）：设为 [5,5]，分别作为两个维度的阈值。
    all_data, all_label, num_classes = label_process(data=all_data, label=label, bounds=setting.bounds, onehot=setting.onehot, label_used=setting.label_used)
    return all_data, all_label, channels, feature_dim, num_classes


available_dataset = [
    "seed_raw", "seediv_raw", "deap", "deap_raw", "hci", "dreamer", "seed_de", "seed_de_lds", "seed_psd", "seed_psd_lds", "seed_dasm", "seed_dasm_lds"
    , "seed_rasm", "seed_rasm_lds", "seed_asm", "seed_asm_lds", "seed_dcau", "seed_dcau_lds", "seediv_de_lds", "seediv_de_movingAve",
    "seediv_psd_movingAve", "seediv_psd_lds",
    # SEED-V features (DE)
    "seedv_de", "seedv_de_lds",
    # SEED-V raw
    "seedv_raw"
]

extract_dataset = {
    "seed_de", "seed_de_lds", "seed_psd", "seed_psd_lds", "seed_dasm", "seed_dasm_lds",
    "seed_rasm", "seed_rasm_lds", "seed_asm", "seed_asm_lds", "seed_dcau", "seed_dcau_lds",
    "seediv_de_lds", "seediv_de_movingAve", "seediv_psd_movingAve", "seediv_psd_lds",
    # SEED-V features treated as extracted features
    "seedv_de", "seedv_de_lds"
}

def get_uniform_data(dataset, dataset_path):
    """
    Mainly aimed at the structure of different datasets,
    it is divided into the form of (session, subject, trail, channel, raw_data).
    :param dataset: the dataset used to train
    :param dataset_path: the dir of the dataset location
    :return: data, baseline, label, and sample rate of the original dataset
    """
    func = {
        "seed_raw": read_seed_raw,
        "deap": read_deap_preprocessed,
        "dreamer": read_dreamer,
        "deap_raw": read_deap_raw,
        "seediv_raw": read_seedIV_raw,
        "hci": read_hci
    }
    # SEED-V first to avoid falling into generic 'seed' branch
    if dataset.startswith("seedv"):
        if dataset == "seedv_de" or dataset == "seedv_de_lds":
            data, baseline, label, sample_rate, channels = read_seedV_feature(dataset_path, feature_type=dataset[6:])
        elif dataset == "seedv_raw":
            data, baseline, label, sample_rate, channels = read_seedV_raw(dataset_path)
        else:
            raise ValueError(f"Unknown SEED-V variant '{dataset}'. Supported: ['seedv_de', 'seedv_de_lds', 'seedv_raw']")
    elif dataset.startswith("seediv") and dataset != "seediv_raw":
        data, baseline, label, sample_rate, channels = read_seedIV_feature(dataset_path, feature_type=dataset[7:])
    elif dataset.startswith("seed") and not dataset.startswith(("seediv", "seedv")) and dataset != "seed_raw":
        # call the read_seed_feature function when using the feature provided by seed official
        data, baseline, label, sample_rate, channels = read_seed_feature(dataset_path, feature_type=dataset[5:])
    else:
        data, baseline, label, sample_rate, channels = func[dataset](dataset_path)
    return data, baseline, label, sample_rate, channels


def read_seed_raw(dir_path):
    # input : 45 files(3 sessions, 15 round) containing all 15 trails with a sampling rate of 200 Hz
    # output : EEG signal with a trail as the basic unit and sample rate of the original dataset
    # output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label)

    # Extract the EEG data of each subject from the SEED dataset, and partition the data of each session
    dir_path += "/Preprocessed_EEG"
    eeg_files = [['1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
                  '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
                  '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
                  '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
                  '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],
                 ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
                  '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
                  '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
                  '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
                  '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],
                 ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
                  '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
                  '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
                  '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
                  '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']
                 ]
    # Extract the label for all trail in three sessions
    label = np.array(loadmat(f"{dir_path}/label.mat")['label'])
    labels = np.tile(label[0]+1, (3, 15, 1))

    # create the empty list of (3, 15, 15) => (session, subject, trail)
    eeg_data = [[[[] for _ in range(15)] for _ in range(15)] for _ in range(3)]
    # Loop processing of EEG mat files
    for session_files, session_id in zip(eeg_files, range(3)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            eeg_data[session_id] = pool.map(
                partial(parallel_read_seed_raw, dir_path), eeg_files[session_id])

    return eeg_data, None, labels, 200, 62

def parallel_read_seed_raw(dir_path, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    label_datas = []
    for i in range(15):
        trail_data = subject_data[keys[i]]
        trail_datas.append(trail_data[:,1:])
    return trail_datas

def read_seed_feature(dir_path, feature_type="de"):
    # input : 45 files(3 sessions, 15 round) containing all 15 trails with a sampling rate of 200 Hz
    # output : EEG signal with a trail as the basic unit
    # output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label),

    # Extract the EEG data of each subject from the SEED dataset, and partition the data of each session
    dir_path += "/ExtractedFeatures"
    eeg_files = [['1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
                  '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
                  '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
                  '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
                  '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],
                 ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
                  '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
                  '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
                  '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
                  '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],
                 ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
                  '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
                  '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
                  '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
                  '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']
                 ]
    feature_index = {
        "de": 0, "de_lds": 1, "psd": 2, "psd_lds": 3, "dasm": 4, "dasm_lds": 5,
        "rasm": 6, "rasm_lds": 7, "asm": 8, "asm_lds": 9, "dcau": 10, "dcau_lds": 11
    }

    # Extract the label for all trail in three sessions, label shape : (15)
    label = np.array(loadmat(f"{dir_path}/label.mat")['label'])
    label = np.tile(label[0] + 1, (3, 15, 1))

    # Set index based on selected characteristics
    fi = feature_index[feature_type]

    eeg_data = [[] for _ in range(3)]
    # Define a function to read a single MAT file
    for session_files, session_id in zip(eeg_files, range(3)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            result_session = pool.map(
                partial(parallel_read_seed_feature, fi, dir_path, label), eeg_files[session_id])
        for i in range(15):
            eeg_data[session_id].append(result_session[i])
    return eeg_data, None, label, None, 62

def parallel_read_seed_feature(fi, dir_path, label, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(15):
        trail_data = list(np.array(subject_data[keys[i * 12+fi]]).transpose((1, 0, 2)))
        trail_datas.append(trail_data)
    return trail_datas

def read_seedIV_raw(dir_path):
    # input : 45 files(3 sessions, 15 round)
    # output : EEG signal with a trail as the basic unit and sample rate of the original dataset
    # output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label)

    dir_path += "/eeg_raw_data"
    eeg_files = [['1_20160518.mat', '2_20150915.mat', '3_20150919.mat',
                  '4_20151111.mat', '5_20160406.mat', '6_20150507.mat',
                  '7_20150715.mat', '8_20151103.mat', '9_20151028.mat',
                  '10_20151014.mat', '11_20150916.mat', '12_20150725.mat',
                  '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
                 ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat',
                  '4_20151118.mat', '5_20160413.mat', '6_20150511.mat',
                  '7_20150717.mat', '8_20151110.mat', '9_20151119.mat',
                  '10_20151021.mat', '11_20150921.mat', '12_20150804.mat',
                  '13_20151125.mat', '14_20151208.mat', '15_20150514.mat', ],
                 ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat',
                  '4_20151123.mat', '5_20160420.mat', '6_20150512.mat',
                  '7_20150721.mat', '8_20151117.mat', '9_20151209.mat',
                  '10_20151023.mat', '11_20151011.mat', '12_20150807.mat',
                  '13_20161130.mat', '14_20151215.mat', '15_20150527.mat', ]
                 ]

    # exctract the label for all trail in three sessions, label shape : (3, 24)
    label = np.zeros((3, 15, 24), dtype=int)
    ses_label1 = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
    ses_label2 = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
    ses_label3 = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    ses_label1s = np.tile(ses_label1, (1, 15, 1))
    ses_label2s = np.tile(ses_label2, (1, 15, 1))
    ses_label3s = np.tile(ses_label3, (1, 15, 1))
    label[0] = ses_label1s
    label[1] = ses_label2s
    label[2] = ses_label3s

    # Add a father session folder to each file
    for i, session in enumerate(eeg_files):
        eeg_files[i] = [f"{i + 1}/{sub_file}" for sub_file in session]

    # create the empty list of (3, 15, 24) => (session, subject, trail)
    eeg_data = [[[[] for _ in range(24)] for _ in range(15)] for _ in range(3)]
    # Loop processing of EEG mat files
    for session_files, session_id in zip(eeg_files, range(3)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            eeg_data[session_id] = pool.map(
                partial(parallel_read_seedIV_raw, dir_path), eeg_files[session_id])
    return eeg_data, None, label, 200, 62

def parallel_read_seedIV_raw(dir_path, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(24):
        trail_data = subject_data[keys[i]]
        trail_datas.append(trail_data[:,1:])
    return trail_datas


def read_seedIV_feature(dir_path, feature_type="de_lds"):
    # 读取seed IV数据集
    # input file : three folder each contains one session of 15 subjects' eeg data
    # output shape : (session(3), subject, trail, channel, feature), (session(3), subject, trail, label)
    # use the feature under eeg_feature_smooth dir, it has 3 dir, each dir represent 15 subejct
    # in each dir, it contains 15 subject files
    dir_path += "/eeg_feature_smooth"
    eeg_files = [['1_20160518.mat', '2_20150915.mat', '3_20150919.mat',
                  '4_20151111.mat', '5_20160406.mat', '6_20150507.mat',
                  '7_20150715.mat', '8_20151103.mat', '9_20151028.mat',
                  '10_20151014.mat', '11_20150916.mat', '12_20150725.mat',
                  '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
                 ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat',
                  '4_20151118.mat', '5_20160413.mat', '6_20150511.mat',
                  '7_20150717.mat', '8_20151110.mat', '9_20151119.mat',
                  '10_20151021.mat', '11_20150921.mat', '12_20150804.mat',
                  '13_20151125.mat', '14_20151208.mat', '15_20150514.mat',],
                 ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat',
                  '4_20151123.mat', '5_20160420.mat', '6_20150512.mat',
                  '7_20150721.mat', '8_20151117.mat', '9_20151209.mat',
                  '10_20151023.mat', '11_20151011.mat', '12_20150807.mat',
                  '13_20161130.mat', '14_20151215.mat', '15_20150527.mat', ]
                 ]

    #exctract the label for all trail in three sessions, label shape : (3, 24)
    label = np.zeros((3,15,24), dtype=int)
    ses_label1 = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    ses_label2 = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    ses_label3 = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    ses_label1s = np.tile(ses_label1, (1,15,1))
    ses_label2s = np.tile(ses_label2, (1,15,1))
    ses_label3s = np.tile(ses_label3, (1,15,1))
    label[0] = ses_label1s
    label[1] = ses_label2s
    label[2] = ses_label3s

    # Add a father session folder to each file
    for i, session in enumerate(eeg_files):
        eeg_files[i] = [f"{i+1}/{sub_file}" for sub_file in session]

    feature_index = {
        "de_movingAve": 0, "de_lds": 1, "psd_movingAve": 2, "psd_lds": 3
    }
    fi = feature_index[feature_type]

    eeg_data = [[] for _ in range(3)]
    # Define a function to read a single Mat file
    for ses_id, session_files in enumerate(eeg_files):
        with mp.Pool(processes=5) as pool:
            result_session = pool.map(
                partial(parallel_read_seedIV_feature, fi, dir_path, label), eeg_files[ses_id]
            )
        for i in range(15):
            eeg_data[ses_id].append(result_session[i])
    return eeg_data, None, label, None, 62
def parallel_read_seedIV_feature(fi, dir_path, label, file):
    subject_data = loadmat(f"{dir_path}/{file}")
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(24):
        trail_data = list(np.array(subject_data[keys[i*4+fi]].transpose((1,0,2))))
        trail_datas.append(trail_data)
    return trail_datas


def read_deap_preprocessed(dir_path):
    # 读取deap数据集
    # input file: 32 files contains 32 subject's eeg data
    # output shape : (session(1), subject, trail, channel, raw_data), (session(1), subject, trail, label)
    # under data_preprocess_python dir, it has 32.dat file, each represent one subject
    # every file contains two arrays:
    # data -> (trail(40), channel(40), data(8064))
    # label -> (trail(40), label(valence, arousal, dominance, liking))
    ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
                'P4', 'P8', 'PO4', 'O2']
    data = [[]]
    label = [[]]
    fs = 128
    pre_time = 3
    end_time = 63
    pretrail = pre_time * fs

    eeg_files = ["s{}.dat".format(str(i).zfill(2)) for i in range(1,33)]
    for s_i, subject_file in enumerate(eeg_files):
        sub_data = pickle.load(open("{}/".format(dir_path)+subject_file, "rb"), encoding="latin")
        baseline = np.mean([sub_data['data'][:,:32,i*fs:(i+1)*fs] for i in range(3)], axis=0)
        for sec in range(pre_time, end_time):
            sub_data['data'][:, :32, sec*fs: (sec+1)*fs] -= baseline
        sub_data_list = []
        sub_label_list = []
        for t_i, (trail_data, trail_label) in enumerate(zip(sub_data['data'], sub_data['labels'])):
            # trail_data shape->(channels(32eeg, 8others), raw_data)
            # trail_label shape->(labels(valence, arousal, dominance, liking))
            sub_data_list.append(trail_data[:32,pretrail:])
            sub_label_list.append(trail_label)
        # sub_data_list -> (trail, channels, raw_data)
        # sub_label_list -> (trail, labels)
        data[0].append(sub_data_list)
        label[0].append(sub_label_list)
    # data -> (session(1), subject, trail, channel, raw_data)
    # label -> (session(1), subject, trail, channel, raw_data)
    return data, None, label, 128, 32

def read_deap_raw(dir_path):
    # 读取deap原始数据集
    # input file : 32 bdf files contains 32 subjects' eeg data
    # output shape : (session(1), subject, trail, channel, raw_data), (session(1), subject, trail, label)
    # under data_original dir, it has 32.bdf file, each represent one subject
    Geneva_ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
                'P4', 'P8', 'PO4', 'O2']
    Twente_ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz',
                       'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4',
                       'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
    transfer_index =  [Twente_ch_names.index(s) for s in Geneva_ch_names]
    fs = 512
    pre_time = 3
    end_time = 63
    pretrail = pre_time * fs
    # when the code is 4, the experiment begin
    before_code = 3
    start_code1 = 4
    start_code2 = 1638148
    start_code3 = 5832452
    after_code = 5

    eeg_files = ["s{}.bdf".format(str(i).zfill(2)) for i in range(1,33)]
    label_file = ["s{}.dat".format(str(i).zfill(2)) for i in range(1,33)]
    all_raw_data = [[]]
    label = [[]]
    for s_i, subject_file in enumerate(eeg_files):
        sub_bdf_data = mne.io.read_raw_bdf("{}/data_original/".format(dir_path)+subject_file, preload=True
                                           , verbose=False)
        # print(sub_bdf_data.info['ch_names'])
        # get label easier
        label_data = pickle.load(open("{}/data_preprocessed_python/".format(dir_path)+label_file[s_i], "rb"),
                               encoding="latin")['labels']
        # read status code data
        status = np.array(sub_bdf_data.get_data()[47]).astype(int)
        changes = np.diff(status) != 0
        changes = np.insert(changes, 0, True)
        indices = np.where(changes)[0]
        # read raw data
        raw_data = np.array(sub_bdf_data.get_data()[:32])
        sub_raw_data = []
        sub_label = []
        pre_code = 0
        for begin, end in zip(indices, np.append(indices[1:], len(status))):
            # if s_i == 27:
            #     print(end-begin, status[begin])
            if pre_code == start_code1 or pre_code == start_code2 or pre_code == start_code3:
                # get last 60 seconds data points
                trail_raw_data = raw_data[:32, end-60*fs:end].tolist()
                if s_i < 22:
                    trail_raw_data = [trail_raw_data[tmp_i] for tmp_i in transfer_index]
                sub_raw_data.append(trail_raw_data)
            pre_code = status[begin]
        for t_i, trail_label in enumerate(label_data):
            sub_label.append(trail_label)
        all_raw_data[0].append(sub_raw_data)
        label[0].append(sub_label)
    return all_raw_data, None, label, 512, 32




def read_dreamer(dir_path, last_seconds = 60, base_seconds = 4):
    # input : 1 file (23 subjects' data)
    # subject data struct :
    #   Age, Gender, EEG, ECG, Valence(18 * 1), Arousal(18 * 1), Dominance(18 * 1)
    # subject's EEG data struct:
    #   sample rate : 128, num of electrodes : 14, num of subjects : 23
    #   electrodes : { 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'}
    # output shape : (session(1), subject, trail, channel, raw_data)
    file_path = dir_path + "/DREAMER.mat"
    data = loadmat(file_path)["DREAMER"]
    # data : [Data, EEG_sample_rate, ECG_sample_rate, EEG_electrodes, noOfSubjects, noOfVideoSequences
    # , Disclaimer, Provider, Version, Acknowledgement]
    # Data : [Age, Gender, EEG, ECG, ScoreValence, ScoreArousal, ScoreDominance]
    # EEG : [baseline, stimuli]
    # baseline & stimuli : [18, 1]
    #
    all_stimuli = [[[[] for _ in range(18)] for _ in range(23)]]
    all_base = [[[[] for _ in range(18)] for _ in range(23)]]
    all_labels = [[[[] for _ in range(18)] for _ in range(23)]]
    for subject in range(23):
        for trail in range(18):
            trail_stim = data[0,0]["Data"][0, subject]["EEG"][0, 0]["stimuli"][0, 0][trail, 0]
            trail_base = data[0,0]["Data"][0, subject]["EEG"][0, 0]["baseline"][0, 0][trail, 0]
            trail_valence = data[0,0]["Data"][0, subject]["ScoreValence"][0, 0][trail, 0]
            trail_arousal = data[0,0]["Data"][0, subject]["ScoreArousal"][0, 0][trail, 0]
            trail_dominance = data[0, 0]["Data"][0, subject]["ScoreDominance"][0, 0][trail, 0]
            trail_label = np.array([trail_valence, trail_arousal, trail_dominance])
            # print(trail_stim)
            # trail_stim shape : [128 * seconds(199), channel(14)]
            # trail_label shape : [3]
            all_stimuli[0][subject][trail] = trail_stim[-last_seconds*128:].transpose()
            all_base[0][subject][trail] = trail_base[-base_seconds*128:].transpose()
            all_labels[0][subject][trail] = trail_label
            # all_stimuli[0][subject][trail] shape : [channel(14), seconds(last_seconds) * sample rate(128)]
            # all_labels[0][subject][trail] shape : [3]
    return all_stimuli, all_base, all_labels, 128, 14


def read_hci(dir_path):
    # 30 subjects, [20, 20, 17, 20, 20, 20, 20, 20, 14, 20, 20, 0, 20, 20, 0, 16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    # input : 1 dir ( contains 1200 file )
    # output shape (session(1), subject, trail, channel, raw_data)
    baseline_sec = 30
    dir_path = dir_path + "/Sessions/"
    file_names = [name for name in os.listdir(dir_path)]
    emo_states = ['@feltVlnc', '@feltArsl']
    data = [[[] for _ in range(30)]]
    base = [[[] for _ in range(30)]]
    labels = [[[] for _ in range(30)]]

    for file in file_names:
        sub_dir = dir_path + file
        label_file = sub_dir + "/session.xml"
        with open(label_file) as f:
            label_info = xmltodict.parse('\n'.join(f.readlines()))
        label_info = json.loads(json.dumps(label_info))["session"]
        if not '@feltArsl' in label_info:
            continue
        trail_label = np.array([int(label_info[k]) for k in emo_states])
        sub = int(label_info['subject']['@id'])
        trail_file = [sub_dir+"/"+f for f in os.listdir(sub_dir) if fnmatch.fnmatch(f,'*.bdf')][0]
        raw = mne.io.read_raw_bdf(trail_file, preload=True, stim_channel='Status', verbose=False)
        events = mne.find_events(raw, stim_channel='Status', verbose=False)
        montage = mne.channels.make_standard_montage(kind='biosemi32')
        raw.set_montage(montage, on_missing='ignore')
        raw.pick(raw.ch_names[:32])
        start_samp, end_samp = events[0][0] + 1, events[1][0] - 1
        baseline = raw.copy().crop(raw.times[0], raw.times[end_samp])
        baseline = baseline.resample(128)
        baseline_data = baseline.to_data_frame().to_numpy()[:, 1:].swapaxes(1, 0)
        baseline_data = baseline_data[:, :baseline_sec * 128]
        baseline_data = baseline_data.reshape(32, baseline_sec, 128).mean(axis=1)

        trail_bdf = raw.copy().crop(raw.times[start_samp], raw.times[end_samp])
        trail_bdf = trail_bdf.resample(128)
        trail_data = trail_bdf.to_data_frame().to_numpy()[:,1:].swapaxes(1,0)
        data[0][sub-1].append(trail_data)
        base[0][sub-1].append(baseline_data)
        labels[0][sub-1].append(trail_label)

    filter_d_l_b = [(d,l,b) for d,l,b in zip(data[0], labels[0], base[0]) if l != []]
    data[0], labels[0], base[0] = zip(*filter_d_l_b) if filter_d_l_b else ([],[],[])
    return data, base, labels, 128, 32


# ===================== SEED-V =====================
def read_seedV_feature(dir_path, feature_type="de"):
    """
    读取 SEED-V 官方特征（基于 datasets/SEED-V/EEG_DE_features/*.npz）。

    目录示例:
      - EEG_DE_features/
          1_123.npz, 2_123.npz, ..., 16_123.npz

    每个 npz 含有两个 0 维字节数组键: 'data', 'label'，它们是 pickled dict：
      - data: {0..44: np.ndarray(num_segments, 310)}  310 = 62 channels × 5 bands
      - label: {0..44: np.ndarray(num_segments,)}      每段的情感类别，取第一个值作为该 trial 的标签
      num_segments 为每个 trial 有多少份数据，不同的 trail 的 num_segments 也不同

    返回：
      data:    (session=3, subject=16, trail=15, sample, channel=62, band=5)
      baseline: None
      label:   (session=3, subject=16, trail=15)  整数标签 0..4
      sample_rate: None（不用于已提取特征的流程）
      channels: 62

    feature_type 支持: 'de' 与 'de_lds'（后者对每 trial 的 (sample,channel,band) 施加 LDS 平滑）
    """
    feat_dir = os.path.join(dir_path, "EEG_DE_features")
    if not os.path.isdir(feat_dir):
        raise FileNotFoundError(f"SEED-V features directory not found: {feat_dir}")

    # 收集受试者 npz 文件（名称形如 '1_123.npz'）
    npz_files = [f for f in os.listdir(feat_dir) if f.endswith('.npz')]
    # 仅匹配以数字开头的 1..n 受试者
    def subj_key(name: str):
        try:
            return int(name.split('_', 1)[0])
        except Exception:
            return 1 << 30
    npz_files.sort(key=subj_key)
    if not npz_files:
        raise FileNotFoundError(f"No npz files found in {feat_dir}")

    subjects = len(npz_files)
    channels = 62
    bands = 5

    # 预分配容器: 3 session × subjects × 15 trials
    data = [[[[] for _ in range(15)] for _ in range(subjects)] for _ in range(3)]
    labels = [[[0 for _ in range(15)] for _ in range(subjects)] for _ in range(3)]

    for s_idx, npz_name in enumerate(npz_files):
        npz_path = os.path.join(feat_dir, npz_name)
        with np.load(npz_path, allow_pickle=True) as f:
            # 解包 pickled dict
            data_obj = pickle.loads(f['data'].tobytes())
            label_obj = pickle.loads(f['label'].tobytes())
        # 按键排序并按 15 试次 × 3 会话切分
        # data_obj.keys() 返回字典试图对象，还不是列表
        keys = sorted(list(data_obj.keys()))
        if len(keys) != 45:
            # 冗余操作，防止trial数不足45个
            # 兜底：尝试按 15 的倍数切分；否则所有 trial 放到第一 session
            # 一个session 15 个trial，三个session放在同一个 .npz 中
            total_trials = len(keys)
            ses_counts = [min(15, total_trials), min(15, max(0, total_trials-15)), max(0, total_trials-30)]
        else:
            ses_counts = [15, 15, 15]

        offset = 0
        for ses_id in range(3): # 0 1 2
            for t in range(ses_counts[ses_id]): # 正常来说循环15次
                k = keys[offset + t]
                trial_2d = np.asarray(data_obj[k])  # (num_segments, 310)
                if trial_2d.ndim != 2 or trial_2d.shape[1] != channels * bands:
                    raise ValueError(f"Unexpected feature shape in {npz_name}, key={k}: {trial_2d.shape}, expect (*, {channels*bands})")
                num_segments = trial_2d.shape[0]
                # 还原为 (sample, channel, band)
                trial_feat = trial_2d.reshape(num_segments, channels, bands)

                # 可选 LDS 平滑（以 sample 维为时间）
                if feature_type.endswith('_lds'):
                    trial_feat = lds(trial_feat)

                # trial 标签：取该 trial 的第一个段标签
                trial_label_arr = np.asarray(label_obj[k])
                if trial_label_arr.size == 0:
                    trial_label = 0
                else:
                    trial_label = int(trial_label_arr.flat[0])

                data[ses_id][s_idx][t] = trial_feat.tolist() # reshape后的数据转换成列表存储
                labels[ses_id][s_idx][t] = trial_label
            # 更新 offset
            offset += ses_counts[ses_id]

    return data, None, labels, None, channels


def _parse_seedv_timestamps(ts_file: str):
    """Parse trial_start_end_timestamp.txt into a dict: {1: [(s,e),...15], 2: [...], 3: [...]} in seconds."""
    if not os.path.isfile(ts_file):
        raise FileNotFoundError(f"SEED-V timestamps file not found: {ts_file}")
    with open(ts_file, 'r', encoding='utf-8') as f:
        text = f.read()
    sessions = {}
    for ses in (1, 2, 3):
        # crude parse for lines after 'Session X:'
        marker = f"Session {ses}:"
        idx = text.find(marker)
        if idx == -1:
            raise ValueError(f"Timestamps missing block for Session {ses}")
        sub = text[idx:]
        # find start and end arrays in brackets
        def extract_array(name):
            midx = sub.find(name)
            if midx == -1:
                raise ValueError(f"Timestamps missing '{name}' for Session {ses}")
            sidx = sub.find('[', midx)
            eidx = sub.find(']', sidx)
            arr = sub[sidx+1:eidx]
            nums = [int(x.strip()) for x in arr.split(',') if x.strip()]
            return nums
        starts = extract_array('start_second')
        ends = extract_array('end_second')
        if len(starts) != 15 or len(ends) != 15:
            raise ValueError(f"Session {ses} expected 15 trials, got {len(starts)} starts and {len(ends)} ends")
        sessions[ses] = list(zip(starts, ends))
    return sessions


def read_seedV_raw(dir_path, target_sfreq: int = 200):
    """
    读取 SEED-V 原始 EEG（EEG_raw/*.cnt），按 trial_start_end_timestamp.txt 切分 trial。

    返回：
      - data: (session=3, subject, trail=15, channel=62, time_points)
      - baseline: None
      - label: (session=3, subject, trail) 使用对应 subject 的 DE 特征 npz 的标签
      - sample_rate: target_sfreq (默认重采样为 200Hz)
      - channels: 62
    """
    raw_dir = os.path.join(dir_path, 'EEG_raw')
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"SEED-V raw directory not found: {raw_dir}")
    ts_path = os.path.join(dir_path, 'trial_start_end_timestamp.txt')
    ses_ts = _parse_seedv_timestamps(ts_path)

    # group files by subject and session from pattern like '1_1_20180804.cnt'
    all_cnt = [f for f in os.listdir(raw_dir) if f.lower().endswith('.cnt')]
    def parse_info(name: str):
        # returns (subject_id:int, session_id:int)
        stem = os.path.splitext(name)[0]
        parts = stem.split('_')
        if len(parts) < 2:
            return None
        try:
            sid = int(parts[0])
            ses = int(parts[1])
            return sid, ses
        except Exception:
            return None
    file_map = {}
    for fname in all_cnt:
        info = parse_info(fname)
        if info is None:
            continue
        subj, ses = info
        file_map.setdefault(subj, {})[ses] = fname

    subj_ids = sorted(file_map.keys())
    if not subj_ids:
        raise FileNotFoundError(f"No CNT files found in {raw_dir}")

    subjects = len(subj_ids)
    channels = 62
    sample_rate = target_sfreq

    data = [[[[] for _ in range(15)] for _ in range(subjects)] for _ in range(3)]
    labels = [[[0 for _ in range(15)] for _ in range(subjects)] for _ in range(3)]

    # Prepare labels from DE feature npz per subject for consistency
    feat_dir = os.path.join(dir_path, 'EEG_DE_features')
    use_npz_labels = os.path.isdir(feat_dir)

    for s_idx, subj in enumerate(subj_ids):
        # optional: load npz labels for this subject
        npz_labels = None
        if use_npz_labels:
            npz_name = None
            # find file like '{subj}_*.npz'
            for f in os.listdir(feat_dir):
                if f.startswith(f"{subj}_") and f.endswith('.npz'):
                    npz_name = os.path.join(feat_dir, f)
                    break
            if npz_name is not None:
                with np.load(npz_name, allow_pickle=True) as ff:
                    try:
                        lab_obj = pickle.loads(ff['label'].tobytes())
                        # build ordered labels by keys 0..44
                        keys = sorted(list(lab_obj.keys()))
                        npz_labels = [int(np.asarray(lab_obj[k]).flat[0]) if np.asarray(lab_obj[k]).size>0 else 0 for k in keys]
                    except Exception:
                        npz_labels = None

        for ses_id in (1, 2, 3):
            if ses_id not in file_map[subj]:
                continue
            cnt_path = os.path.join(raw_dir, file_map[subj][ses_id])
            # Suppress benign meas date warning from CNT headers
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*Could not parse meas date from the header.*",
                    category=RuntimeWarning,
                )
                raw = mne.io.read_raw_cnt(cnt_path, preload=True, verbose=False)
            # pick first 62 EEG channels (assumption: EEG at the top)
            raw.pick(raw.ch_names[:channels])
            if target_sfreq is not None and abs(raw.info['sfreq'] - target_sfreq) > 1e-6:
                raw.resample(target_sfreq)
            # slice trials per timestamps
            for t_idx, (start_s, end_s) in enumerate(ses_ts[ses_id]):
                # safety guards
                start_samp = max(0, int(start_s * target_sfreq))
                end_samp = min(int(end_s * target_sfreq), raw.n_times)
                seg = raw.get_data(start=start_samp, stop=end_samp)  # (channels, time)
                data[ses_id-1][s_idx][t_idx] = seg.tolist()
                # label from npz if available
                if npz_labels is not None:
                    k = (ses_id-1)*15 + t_idx
                    if 0 <= k < len(npz_labels):
                        labels[ses_id-1][s_idx][t_idx] = int(npz_labels[k])
                    else:
                        labels[ses_id-1][s_idx][t_idx] = 0
                else:
                    labels[ses_id-1][s_idx][t_idx] = 0

    return data, None, labels, sample_rate, channels
