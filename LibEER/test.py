import fnmatch
import time
import mne

from scipy.signal import butter, filtfilt, stft
import multiprocessing as mp
from functools import partial

from torch.nn.modules.module import T

from ..data_utils.load_data import get_uniform_data, read_seedIV_raw, read_seed_feature, read_seedIV_feature
from ..data_utils.preprocess import bandpass_filter, feature_extraction, psd_extraction
from scipy.io import loadmat
import numpy as np
import torch



def parallel_read_seed_raw(dir_path, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    label_datas = []
    for i in range(15):
        trail_data = subject_data[keys[i]]
        trail_datas.append(trail_data[:, 1:])
    return trail_datas


def read_seed_raw_new(dir_path):
    # input : 45 files(3 sessions, 15 round) containing all 15 trails with a sampling rate of 200 Hz
    # output : EEG signal with a trail as the basic unit and sample rate of the original dataset
    # output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label)

    # Extract the EEG data of each subject from the SEED dataset, and partition the data of each session
    dir_path += "/Preprocessed_EEG"
    eeg_files = [['1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
                  # '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
                  # '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
                  # '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
                  # '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'
                  ],
                 [
                     # '1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
                     #  '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
                     #  '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
                     #  '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
                     #  '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'
                 ],
                 [
                     # '1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
                     #  '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
                     #  '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
                     #  '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
                     #  '13_20140610.mat', '14_20140627.mat', '15_20131105.mat'
                 ]
                 ]
    # Extract the label for all trail in three sessions
    label = np.array(loadmat(f"{dir_path}/label.mat")['label'])
    labels = np.tile(label, (3, 15, 1))

    # create the empty list of (3, 15, 15) => (session, subject, trail)
    eeg_data = [[] for _ in range(1)]
    # Loop processing of EEG mat files
    for session_files, session_id in zip(eeg_files, range(1)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            eeg_data[session_id] = pool.map(
                partial(parallel_read_seed_raw, dir_path), eeg_files[session_id])
    return eeg_data, labels, 200


def bandpass_filter_new(data, frequency, pass_band):
    """
    Perform baseband filtering operation on EEG signal
    input: EEG signal
    output: EEG signal with band-pass filtering
    input shape : (session, subject, trail, channel, original_data)
    output shape : (session, subject, trail, channel, filter_data)
    """
    # define Nyquist frequency which is the minimum sampling rate defined to prevent signal aliasing
    nyq = 0.5 * frequency
    # get the coefficients of a Butterworth filter
    b, a = butter(N=5, Wn=[pass_band[0] / nyq, pass_band[1] / nyq], btype='bandpass')
    # perform linear filtering
    # process on all channels
    with mp.Pool(processes=3) as pool:
        data = pool.map(partial(session_filtfilt, b, a), data)
    return data


def session_filtfilt(b, a, raw_data):
    filter_data = []
    for subject in raw_data:
        filter_trail_data = []
        for trail in subject:
            filter_trail_data.append(filtfilt(b, a, trail))
        filter_data.append(filter_trail_data)
    return filter_data

# def compute_de(signal):


def feature_band_extraction(data, label, sample_rate, extract_bands=None, time_window=1, nperseg=200):
    """
    Extract data from various frequency bands of the EEG signal
    input: EEG signal with band-pass filtering
    output: information extracted from multiple frequency bands of EEG signals
    input shape -> data:  (session, subject, trail, channel, filter_data)
                   label: (session, subject, trail, label)
    output shape -> data:  (session, subject, trail, sample, time_window, channel, band, extract_data),
                    label: (session, subject, trail, sample, label)
    """
    # print(data)
    if extract_bands is None:
        extract_bands = [[1, 3]   , [4, 7], [8, 13], [14, 30], [31, 50]]
    de_features =[[[[[] for _ in range(len(extract_bands))] for _ in range(len(data[0][0]))]for _ in range(len(data[0]))] for _ in range(len(data))]
    for ses_id, ses_data in enumerate(data):
        for sub_id, sub_data in enumerate(ses_data):
            for trail_id, trail_data in enumerate(sub_data):
                # fs shape->(nperseg/2), ts shape->(num of sampling rate/ nperseg)
                # Zxx shape->(channel, fs_shape, ts_shape)
                fs, ts, Zxx = stft(trail_data, fs=sample_rate, window='hann', nperseg=nperseg, noverlap=0,
                                   boundary=None)
                for b_idx, band in enumerate(extract_bands):
                    fb_indices = np.where((fs >= band[0]) & (fs <= band[1]))[0]
                    fourier_coe = np.real(Zxx[:, fb_indices, :])
                    print(fourier_coe.shape)
                    parseval_energy = np.sum(np.square(fourier_coe), axis=1) / nperseg
                    de_features[ses_id][sub_id][trail_id][b_idx] = 1 / 2 * np.log2(parseval_energy) + 1 / 2 * np.log2(2 * np.pi * np.e / nperseg)
    return de_features

import os
import xmltodict
import json






def main():
    data,_,label,_,a = read_seedIV_raw("/data1/cxx/SEED数据集/SEED_IV")
    maxLen = 0
    minLen = 10000000
    for ses in data:
        for sub in ses:
            for trail in sub:
                print(len(trail[0]))
                maxLen = max(len(trail[0]), maxLen)
                minLen = min(len(trail[0]),minLen)
    print(maxLen/200, minLen/200)

    print(len(data[0][0][0]))

def test():
    # available_label = ['valence', 'arousal', 'dominance', 'liking']
    # label_used = ['valence']
    # used_id = [available_label.index(t) for t in label_used]
    # print(used_id)
    # pro_label = [1,0]
    # print(int("".join(str(i) for i in pro_label), 2))
    # a = 0
    # if type(a) is np.ndarray:
    #     return
    a = [1,2,3,4,5,6]
    b = [0,2]
    print(a[i] for i in b)


import torch
import torch.nn as nn

class eegnet(nn.Module):
    def __init__(self,ele_channel = 32, datapoints = 128, num_classes = 3, F1 = 3, F2=5, D = 5, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 64), padding='same', bias=True)
        self.BN1 = nn.BatchNorm2d(F1)
        self.depth_conv = nn.Conv2d(in_channels=F1, out_channels=F1*D, kernel_size=(ele_channel, 1), bias=True, groups=F1)
        self.BN2 = nn.BatchNorm2d(D*F1)
        self.act1 = nn.ELU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(1,4), stride=4)
        self.dropout1 = nn.Dropout(dropout)
        self.sep_conv = nn.ModuleList()
        self.sep_conv.append(nn.Conv2d(in_channels=D*F1, out_channels=D*F1, kernel_size=(1,16), padding='same', bias=True, groups=D*F1))
        self.sep_conv.append(nn.Conv2d(in_channels=D*F1, out_channels=F2, kernel_size=1, bias=True))
        self.BN3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=(1,8), stride=8)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(F2*datapoints//32, num_classes)

    def init_weight(self):
        return
    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.depth_conv(x)
        x = self.BN2(x)
        x = self.act1(x)
        x = self.pool1(x)
        print(x.shape)
        x = self.dropout1(x)
        for conv in self.sep_conv:
            x = conv(x)
        print(x.shape)
        x = self.BN3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.fc(x)
        return x


def depth_wise_conv():
    batch_size = 128
    ele_channel = 32
    datapoints = 128
    F1 = 3
    D = 5
    x = torch.rand(batch_size, 1, ele_channel, datapoints)
    model = eegnet()


    out = model(x)
    print(out.shape)



from glob import glob
import scipy.signal
import matplotlib.pyplot as plt
import pickle
from ..data_utils.load_data import read_deap_preprocessed, label_process

def deap_preprocessing():
    t1 = time.time()
    data, label, fre, _ = read_deap_preprocessed('/date1/yss/data/DEAP数据集/data_preprocessed_python')
    valence_label = []
    for sub_label in label[0]:
        v_sub_label = []
        for trail_label in sub_label:
            if trail_label[0]<=4 or trail_label[0]>=6:
                v_sub_label.append(trail_label[0])
        valence_label.append(len(v_sub_label))
    print(valence_label)
    print(len(data), len(data[0]), len(data[0][0]), len(data[0][0][0]))
    print(len(label), len(label[0]), len(label[0][0]),len(label[0][0][0]))
    print(fre)
    t2 = time.time()
    print(t2-t1)
    extract_bands = [[4,7], [8,10], [8,12], [13,30], [30,47]]
    psd_data = feature_extraction(data, fre, extract_bands, 6, 3, 'psd')

    print(len(psd_data), len(psd_data[0]), len(psd_data[0][0]), len(psd_data[0][0][0]), psd_data[0][0][0].shape)
    data, label, num_classes = label_process(psd_data, label, bounds=[4, 6], onehot=False, label_used=['valence'])

    # print(len(label), len(label[0]), len(label[0][0]), len(label[0][0][0]), label[0][0][0].shape)
    # print(len(data) * (len(data[0])) * len(data[0][0]))
    # for ses_data in data:
    #     for sub_data in ses_data:
    #         for trail_data in sub_data:
    #             print(trail_data.shape)
from ..data_utils.preprocess import segment_data
from ..data_utils.split import merge_to_part
from ..config.setting import seed_sub_dependent_front_back_setting
from ..utils.args import get_args_parser
def test_de():
    pass_band = [0.5 ,60]
    extract_bands = [[1, 4], [4, 8], [8, 14], [14, 31], [31, 50]]
    data, label, sample_rate, channels = get_uniform_data('seed_raw', '/date1/yss/data/SEED数据集/SEED')
    data = bandpass_filter(data, sample_rate, pass_band)
    data = feature_extraction(data, sample_rate, extract_bands, 1, 0, 'de')
    data, feature_dim = segment_data(data,1,1)
    args = get_args_parser()
    args = args.parse_args()
    setting = seed_sub_dependent_front_back_setting(args=args)
    data, label, num_classes = label_process(data=data, label=label, bounds=setting.bounds, onehot=setting.onehot, label_used=setting.label_used)
    data, label = merge_to_part(data, label, setting)

def test_m():
    from ..data_utils.preprocess import generate_rgnn_adjacency_matrix
    from ..data_utils.constants.seed import SEED_CHANNEL_NAME, SEED_GLOBAL_CHANNEL_PAIRS
    from ..data_utils.constants.channel_location import system_10_20_loc, system_10_05_loc
    a = generate_rgnn_adjacency_matrix(channel_names=SEED_CHANNEL_NAME,channel_loc=system_10_05_loc,global_channel_pair=SEED_GLOBAL_CHANNEL_PAIRS)
    # print(a)
    # if np.allclose(a, np.transpose(a)):
    #     print(a)

# def test_hci():
#     from scipy.io import loadmat
#     file1_path = "/home/zyz/eegdata/hci_preprocess_eeg+peripheral_10s_28_XX_38_10_256.mat"
#     # file2_path = "/home/zyz/eegdata/hci_DE_eeg+peripheral_10s_28_XX_38_10_5.mat"
#     import hdf5storage
#     hci_data = hdf5storage.loadmat(file1_path)
#     print(len(hci_data['X']), len(hci_data['X'][0]))
#     print((np.array(hci_data['X'][0].shape)))
# def read_hci():
#     import hdf5storage
#     file1_path = "/home/zyz/eegdata/hci_preprocess_eeg+peripheral_10s_28_XX_38_10_256.mat"
#     hci_data = hdf5storage.loadmat(file1_path)
#     #hci_data X: (sub, sample, channel, time, fre)
#     #Y:(sub, sample, label)
#     data = []
#     for sub_data in hci_data['X']:
#         new_sub_data = []
#         for trail_data in sub_data:
#             trail_data = np.array(trail_data)
#             trail_data = trail_data.reshape(trail_data.shape[0], trail_data.shape[1] * trail_data.shape[2])
#             # trail_data shape (channel, data_points)
#             new_sub_data.append([trail_data[:32]])
#         data.append(new_sub_data)
#     label = []
#     for sub_label in hci_data['Y']:
#         new_sub_label = []
#         new_sub_label = []
#         for trail_label in sub_label:
#             trail_label = np.array(trail_label)
#             # new_trail_label = trail_label[1] # arousal
#             new_trail_label = trail_label[2]  # valence
#             # new_trail_label = trail_label[1] + trail_label[2] * 2 # av
#             new_sub_label.append([new_trail_label])
#         label.append(new_sub_label)
#     # data shape (sub, sample, point, channel)
#     # label shape (sub, sample, label)
#     return data, label
#     # for new_sub_label in label:
#     #     new_sub_label = np.array(new_sub_label)
#     #     print(new_sub_label.shape)

def test_deap_original():
    from ..data_utils.load_data import read_deap_raw
    read_deap_raw("/date1/yss/data/DEAP数据集")

def test_dreamer():
    from ..data_utils.load_data import read_dreamer
    from ..data_utils.preprocess import preprocess
    import matplotlib.pyplot as plt
    data, base, label, sample_rate, channel = read_dreamer("/date1/yss/data/Dreamer数据集")
    data = [[[data[0][0][0]]]]
    baseline = [[[base[0][0][0]]]]
    print(len(data[0][0][0]))

    ax = plt.subplot()
    plot_data = baseline[0][0][0]
    # ax.plot(plot_data[0, :], label=f'Row {1}', linewidth=0.1)
    data = bandpass_filter(data, sample_rate, pass_band=[4,30])
    baseline = bandpass_filter(baseline, sample_rate, pass_band=[4,30])

    # data, label, sample_rate, channel = read_deap_preprocessed('/date1/yss/data/DEAP数据集/data_preprocessed_python')
    # print(data[0][0][0])
    # data, feature_dim = preprocess(data=data, baseline=base, sample_rate=sample_rate, pass_band=[4,30],extract_bands=[[4,7],[8,13],[14,30]]
    #                                ,sample_length=1, stride=1, time_window=2, overlap=1,only_seg=True,feature_type="psd",eog_clean=False)

    # plot_data = data[0][0][0
    plot_data = baseline[0][0][0]
    print(len(plot_data))
    # ax.plot(baseline[0, :], linewidth=0.1)
    ax.plot(plot_data[0, :], label=f'Row {1}', linewidth=0.1)
    # ax.legend(loc='upper right')
    # for i in range(len(plot_data)):
    #     ax = plt.subplot(len(plot_data), 1, i+1)
    #     ax.plot(plot_data[i,:],label=f'Row {i+1}')
    #     ax.legend(loc='upper right')
    #     plt.tight_layout()
    plt.show()

    # print(data[0][0][0])
    # data, label, num_classes = label_process(data=data, label=label,bounds=[3,4],label_used=["valence", "arousal"])

    # print(data)
    # print(label)

def testTSception():
    def generate_TS_channel_order(original_order: list):
        """
            This function will generate the channel order for TSception
            Parameters
            ----------
            original_order: list of the channel names

            Returns
            -------
            TS: list of channel names which is for TSception
            """
        chan_name, chan_num, chan_final = [], [], []
        for channel in original_order:
            chan_name_len = len(channel)
            k = 0
            for s in [*channel[:]]:
                if s.isdigit():
                    k += 1
            if k != 0:
                chan_name.append(channel[:chan_name_len - k])
                chan_num.append(int(channel[chan_name_len - k:]))
                chan_final.append(channel)
        chan_pair = []
        for ch, id in enumerate(chan_num):
            if id % 2 == 0:
                chan_pair.append(chan_name[ch] + str(id - 1))
            else:
                chan_pair.append(chan_name[ch] + str(id + 1))
        chan_no_duplicate = []
        [chan_no_duplicate.extend([f, chan_pair[i]]) for i, f in enumerate(chan_final) if f not in chan_no_duplicate]
        return chan_no_duplicate[0::2] + chan_no_duplicate[1::2]
    from ..data_utils.constants.seed import SEED_CHANNEL_NAME
    # print(generate_TS_channel_order(SEED_CHANNEL_NAME))
    # print(len(generate_TS_channel_order(SEED_CHANNEL_NAME)))

# def read_hci(dir_path):
#     # 30 subjects, [20, 20, 17, 20, 20, 20, 20, 20, 14, 20, 20, 0, 20, 20, 0, 16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
#     # input : 1 dir ( contains 1200 file )
#     # output shape (session(1), subject, trail, channel, raw_data)
#     baseline_sec = 30
#     dir_path = dir_path + "/Sessions/"
#     file_names = [name for name in os.listdir(dir_path)]
#     emo_states = ['@feltVlnc', '@feltArsl']
#     data = [[[] for _ in range(30)]]
#     base = [[[] for _ in range(30)]]
#     labels = [[[] for _ in range(30)]]
#
#     for file in file_names:
#         sub_dir = dir_path + file
#         label_file = sub_dir + "/session.xml"
#         with open(label_file) as f:
#             label_info = xmltodict.parse('\n'.join(f.readlines()))
#         label_info = json.loads(json.dumps(label_info))["session"]
#         if not '@feltArsl' in label_info:
#             # skip label_info['@isStim'] == '0' and other exception
#             continue
#
#         trail_label = [int(label_info[k]) for k in emo_states]
#         sub = int(label_info['subject']['@id'])
#         trail_file = [sub_dir+"/"+f for f in os.listdir(sub_dir) if fnmatch.fnmatch(f,'*.bdf')][0]
#         raw = mne.io.read_raw_bdf(trail_file, preload=True, stim_channel='Status')
#         events = mne.find_events(raw, stim_channel='Status')
#         montage = mne.channels.make_standard_montage(kind='biosemi32')
#         raw.set_montage(montage, on_missing='ignore')
#         raw.pick_channels(raw.ch_names[:32])
#         start_samp, end_samp = events[0][0] + 1, events[1][0] - 1
#         baseline = raw.copy().crop(raw.times[0], raw.times[end_samp])
#         baseline = baseline.resample(128)
#         baseline_data = baseline.to_data_frame().to_numpy()[:, 1:].swapaxes(1, 0)
#         baseline_data = baseline_data[:, :baseline_sec * 128]
#         baseline_data = baseline_data.reshape(32, baseline_sec, 128).mean(axis=1)
#
#         trail_bdf = raw.copy().crop(raw.times[start_samp], raw.times[end_samp])
#         trail_bdf = trail_bdf.resample(128)
#         trail_data = trail_bdf.to_data_frame().to_numpy()[:,1:].swapaxes(1,0)
#
#         data[0][sub-1].append(trail_data)
#         base[0][sub-1].append(baseline_data)
#         labels[0][sub-1].append(trail_label)
#     # for sub_data in data[0]:
#     #     print(len(sub_data))
#     #     for trail_data in sub_data:
#     #         print(trail_data.shape)
#
#     return data, base, labels, 128, 32



if __name__ == '__main__':
    # test()
    # main()
    # deap_preprocessing()
    # depth_wise_conv()
    # test_de()
    # test_de()
    # test_hci()
    # read_hci()
    # hci_de()
    # print("hello best")
    # test_dreamer()
    # test_deap_original()
    # testTSception()
    main()