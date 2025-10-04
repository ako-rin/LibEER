
SEED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4','F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
HSLT_SEED_Regions = {
    'PF': ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4'],
    'F':  ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    'LT': ['FT7', 'FC5', 'FC3', 'T7', 'C5', 'C1'], # C3
    'RT': ['FT8', 'FC4', 'FC6', 'T8', 'C2', 'C6', 'CP6'], # C4
    'C':  ['FC1', 'C3', 'CZ', 'FCZ', 'FC2', 'C4'],
    'LP': ['TP7', 'CP5', 'CP3', 'P7', 'P5', 'P3', 'P1', 'PO3'],
    'P':  ['CP1', 'CP2', 'CPZ', 'PZ'],
    'RP': ['TP8', 'CP4', 'P8', 'P6', 'P2', 'P4', 'PO4'],
    'O':  ['PO7', 'PO5', 'POZ', 'PO6', 'PO8', 'CB1', 'O1', 'O2', 'OZ', 'CB2']
}
sum = 0
hashset = set()
for key, value in HSLT_SEED_Regions.items():
    print(len(value))
    sum+= len(value)

print(sum)


from .config.setting import Setting

setting = Setting(dataset='deap',
                  dataset_path='DEAP/data_preprocessed_python',
                  pass_band=[0.3, 50],
                  extract_bands=[[0.5,4],[4,8],[8,14],[14,30],[30,50]],
                  time_window=1,
                  overlap=0,
                  sample_length=1,
                  stride=1,
                  seed=2024,
                  feature_type='de',
                  only_seg=False,
                  experiment_mode="subject-dependent",
                  split_type='train-val-test',
                  test_size=0.2,
                  val_size=0.2)
from .config.setting import seed_sub_dependent_front_back_setting