# DEAP dataset constants

# Standard 32-channel EEG montage for DEAP
DEAP_CHANNEL_NAME = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
    'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
    'P4', 'P8', 'PO4', 'O2'
]

import numpy as np

# Placeholder adjacency matrix for DEAP dataset RGNN (32x32 identity matrix as fallback)
DEAP_RGNN_ADJACENCY_MATRIX = np.eye(32, dtype=np.float32)

# HSLT model brain regions for DEAP dataset
# Based on standard EEG 10-20 system regions
HSLT_DEAP_Regions = {
    'frontal': [0, 1, 2, 3, 18, 19],      # Fp1, AF3, F3, F7, Fz, F4
    'central': [4, 5, 6, 21, 22, 23, 24], # FC5, FC1, C3, FC6, FC2, Cz, C4
    'temporal': [7, 25],                   # T7, T8
    'parietal': [8, 9, 10, 15, 26, 27, 28, 29], # CP5, CP1, P3, Pz, CP6, CP2, P4, P8
    'occipital': [11, 12, 13, 14, 30, 31] # P7, PO3, O1, Oz, PO4, O2
}
