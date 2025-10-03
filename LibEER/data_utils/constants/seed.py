# SEED dataset constants
import numpy as np

# Basic channel names for SEED dataset
SEED_CHANNEL_NAME = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ',
    'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7',
    'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6',
    'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

# Placeholder adjacency matrix for RGNN (62x62 identity matrix as fallback)
SEED_RGNN_ADJACENCY_MATRIX = np.eye(62, dtype=np.float32)

# HSLT model brain regions for SEED dataset
# Based on SEED 62-channel layout
HSLT_SEED_Regions = {
    'frontal': list(range(0, 14)),        # Front region
    'central': list(range(14, 32)),       # Central region  
    'temporal': list(range(32, 42)),      # Temporal region
    'parietal': list(range(42, 56)),      # Parietal region
    'occipital': list(range(56, 62))      # Occipital region
}
