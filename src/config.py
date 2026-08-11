import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class Config:
    # Hardware & Training Hyperparameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    BATCH_SIZE = 4
    EPOCHS = 10
    MAX_LR = 1e-4
    WEIGHT_DECAY = 1e-2
    
    # Dataset Specifications
    IMAGE_SIZE = (336, 336)
    PHYSICAL_CROP_MM = 130.0
    CACHE_SLICES = 6
    SLICE_BAND = (0.20, 0.80)
    GROUP_SIZE = 3  # Sliding window TTA size
    
    # 12 Target Abnormalities
    TARGETS = [
        'ACL', 'MCL', 'Medial_Meniscus', 'Lateral_Meniscus',
        'Medial_OA', 'Lateral_OA', 'PF_OA', 'Effusion',
        'Synovitis', 'Bakers', 'Contusion', 'Fracture'
    ]
    
    # 6 Target Slots
    SLOTS = [
        'SAG_FLUID_FS', 'COR_FLUID_FS', 'AX_FLUID_FS',
        'SAG_FLUID_NOFS', 'COR_T1', 'SAG_T1'
    ]
    
    # Model Architecture Specifications
    MODEL_NAME = 'facebook/dinov2-small'
    UNFREEZE_LAST = 6
    SLOT_EMBED_DIM = 768  # CLS (384) + Mean Patches (384)
    HEAD_HIDDEN_DIM = 256
    SLOT_PRIOR_STRENGTH = 0.55

# Anatomical Slot Prior Knowledge Matrix
# Shape: (12 targets x 6 slots)
SLOT_PRIOR_TABLE = torch.tensor([
    [1.0, 0.4, 0.2, 0.8, 0.3, 0.7],  # ACL
    [0.3, 1.0, 0.3, 0.4, 0.8, 0.3],  # MCL
    [0.9, 0.7, 0.3, 0.7, 0.5, 0.8],  # Medial Meniscus
    [0.9, 0.7, 0.3, 0.7, 0.5, 0.8],  # Lateral Meniscus
    [0.5, 0.9, 0.2, 0.4, 0.8, 0.5],  # Medial OA
    [0.5, 0.9, 0.2, 0.4, 0.8, 0.5],  # Lateral OA
    [0.8, 0.3, 0.9, 0.6, 0.3, 0.6],  # PF OA
    [0.8, 0.8, 0.8, 0.5, 0.3, 0.5],  # Effusion
    [0.8, 0.8, 0.8, 0.5, 0.3, 0.5],  # Synovitis
    [0.9, 0.3, 0.6, 0.8, 0.2, 0.7],  # Baker's
    [0.7, 0.7, 0.7, 0.6, 0.6, 0.6],  # Contusion
    [0.7, 0.7, 0.7, 0.6, 0.6, 0.6],  # Fracture
], dtype=torch.float32)
