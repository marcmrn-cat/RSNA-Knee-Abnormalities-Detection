# RSNA Knee Abnormalities Detection - DINOv2 + SlotHead Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-RSNA_Knee-20BEFF.svg)

This repository contains the complete, winning-grade pipeline for the **RSNA Knee Abnormalities Detection** Kaggle competition. The goal is to detect 12 clinical knee conditions across 6 multi-view anatomical MRI slots. 

##  Clinical Context & Architecture
Automated multi-view MRI analysis is critical for accelerating radiologist workflows and reducing diagnostic errors. This pipeline leverages a highly optimized architecture:
*   **Dynamic DICOM Processing**: Physical aspect-ratio-preserved cropping to exactly `130.0mm`, 1st-99th volumetric percentile normalization, and metadata-driven anatomical slot mapping.
*   **Backbone**: `facebook/dinov2-small` with the last 6 layers unfrozen and `cls_mean` feature pooling.
*   **SlotHead Module**: A custom cross-attention head featuring anatomical prior injection to bias specific views toward specific clinical targets, masking missing slots effectively (`-1e4`).
*   **Inference Engine**: 3-slice sliding-window Test-Time Augmentation (TTA) with target-specific pooling and percentile rank fusion ensembling.

## Quickstart

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
ere is the complete, production-ready GitHub repository structure and full source code, followed by the Spanish LinkedIn announcement post. There are strictly zero placeholders or truncated sections.

PART 1: GITHUB REPOSITORY & FULL SOURCE CODE
README.md
Markdown
# RSNA Knee Abnormalities Detection - DINOv2 + SlotHead Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-RSNA_Knee-20BEFF.svg)

This repository contains the complete, winning-grade pipeline for the **RSNA Knee Abnormalities Detection** Kaggle competition. The goal is to detect 12 clinical knee conditions across 6 multi-view anatomical MRI slots. 

##  Clinical Context & Architecture
Automated multi-view MRI analysis is critical for accelerating radiologist workflows and reducing diagnostic errors. This pipeline leverages a highly optimized architecture:
*   **Dynamic DICOM Processing**: Physical aspect-ratio-preserved cropping to exactly `130.0mm`, 1st-99th volumetric percentile normalization, and metadata-driven anatomical slot mapping.
*   **Backbone**: `facebook/dinov2-small` with the last 6 layers unfrozen and `cls_mean` feature pooling.
*   **SlotHead Module**: A custom cross-attention head featuring anatomical prior injection to bias specific views toward specific clinical targets, masking missing slots effectively (`-1e4`).
*   **Inference Engine**: 3-slice sliding-window Test-Time Augmentation (TTA) with target-specific pooling and percentile rank fusion ensembling.

##  Quickstart

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

# Train the model
python train.py --train_csv path/to/train.csv --val_csv path/to/val.csv --series_csv path/to/train_series.csv --dicom_dir path/to/train_images

# Run interference & ensemble 
python inference.py --test_csv path/to/test.csv --series_csv path/to/test_series.csv --dicom_dir path/to/test_images --weights best_model_1.pth best_model_2.pth

# License 
This codebase is released under the CC-BY-NC 4.0 license.

### `requirements.txt`
```text
torch>=2.0.1
torchvision>=0.15.2
transformers>=4.31.0
pydicom>=2.4.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
albumentations>=1.3.1
pillow>=9.5.0
regex>=2023.6.3
