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
