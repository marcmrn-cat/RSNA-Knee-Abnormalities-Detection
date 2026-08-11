import os
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from src.config import Config

class KneeSlotDataset(Dataset):
    def __init__(self, df: pd.DataFrame, series_df: pd.DataFrame, dicom_dir: str, is_train: bool = True):
        self.df = df
        self.series_df = series_df
        self.dicom_dir = dicom_dir
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _get_slot_mapping(self, study_id: str) -> Dict[str, str]:
        study_series = self.series_df[self.series_df['StudyInstanceUID'] == study_id]
        mapping = {}
        
        def find_best_series(plane: str, fluid: int, fs: int) -> Optional[str]:
            matches = study_series[
                (study_series['Anatomical_Plane'].str.upper() == plane.upper()) &
                (study_series['Fluid_Sensitive'] == fluid) &
                (study_series['Fat_Suppression'] == fs)
            ]
            if not matches.empty:
                return matches.loc[matches['SliceCount'].idxmax(), 'SeriesInstanceUID']
            return None

        mapping['SAG_FLUID_FS'] = find_best_series('SAGITTAL', 1, 1)
        mapping['COR_FLUID_FS'] = find_best_series('CORONAL', 1, 1)
        mapping['AX_FLUID_FS'] = find_best_series('AXIAL', 1, 1)
        mapping['SAG_FLUID_NOFS'] = find_best_series('SAGITTAL', 1, 0)
        mapping['COR_T1'] = find_best_series('CORONAL', 0, 0)
        mapping['SAG_T1'] = find_best_series('SAGITTAL', 0, 0)
        return mapping

    def _process_dicom_series(self, slice_paths: List[str], plane: str, laterality: str) -> torch.Tensor:
        n_slices = len(slice_paths)
        if n_slices == 0:
            return torch.zeros((Config.CACHE_SLICES, 3, *Config.IMAGE_SIZE))

        # 1. Read entire 3D volume stack for accurate percentiles
        volume_arrays, dicoms = [], []
        for path in slice_paths:
            dcm = pydicom.dcmread(path)
            pixel_array = dcm.pixel_array.astype(np.float32)
            slope = getattr(dcm, 'RescaleSlope', 1.0)
            intercept = getattr(dcm, 'RescaleIntercept', 0.0)
            volume_arrays.append(pixel_array * slope + intercept)
            dicoms.append(dcm)
            
        volume = np.stack(volume_arrays)
        p1, p99 = np.percentile(volume, 1), np.percentile(volume, 99)

        # 2. Slice Banding (20-80%)
        low_idx = int(n_slices * Config.SLICE_BAND[0])
        high_idx = int(n_slices * Config.SLICE_BAND[1]) - 1
        selected_indices = np.linspace(low_idx, max(low_idx, high_idx), Config.CACHE_SLICES, dtype=int)
        
        processed_slices = []
        for idx in selected_indices:
            pixel_array = volume[idx]
            dcm = dicoms[idx]
            
            # 3. Volumetric Normalization
            if p99 > p1:
                pixel_array = np.clip(pixel_array, p1, p99)
                pixel_array = (pixel_array - p1) / (p99 - p1)
            else:
                pixel_array = np.zeros_like(pixel_array)

            img_tensor = torch.from_numpy(pixel_array).unsqueeze(0)
            
            # 4. Aspect-Ratio-Preserving Physical Crop (130.0mm)
            pixel_spacing = getattr(dcm, 'PixelSpacing', [1.0, 1.0])
            row_spacing, col_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])
            crop_h = int(round(Config.PHYSICAL_CROP_MM / row_spacing))
            crop_w = int(round(Config.PHYSICAL_CROP_MM / col_spacing))
            
            _, h, w = img_tensor.shape
            start_h = max(0, (h - crop_h) // 2)
            start_w = max(0, (w - crop_w) // 2)
            cropped_img = img_tensor[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
            
            # Pad to square
            _, ch, cw = cropped_img.shape
            max_dim = max(ch, cw)
            pad_h, pad_w = max_dim - ch, max_dim - cw
            pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            padded_img = F.pad(cropped_img, pad, mode='constant', value=0.0)
            
            # Interpolate to 336x336
            resized_img = F.interpolate(
                padded_img.unsqueeze(0), size=Config.IMAGE_SIZE, mode='bilinear', align_corners=False
            ).squeeze(0)
            
            # 5. Flip Laterality for AX/COR planes
            if laterality.upper() == 'R' and plane.upper() in ['COR', 'AX', 'CORONAL', 'AXIAL']:
                resized_img = torch.flip(resized_img, dims=[-1])
                
            processed_slices.append(resized_img.repeat(3, 1, 1))
            
        return torch.stack(processed_slices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        study_id = row['StudyInstanceUID']
        laterality = row.get('Laterality', 'L')
        
        slot_mapping = self._get_slot_mapping(study_id)
        slot_tensors, presence_mask = [], []

        for slot_name in Config.SLOTS:
            series_id = slot_mapping.get(slot_name)
            if series_id:
                series_dir = os.path.join(self.dicom_dir, str(study_id), str(series_id))
                slice_files = sorted([os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith('.dcm')])
                plane = slot_name.split('_')[0]
                imgs = self._process_dicom_series(slice_files, plane, laterality)
                slot_tensors.append(imgs)
                presence_mask.append(1.0)
            else:
                slot_tensors.append(torch.zeros((Config.CACHE_SLICES, 3, *Config.IMAGE_SIZE)))
                presence_mask.append(0.0)

        out_dict = {
            'slots': torch.stack(slot_tensors),
            'mask': torch.tensor(presence_mask, dtype=torch.float32),
            'study_id': study_id
        }

        if self.is_train:
            labels = row[Config.TARGETS].values.astype(np.float32)
            out_dict['labels'] = torch.tensor(labels, dtype=torch.float32)

        return out_dict
