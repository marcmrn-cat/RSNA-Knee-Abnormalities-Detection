import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.data.dataset import KneeSlotDataset
from src.models.dinov2_slothead import KneeDINOv2Model

def extract_sliding_windows(slots, group_size=Config.GROUP_SIZE):
    B, S, C_slices, C, H, W = slots.shape
    return torch.stack([slots[:, :, i:i+group_size, :, :, :] for i in range(C_slices - group_size + 1)], dim=1)

def apply_target_tta_pooling(window_probs):
    B, W, T = window_probs.shape
    pooled = torch.zeros((B, T), device=window_probs.device)
    for t_idx, target in enumerate(Config.TARGETS):
        t_preds = window_probs[:, :, t_idx]
        if target in ['Fracture', 'Contusion', 'Medial_Meniscus', 'Lateral_Meniscus', 'Bakers']:
            pooled[:, t_idx] = torch.max(t_preds, dim=1)[0]
        elif target in ['ACL', 'MCL']:
            pooled[:, t_idx] = torch.mean(torch.topk(t_preds, k=2, dim=1)[0], dim=1)
        else:
            pooled[:, t_idx] = torch.mean(t_preds, dim=1)
    return pooled

@torch.no_grad()
def predict_with_tta(model, test_loader):
    model.eval()
    preds_list = []
    for batch in test_loader:
        slots, mask = batch['slots'].to(Config.DEVICE), batch['mask'].to(Config.DEVICE)
        windows = extract_sliding_windows(slots)
        B, Num_W = windows.shape[0], windows.shape[1]
        
        w_probs = []
        for w in range(Num_W):
            with torch.cuda.amp.autocast():
                w_probs.append(torch.sigmoid(model(windows[:, w, ...], mask)))
                
        final_probs = apply_target_tta_pooling(torch.stack(w_probs, dim=1))
        preds_list.append(final_probs.cpu().numpy())
    return np.concatenate(preds_list, axis=0)

def main(args):
    test_df = pd.read_csv(args.test_csv)
    series_df = pd.read_csv(args.series_csv)
    test_ds = KneeSlotDataset(test_df, series_df, args.dicom_dir, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    ensemble_preds = []
    for weight_path in args.weights:
        print(f"Loading {weight_path}...")
        model = KneeDINOv2Model().to(Config.DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
        ensemble_preds.append(predict_with_tta(model, test_loader))
        
    # Percentile Rank Fusion
    ranked_preds = [pd.DataFrame(p).rank(pct=True).values for p in ensemble_preds]
    fused_preds = np.mean(ranked_preds, axis=0)
    
    # Format and Validate Output
    sub_df = pd.DataFrame(fused_preds, columns=Config.TARGETS)
    sub_df.insert(0, 'StudyInstanceUID', test_df['StudyInstanceUID'])
    
    assert not sub_df.isnull().values.any(), "Submission contains NULL values!"
    assert len(sub_df) == len(test_df), "Row count mismatch!"
    
    sub_df[Config.TARGETS] = sub_df[Config.TARGETS].clip(0.0, 1.0)
    sub_df.to_csv("submission.csv", index=False)
    print("Inference complete. submission.csv generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--series_csv', type=str, required=True)
    parser.add_argument('--dicom_dir', type=str, required=True)
    parser.add_argument('--weights', nargs='+', required=True)
    main(parser.parse_args())
