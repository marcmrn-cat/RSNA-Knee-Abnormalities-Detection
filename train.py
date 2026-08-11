import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import Config
from src.data.dataset import KneeSlotDataset
from src.models.dinov2_slothead import KneeDINOv2Model
from src.utils.nlp_parser import apply_pseudo_labels

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, scaler):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        slots = batch['slots'].to(Config.DEVICE)
        mask = batch['mask'].to(Config.DEVICE)
        labels = batch['labels'].to(Config.DEVICE)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(slots, mask)
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item() * slots.size(0)
    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    for batch in dataloader:
        slots = batch['slots'].to(Config.DEVICE)
        mask = batch['mask'].to(Config.DEVICE)
        labels = batch['labels'].to(Config.DEVICE)
        with torch.cuda.amp.autocast():
            logits = model(slots, mask)
            loss = criterion(logits, labels)
        running_loss += loss.item() * slots.size(0)
    return running_loss / len(dataloader.dataset)

def main(args):
    train_df = apply_pseudo_labels(pd.read_csv(args.train_csv))
    val_df = apply_pseudo_labels(pd.read_csv(args.val_csv))
    series_df = pd.read_csv(args.series_csv)
    
    train_ds = KneeSlotDataset(train_df, series_df, args.dicom_dir, is_train=True)
    val_ds = KneeSlotDataset(val_df, series_df, args.dicom_dir, is_train=True)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    model = KneeDINOv2Model().to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.MAX_LR, steps_per_epoch=len(train_loader), epochs=Config.EPOCHS
    )
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    for epoch in range(Config.EPOCHS):
        t_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler)
        v_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1:02d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Checkpoint Saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--series_csv', type=str, required=True)
    parser.add_argument('--dicom_dir', type=str, required=True)
    main(parser.parse_args())
