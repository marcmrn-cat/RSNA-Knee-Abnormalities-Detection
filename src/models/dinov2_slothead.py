import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.config import Config, SLOT_PRIOR_TABLE

class SlotHead(nn.Module):
    def __init__(self, num_targets: int = 12, num_slots: int = 6, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.num_targets = num_targets
        self.num_slots = num_slots
        
        self.target_queries = nn.Parameter(torch.randn(num_targets, hidden_dim))
        self.slot_proj = nn.Linear(embed_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.register_buffer('slot_prior', SLOT_PRIOR_TABLE)
        self.prior_strength = Config.SLOT_PRIOR_STRENGTH
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_targets)
        ])

    def forward(self, slot_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = slot_features.size(0)
        H_slots = self.slot_proj(slot_features)
        
        Q = self.query_proj(self.target_queries).unsqueeze(0).expand(B, -1, -1)
        K = self.key_proj(H_slots)
        V = self.value_proj(H_slots)
        
        att = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        
        # Inject Anatomical Prior
        att = att + self.prior_strength * self.slot_prior.unsqueeze(0)
        
        # Mask Missing Slots
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4)
        
        att_weights = F.softmax(att, dim=-1)
        context = torch.matmul(att_weights, V)
        
        logits = [self.classifiers[i](context[:, i, :]) for i in range(self.num_targets)]
        return torch.cat(logits, dim=-1)

class KneeDINOv2Model(nn.Module):
    def __init__(self, unfreeze_last: int = Config.UNFREEZE_LAST):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        if unfreeze_last > 0:
            for layer in self.backbone.encoder.layer[-unfreeze_last:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.slot_head = SlotHead(
            num_targets=len(Config.TARGETS), num_slots=len(Config.SLOTS),
            embed_dim=Config.SLOT_EMBED_DIM, hidden_dim=Config.HEAD_HIDDEN_DIM
        )

    def extract_slice_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        patch_mean = last_hidden_state[:, 1:, :].mean(dim=1)
        return torch.cat([cls_token, patch_mean], dim=-1)

    def forward(self, slots: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, C_slices, C, H, W = slots.shape
        flat_slots = slots.view(B * S * C_slices, C, H, W)
        flat_features = self.extract_slice_features(flat_slots) 
        
        features = flat_features.view(B, S, C_slices, Config.SLOT_EMBED_DIM)
        slot_features = features.mean(dim=2) 
        return self.slot_head(slot_features, mask)
