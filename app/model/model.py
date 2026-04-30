"""
model.py — Model definitions and loaders for all cancer types.

  BoneCancerModel  : EfficientNet-B0 backbone (2-class: Cancer / Normal)
  ColonCancerModel : ViT-Base backbone     (5-class: colon & lung subtypes)
"""

import os
import logging

import timm
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Bone Cancer  —  EfficientNet-B0
# ─────────────────────────────────────────────────────────────────────────────

class BoneCancerModel(nn.Module):
    """EfficientNet-B0 fine-tuned for binary bone-cancer classification."""

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def enable_dropout(self):
        """Enable dropout layers at inference time (for MC-Dropout)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def load_bone_model(path: str, device: torch.device) -> BoneCancerModel:
    """Load a BoneCancerModel checkpoint from *path* onto *device*."""
    model = BoneCancerModel(num_classes=2)
    checkpoint = torch.load(path, map_location=device)

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("BoneCancerModel loaded on %s", device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Colon / Lung Cancer  —  ViT-Base
# ─────────────────────────────────────────────────────────────────────────────

class ColonCancerModel(nn.Module):
    """ViT-Base/16 fine-tuned for 5-class colon+lung cancer classification."""

    HIDDEN_DIM = 768

    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=num_classes
        )
        self.backbone.head = nn.Sequential(
            nn.LayerNorm(self.HIDDEN_DIM),
            nn.Dropout(dropout_rate),
            nn.Linear(self.HIDDEN_DIM, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_colon_model(model_path: str, device):
    model = ColonCancerModel()  # adapte selon ton constructeur
    
    state_dict = torch.load(model_path, map_location=device)
    
    # Remap : ajoute le préfixe "backbone." si absent
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("backbone."):
            new_state_dict[f"backbone.{k}"] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
