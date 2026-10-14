"""
Model builder — wraps timm models for FETAL_PLANES_DB.
Supported:
  - efficientnet_b0  (baseline)
  - repvit_m1        (structural reparameterization, CVPR 2024)
  - mobileone_s2     (Apple, reparameterization)

Each model:
  - pretrained on ImageNet
  - classifier head replaced for NUM_CLASSES=6
  - has a .get_features() method to extract penultimate embeddings
"""

import torch
import torch.nn as nn
import timm
from config import NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
class FetalClassifier(nn.Module):
    """
    Thin wrapper around a timm backbone.
    Adds a hook to capture penultimate-layer features for:
      - t-SNE / UMAP visualization
      - Calibration-set selection (K-Means on embeddings)
      - Grad-CAM
    """

    def __init__(self, timm_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.timm_name = timm_name

        # Build backbone with no head so we can inspect embedding dim
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,          # remove original classifier
            global_pool='avg',      # global average pool → 1D embedding
        )
        self.embed_dim = self.backbone.num_features

        # Dropout before classifier (mild regularization)
        self.dropout = nn.Dropout(p=0.2)

        # New classifier head
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        # Hook storage
        self._feature_cache = None
        self._hook_handle    = None

    def forward(self, x):
        feats = self.backbone(x)          # [B, embed_dim]
        feats = self.dropout(feats)
        logits = self.classifier(feats)
        return logits

    def get_features(self, x):
        """Return (logits, embeddings) — useful for feature extraction in one pass."""
        feats  = self.backbone(x)
        feats  = self.dropout(feats)
        logits = self.classifier(feats)
        return logits, feats

    def get_embed_dim(self):
        return self.embed_dim


def build_model(model_name: str, pretrained: bool = True) -> FetalClassifier:
    """
    model_name: one of 'efficientnet_b0', 'repvit_m1', 'mobileone_s2'
    Returns FetalClassifier ready for training.
    """
    timm_name_map = {
        'efficientnet_b0': 'efficientnet_b0',
        'repvit_m1'      : 'repvit_m1',
        'mobileone_s2'   : 'mobileone_s2',
    }
    if model_name not in timm_name_map:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from: {list(timm_name_map.keys())}")

    timm_name = timm_name_map[model_name]
    model = FetalClassifier(timm_name=timm_name, num_classes=NUM_CLASSES, pretrained=pretrained)

    total_params  = sum(p.numel() for p in model.parameters())
    train_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed_dim     = model.get_embed_dim()

    print(f"[Model] {model_name} | timm: {timm_name}")
    print(f"[Model] Total params  : {total_params:,}")
    print(f"[Model] Trainable     : {train_params:,}")
    print(f"[Model] Embed dim     : {embed_dim}")
    return model


if __name__ == "__main__":
    for name in ['efficientnet_b0', 'repvit_m1', 'mobileone_s2']:
        m = build_model(name)
        dummy = torch.randn(2, 3, 224, 224)
        logits, feats = m.get_features(dummy)
        print(f"  logits: {logits.shape}, feats: {feats.shape}\n")
