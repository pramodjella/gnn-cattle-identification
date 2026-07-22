"""
CNN Baseline Model: EfficientNet-B4 + ArcFace  [TUNED for 98%+]
================================================================
Upgraded from B3→B4 for higher feature capacity.
Embedding expanded 256→512 for richer metric learning space.
Label smoothing added to ArcFace for better calibration.

Architecture:
    Preprocessed Muzzle Image (256×256×3)
        → EfficientNet-B4 Backbone (ImageNet pretrained, fine-tuned)
        → Global Average Pool → 1792-d
        → Embedding Head: Dropout(0.35)
                         → Linear(1792, 1024) + BN + GELU
                         → Dropout(0.175)
                         → Linear(1024, 512) + BN + GELU
                         → Linear(512, 512) + L2-normalize
        → ArcFace Head (512, num_classes, margin=0.45, scale=96)

Training uses differential learning rates:
  - Backbone: 3e-5 (fine-tuning, more aggressive than B3)
  - Embedding head + ArcFace: 1e-3 (training from scratch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b4, EfficientNet_B4_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
)

from .arcface import ArcFaceLoss


class CNNMuzzleModel(nn.Module):
    """
    EfficientNet-B4 based cattle muzzle identification model.

    Upgraded from B3 for higher feature capacity (1792-d vs 1536-d).
    Embedding dimension increased to 512 for richer metric learning.
    """

    BACKBONE_CONFIGS = {
        'efficientnet_b3': {'dim': 1536, 'weights': EfficientNet_B3_Weights.IMAGENET1K_V1, 'fn': efficientnet_b3},
        'efficientnet_b4': {'dim': 1792, 'weights': EfficientNet_B4_Weights.IMAGENET1K_V1, 'fn': efficientnet_b4},
    }

    def __init__(self, num_classes: int, embedding_dim: int = 512,
                 dropout: float = 0.35, pretrained: bool = True,
                 backbone: str = 'efficientnet_b4',
                 arcface_scale: float = 96.0, arcface_margin: float = 0.45,
                 label_smoothing: float = 0.1):
        """
        Args:
            num_classes: Number of cattle identities (260).
            embedding_dim: Output embedding dimension (512 for tuned version).
            dropout: Dropout rate in embedding head.
            pretrained: Use ImageNet pretrained weights.
            backbone: Backbone variant ('efficientnet_b4' or 'efficientnet_b3').
            arcface_scale: Feature scale for ArcFace (96 for tuned version).
            arcface_margin: Angular margin for ArcFace (0.45 for tuned version).
            label_smoothing: Label smoothing epsilon for cross-entropy.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.backbone_name = backbone

        # ── Backbone ──────────────────────────────────────────────────────────
        cfg = self.BACKBONE_CONFIGS.get(backbone, self.BACKBONE_CONFIGS['efficientnet_b4'])
        weights = cfg['weights'] if pretrained else None
        net = cfg['fn'](weights=weights)

        self.features = net.features
        self.avgpool = net.avgpool
        self.backbone_out_dim = cfg['dim']

        # ── Embedding Head ─────────────────────────────────────────────────
        # 3-layer bottleneck: backbone_dim → 1024 → 512 → embedding_dim
        self.embedding_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone_out_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(512, embedding_dim),
        )

        # ── ArcFace Loss (training) ────────────────────────────────────────
        self.arcface = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale,
            triplet_weight=0.1,
            triplet_margin=0.3,
            label_smoothing=label_smoothing,
        )

        self._init_head()

    def _init_head(self):
        """Initialize embedding head weights."""
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_parameter_groups(self, backbone_lr: float = 3e-5, head_lr: float = 1e-3):
        """
        Return parameter groups with differential learning rates.
        B4 backbone uses 3e-5 (higher than B3's 1e-5) for faster feature adaptation.
        """
        return [
            {'params': self.features.parameters(), 'lr': backbone_lr, 'name': 'backbone'},
            {'params': self.avgpool.parameters(), 'lr': backbone_lr, 'name': 'avgpool'},
            {'params': self.embedding_head.parameters(), 'lr': head_lr, 'name': 'head'},
            {'params': self.arcface.parameters(), 'lr': head_lr, 'name': 'arcface'},
        ]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features (before embedding head)."""
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)   # (B, backbone_out_dim)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized embedding (inference mode).
        No ArcFace head — just the backbone + embedding MLP.
        """
        feat = self.extract_features(x)
        emb = self.embedding_head(feat)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) input images.
            labels: (B,) ground-truth class indices. If None, inference mode.

        Returns:
            dict with:
                'embedding': (B, embedding_dim) L2-normalized embedding
                'loss': scalar loss (only if labels provided)
                'stats': training statistics dict (only if labels provided)
        """
        feat = self.extract_features(x)
        emb = self.embedding_head(feat)
        embedding = F.normalize(emb, p=2, dim=-1)

        result = {'embedding': embedding}

        if labels is not None:
            loss, stats = self.arcface(embedding, labels)
            result['loss'] = loss
            result['stats'] = stats

        return result

    def summary(self):
        """Return a dict of parameter counts (total / trainable / backbone / head)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.features.parameters())
        head_params = sum(p.numel() for p in self.embedding_head.parameters())
        arcface_params = sum(p.numel() for p in self.arcface.parameters())

        print(f"\n{'=' * 60}")
        print(f"CNN Model Summary ({self.backbone_name} + ArcFace) [TUNED]")
        print(f"{'=' * 60}")
        print(f"  Backbone ({self.backbone_name}): {backbone_params:,} params | {self.backbone_out_dim}-d features")
        print(f"  Embedding Head:   {head_params:,} params -> {self.embedding_dim}-d")
        print(f"  ArcFace Head:     {arcface_params:,} params | {self.num_classes} classes")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable:        {trainable:,}")
        print(f"  Parameter Size:   {total_params * 4 / 1e6:.1f} MB (fp32)")
        print(f"{'=' * 60}")
        return {
            'architecture': f'CNN ({self.backbone_name} + ArcFace)',
            'total_parameters': total_params,
            'trainable_parameters': trainable,
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
            'backbone': self.backbone_name,
            'backbone_out_dim': self.backbone_out_dim,
        }
