"""
Hybrid CNN-GNN Model (Novel Contribution)
==========================================
Combines CNN feature richness with GNN topological invariance.

Key innovation: Instead of handcrafted SIFT/SuperPoint descriptors as GNN
node features, this model uses CNN feature map samples at keypoint locations.

Architecture:
    Preprocessed Image (256×256×3)
        ↓
    EfficientNet-B3 Feature Extractor (shared backbone)
        → Feature map: (B, 1536, H', W')  where H'=W'=8 for 256px input
        ↓
    Bilinear Sampling at Keypoint Locations
        → Per-keypoint CNN features: (N_total, 1536)
        → Feature projection: Linear(1536, 256) + BN + ReLU
        → Node features: (N_total, 256)  ← richer than SIFT!
        ↓
    KNN Graph Structure (from pre-built graphs, same k=8)
        ↓
    3× Dynamic EdgeConv [256, 512, 512]
        ↓
    Topological Relation Module (GAT, 4 heads, 2 layers)
        ↓
    Global Pool (Mean + Max → 512)
        ↓
    Projection Head (512 → 256) + L2 normalize
        ↓
    ArcFace Head (256, num_classes)

Why Feature Map Sampling (not patch crops)?
  - Patch crops would require N separate CNN forward passes per image (expensive)
  - Feature map sampling uses a SINGLE CNN forward pass + bilinear interpolation
  - The feature map captures the full receptive field context around each keypoint
  - This is exactly how modern object detection uses backbone features (e.g., FPN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch_geometric.nn import global_mean_pool, global_max_pool

from .arcface import ArcFaceLoss
from .edge_conv import DynamicEdgeConvBlock
from .trm import TopologicalRelationModule
from .adaptive_graph import AdaptiveGraphConstruction


class HybridCNNGNN(nn.Module):
    """
    Hybrid CNN-GNN model for cattle muzzle biometric identification.

    This is the proposed novel architecture for the research paper.
    It bridges the gap between raw-image CNN approaches and graph-structural GNN
    approaches by using CNN features as graph node inputs.
    """

    def __init__(self, num_classes: int, config: dict = None,
                 embedding_dim: int = 256,
                 edge_conv_dims=None,
                 edge_conv_k: int = 8,
                 trm_heads: int = 4,
                 trm_layers: int = 2,
                 dropout: float = 0.3,
                 pretrained: bool = True,
                 multi_scale: bool = False,
                 ms_stage_indices=(4, 6, 8),
                 learned_edges: bool = False,
                 edge_prune_threshold: float = 0.1):
        """
        Args:
            num_classes: Number of cattle identities.
            config: Full config dict (overrides other args if provided).
            embedding_dim: Output embedding dimension (256).
            edge_conv_dims: Hidden dimensions for EdgeConv blocks.
            edge_conv_k: K for dynamic graph recomputation in EdgeConv.
            trm_heads: Number of GAT attention heads in TRM.
            trm_layers: Number of TRM layers.
            dropout: Dropout rate.
            pretrained: Use ImageNet pretrained EfficientNet-B3.
        """
        super().__init__()

        if config is not None:
            model_cfg = config.get('model', {})
            hybrid_cfg = config.get('hybrid', {})
            ec_cfg = model_cfg.get('edge_conv', {})
            trm_cfg = model_cfg.get('trm', {})
            edge_conv_dims = hybrid_cfg.get('edge_conv_dims', ec_cfg.get('hidden_dims', [256, 512, 512]))
            edge_conv_k = ec_cfg.get('k_dynamic', 8)
            trm_heads = trm_cfg.get('num_heads', 4)
            trm_layers = trm_cfg.get('num_layers', 2)
            dropout = ec_cfg.get('dropout', 0.3)
            embedding_dim = model_cfg.get('embedding_dim', 256)
            hy_cfg = config.get('hybrid', {})
            multi_scale = hy_cfg.get('multi_scale', multi_scale)
            ms_stage_indices = tuple(hy_cfg.get('ms_stage_indices', ms_stage_indices))
            learned_edges = hy_cfg.get('learned_edges', learned_edges)
            edge_prune_threshold = hy_cfg.get('edge_prune_threshold', edge_prune_threshold)

        if edge_conv_dims is None:
            edge_conv_dims = [256, 512, 512]

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.backbone_out_dim = 1536  # EfficientNet-B3 final feature channels

        # ── 1. CNN Backbone (shared feature extractor) ─────────────────────
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        # Keep only the convolutional feature extractor (no global pool, no classifier)
        self.cnn_features = backbone.features   # (B, 1536, H', W')

        # ── Multi-scale sampling config ────────────────────────────────────
        # Sampling from several backbone stages (FPN-style) gives each node both
        # fine groove texture (early, high-res stages) and coarse context (late
        # stages), instead of only the stride-32 final map. Off by default so
        # existing single-scale checkpoints load unchanged.
        self.multi_scale = multi_scale
        self.ms_stage_indices = tuple(sorted(set(ms_stage_indices)))
        # EfficientNet-B3 per-stage output channels (torchvision .features):
        _b3_stage_channels = {0: 40, 1: 24, 2: 32, 3: 48, 4: 96,
                              5: 136, 6: 232, 7: 384, 8: 1536}
        if self.multi_scale:
            node_in_dim = sum(_b3_stage_channels[i] for i in self.ms_stage_indices)
        else:
            node_in_dim = self.backbone_out_dim

        # ── 2. Node Feature Projection ─────────────────────────────────────
        # Projects sampled CNN features → edge_conv_dims[0] for GNN compatibility
        self.node_proj = nn.Sequential(
            nn.Linear(node_in_dim, edge_conv_dims[0]),
            nn.BatchNorm1d(edge_conv_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 2b. Adaptive Graph Construction (learned edge gating) ──────────
        # Replaces the static geometric k-NN topology fed to the relation
        # module with a learned, per-image edge relevance gate.
        self.learned_edges = learned_edges
        if learned_edges:
            edge_feat_dim = 5
            if config is not None:
                edge_feat_dim = config.get('graph', {}).get('edge_feature_dim', 5)
            self.adaptive_graph = AdaptiveGraphConstruction(
                node_dim=edge_conv_dims[0],
                edge_dim=edge_feat_dim,
                prune_threshold=edge_prune_threshold,
            )

        # ── 3. Dynamic EdgeConv Blocks ─────────────────────────────────────
        self.edge_conv = DynamicEdgeConvBlock(
            in_dim=edge_conv_dims[0],
            hidden_dims=edge_conv_dims,
            k=edge_conv_k,
            aggr='max',
            dropout=dropout,
        )

        # ── 4. Topological Relation Module (GAT) ──────────────────────────
        self.trm = TopologicalRelationModule(
            in_dim=edge_conv_dims[-1],
            hidden_dim=256,
            num_heads=trm_heads,
            num_layers=trm_layers,
            dropout=dropout * 0.67,
        )

        # ── 5. Global Pooling → Projection ─────────────────────────────────
        pool_dim = 256 * 2   # mean + max concatenation
        self.projection_head = nn.Sequential(
            nn.Linear(pool_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, embedding_dim),
        )

        # ── 6. ArcFace Loss ────────────────────────────────────────────────
        self.arcface = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=0.5,
            scale=64.0,
            triplet_weight=0.1,
            triplet_margin=0.3,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize non-backbone weights."""
        for m in list(self.node_proj.modules()) + list(self.projection_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_parameter_groups(self, backbone_lr: float = 1e-5, head_lr: float = 1e-3):
        """Differential learning rates: backbone slow, GNN head fast."""
        return [
            {'params': self.cnn_features.parameters(), 'lr': backbone_lr, 'name': 'cnn_backbone'},
            {'params': self.node_proj.parameters(), 'lr': head_lr, 'name': 'node_proj'},
            {'params': self.edge_conv.parameters(), 'lr': head_lr, 'name': 'edge_conv'},
            {'params': self.trm.parameters(), 'lr': head_lr, 'name': 'trm'},
            {'params': self.projection_head.parameters(), 'lr': head_lr, 'name': 'proj_head'},
            {'params': self.arcface.parameters(), 'lr': head_lr, 'name': 'arcface'},
        ]

    def _sample_cnn_features_at_keypoints(self, images: torch.Tensor,
                                            positions: torch.Tensor,
                                            batch_vector: torch.Tensor) -> torch.Tensor:
        """
        Sample CNN feature map at keypoint spatial locations using bilinear interpolation.

        This is the core innovation of the Hybrid model.

        Args:
            images: (B, 3, H, W) input images (whole batch).
            positions: (N_total, 2) normalized keypoint positions [0,1]
                       for all nodes in the batched graph.
            batch_vector: (N_total,) batch assignment for each node.

        Returns:
            node_features: (N_total, 1536) CNN features per keypoint.
        """
        device_type = 'cuda' if images.device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # Run in fp32 for numerical stability in sampling.
            if self.multi_scale:
                feature_maps = self._forward_stages(images.float())  # list of (B,C_s,H_s,W_s)
            else:
                feature_maps = [self.cnn_features(images.float())]   # single (B,1536,H',W')

        return self._sample_maps(feature_maps, positions, batch_vector)

    def _forward_stages(self, images: torch.Tensor):
        """Run the backbone stage-by-stage, returning the selected stage maps."""
        maps = []
        x = images
        for i, stage in enumerate(self.cnn_features):
            x = stage(x)
            if i in self.ms_stage_indices:
                maps.append(x)
        return maps

    def _sample_maps(self, feature_maps, positions: torch.Tensor,
                     batch_vector: torch.Tensor) -> torch.Tensor:
        """Bilinear-sample every feature map at each node position, then concat.

        Args:
            feature_maps: list of (B, C_s, H_s, W_s) maps.
            positions:    (N_total, 2) normalized [0,1] (x, y).
            batch_vector: (N_total,) graph/image assignment.

        Returns:
            (N_total, sum_s C_s) sampled node features.
        """
        B = feature_maps[0].shape[0]
        node_features_list = []
        for b in range(B):
            node_mask = batch_vector == b
            if not node_mask.any():
                continue
            pos_b = positions[node_mask]                 # (N_b, 2) in [0,1]
            grid = (pos_b * 2.0 - 1.0).view(1, 1, -1, 2)  # (1,1,N_b,2) in [-1,1]

            per_scale = []
            for fmap in feature_maps:
                sampled = F.grid_sample(
                    fmap[b:b + 1], grid,
                    mode='bilinear', padding_mode='border', align_corners=True
                )                                        # (1, C_s, 1, N_b)
                C_s = sampled.shape[1]
                per_scale.append(sampled.permute(0, 2, 3, 1).view(-1, C_s))
            node_features_list.append(torch.cat(per_scale, dim=-1))  # (N_b, sum C_s)

        return torch.cat(node_features_list, dim=0)

    def forward(self, images: torch.Tensor, graph_batch,
                labels: torch.Tensor = None) -> dict:
        """
        Forward pass for Hybrid CNN-GNN.

        Args:
            images: (B, 3, H, W) input images (one per graph in batch).
            graph_batch: PyG Batch containing the graph structure.
                         Must have: x (used only for shape), edge_index, batch, pos.
            labels: (B,) ground-truth labels. If None, inference mode.

        Returns:
            dict with 'embedding', and optionally 'loss', 'stats', 'attention'.
        """
        edge_index = graph_batch.edge_index
        batch_vec = graph_batch.batch
        num_nodes = graph_batch.num_nodes

        # ── Step 1: Extract CNN features at keypoint locations ───────────────
        if hasattr(graph_batch, 'pos') and graph_batch.pos is not None:
            positions = graph_batch.pos[:, :2]  # (N_total, 2), normalized
        else:
            # Fallback: use uniform grid positions if pos not stored
            positions = torch.rand(num_nodes, 2, device=images.device)

        # Sample CNN feature map at keypoint positions
        node_feats_raw = self._sample_cnn_features_at_keypoints(
            images, positions, batch_vec
        )  # (N_total, 1536)

        # ── Step 2: Project to GNN-compatible dimension ───────────────────────
        x = self.node_proj(node_feats_raw)   # (N_total, 256)

        # ── Step 2b: Adaptive Graph Construction (optional) ───────────────────
        edge_gate = None
        if self.learned_edges:
            edge_attr = getattr(graph_batch, 'edge_attr', None)
            edge_index, _, edge_gate = self.adaptive_graph(x, edge_index, edge_attr)

        # ── Step 3: Dynamic EdgeConv ──────────────────────────────────────────
        x, _ = self.edge_conv(x, batch=batch_vec)  # (N_total, 512)

        # ── Step 4: Topological Relation Module ───────────────────────────────
        x, attention = self.trm(x, edge_index, batch=batch_vec)  # (N_total, 256)

        # ── Step 5: Global Pooling ────────────────────────────────────────────
        x_mean = global_mean_pool(x, batch_vec)   # (B, 256)
        x_max = global_max_pool(x, batch_vec)     # (B, 256)
        x_pooled = torch.cat([x_mean, x_max], dim=-1)  # (B, 512)

        # ── Step 6: Projection + L2 normalize ────────────────────────────────
        emb = self.projection_head(x_pooled)      # (B, 256)
        embedding = F.normalize(emb, p=2, dim=-1)

        result = {'embedding': embedding, 'attention': attention,
                  'edge_gate': edge_gate}

        if labels is not None:
            loss, stats = self.arcface(embedding, labels)
            result['loss'] = loss
            result['stats'] = stats

        return result

    def get_embedding(self, images: torch.Tensor, graph_batch) -> torch.Tensor:
        """Inference: extract embedding without computing loss."""
        with torch.no_grad():
            return self.forward(images, graph_batch)['embedding']

    def summary(self):
        """Return a dict of parameter counts split across the CNN backbone and the
        graph head (node_proj + edge_conv + trm + projection_head)."""
        cnn_params = sum(p.numel() for p in self.cnn_features.parameters())
        gnn_params = (sum(p.numel() for p in self.node_proj.parameters()) +
                      sum(p.numel() for p in self.edge_conv.parameters()) +
                      sum(p.numel() for p in self.trm.parameters()) +
                      sum(p.numel() for p in self.projection_head.parameters()))
        arc_params = sum(p.numel() for p in self.arcface.parameters())
        total = cnn_params + gnn_params + arc_params

        print(f"\n{'=' * 60}")
        print("Hybrid CNN-GNN Model Summary")
        print(f"{'=' * 60}")
        print(f"  CNN Backbone (EfficientNet-B3): {cnn_params:,} params")
        print(f"  GNN Head (EdgeConv + TRM):      {gnn_params:,} params")
        print(f"  ArcFace Head:                   {arc_params:,} params")
        print(f"  Total Parameters:               {total:,}")
        print(f"  Embedding Dimension:            {self.embedding_dim}")
        print(f"  Num Classes:                    {self.num_classes}")
        print(f"  Innovation: CNN feature map sampling at keypoint locations")
        print(f"{'=' * 60}")
        return {
            'architecture': 'Hybrid CNN-GNN (EfficientNet-B3 + EdgeConv + TRM + ArcFace)',
            'total_parameters': total,
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
        }
