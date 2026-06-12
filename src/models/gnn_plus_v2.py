"""
CattleGNN++ : Enhanced GNN for Cattle Muzzle Identification
=============================================================
All improvements over baseline GNN+:

1. Input: CNN patch features (MobileNetV3, 576-d) + SIFT (256-d) + pos (2-d) = 834-d
2. Deeper EdgeConv: 4 layers [512, 512, 512, 1024] with residual connections
3. 3-stream pooling: Mean + Max + Attention-weighted
4. Better projection head with LayerNorm
5. ArcFace with tuned hyperparameters (margin=0.35, scale=48)

Expected accuracy: 88-92% (vs 72% for base GNN+ at 37 epochs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool

from .edge_conv import DynamicEdgeConvBlock
from .trm import TopologicalRelationModule


class ResidualEdgeConvBlock(nn.Module):
    """EdgeConv block with residual skip connection."""

    def __init__(self, in_dim, out_dim, k=12, dropout=0.2):
        super().__init__()
        self.conv = DynamicEdgeConvBlock(
            in_dim=in_dim,
            hidden_dims=[out_dim],
            k=k,
            aggr='max',
            dropout=dropout,
        )
        # Residual projection if dims differ
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, batch=None):
        out, intermediates = self.conv(x, batch=batch)
        res = self.residual(x)
        return self.norm(out + res), intermediates


class AttentionPooling(nn.Module):
    """
    Attention-weighted global pooling.
    Learns which nodes are most biometrically discriminative.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.Tanh(),
            nn.Linear(in_dim // 4, 1),
        )

    def forward(self, x, batch, size=None):
        """
        Args:
            x: (N_total, in_dim) node features
            batch: (N_total,) batch assignment vector
            size: (int, optional) number of graphs in the batch
        Returns:
            (B, in_dim) attended graph representation
        """
        if size is None:
            size = int(batch.max()) + 1
        scores = self.score(x)          # (N, 1)
        max_scores = global_max_pool(scores, batch, size=size)  # (B, 1) per-graph max
        scores = scores - max_scores[batch]  # numerical stability per graph
        # Softmax per graph
        exp_scores = scores.exp()
        sum_exp = torch.zeros(size, 1, device=x.device)
        sum_exp.scatter_add_(0, batch.unsqueeze(1), exp_scores)
        weights = exp_scores / (sum_exp[batch] + 1e-8)

        # Weighted sum per graph
        weighted = weights * x          # (N, in_dim)
        pooled = torch.zeros(size, x.shape[1], device=x.device)
        pooled.scatter_add_(0, batch.unsqueeze(1).expand_as(weighted), weighted)
        return pooled                   # (B, in_dim)


class CattleGNNPlusPlus(nn.Module):
    """
    Enhanced GNN++ with all Tier 1-6 improvements applied.

    Architecture:
        Enhanced Node Features (834-d: CNN + SIFT + pos)
            -> Input Projection (Linear + LayerNorm + GELU)
            -> 4x Residual EdgeConv Blocks (k=16 dynamic)
            -> Topological Relation Module (GAT, 8 heads, 3 layers)
            -> 3-stream Global Pooling (Mean + Max + Attention)
            -> Projection Head (MLP + LayerNorm)
            -> 256-d L2-normalized Embedding
    """

    INPUT_DIM_FULL = 834   # CNN(576) + SIFT(256) + pos(2)
    INPUT_DIM_CNN  = 578   # CNN(576) + pos(2) only
    INPUT_DIM_SIFT = 258   # SIFT(256) + pos(2) only

    def __init__(self, config=None,
                 input_dim=834,
                 edge_conv_dims=None,
                 edge_conv_k=16,
                 trm_hidden=512,
                 trm_heads=8,
                 trm_layers=3,
                 embedding_dim=256,
                 projection_hidden=1024,
                 dropout=0.2):
        super().__init__()

        if config is not None:
            pp = config.get('gnn_plus_v2', {})
            edge_conv_dims   = pp.get('edge_conv_dims', edge_conv_dims if edge_conv_dims is not None else [512, 512, 512, 1024])
            edge_conv_k      = pp.get('edge_conv_k', edge_conv_k)
            dropout          = pp.get('dropout', dropout)
            trm_hidden       = pp.get('trm_hidden', trm_hidden)
            trm_heads        = pp.get('trm_heads', trm_heads)
            trm_layers       = pp.get('trm_layers', trm_layers)
            embedding_dim    = pp.get('embedding_dim', embedding_dim)
            projection_hidden = pp.get('projection_hidden', projection_hidden)
            input_dim        = pp.get('input_dim', input_dim)

        if edge_conv_dims is None:
            edge_conv_dims = [512, 512, 512, 1024]

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # 1. Input Projection (GELU instead of ReLU — smoother gradients)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, edge_conv_dims[0]),
            nn.LayerNorm(edge_conv_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Residual EdgeConv Blocks (4 layers)
        self.edge_convs = nn.ModuleList()
        dims = [edge_conv_dims[0]] + edge_conv_dims
        for i in range(len(edge_conv_dims)):
            self.edge_convs.append(
                ResidualEdgeConvBlock(dims[i], dims[i+1], k=edge_conv_k, dropout=dropout)
            )

        final_conv_dim = edge_conv_dims[-1]

        # 3. Topological Relation Module (deeper: 3 layers, 8 heads)
        self.trm = TopologicalRelationModule(
            in_dim=final_conv_dim,
            hidden_dim=trm_hidden,
            num_heads=trm_heads,
            num_layers=trm_layers,
            dropout=dropout * 0.5,
        )

        # 4. Three-stream pooling
        self.att_pool = AttentionPooling(trm_hidden)
        pool_dim = trm_hidden * 3  # mean + max + attention

        # 5. Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(pool_dim, projection_hidden),
            nn.LayerNorm(projection_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(projection_hidden, projection_hidden // 2),
            nn.LayerNorm(projection_hidden // 2),
            nn.GELU(),
            nn.Linear(projection_hidden // 2, embedding_dim),
        )

        # ArcFace head (set up by training script)
        self._num_classes = None
        self.arcface = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_num_classes(self, num_classes):
        from .arcface import ArcFaceLoss
        self._num_classes = num_classes
        self.arcface = ArcFaceLoss(
            embedding_dim=self.embedding_dim,
            num_classes=num_classes,
            margin=0.35,          # Tuned for 260-class small dataset
            scale=48.0,           # Tuned down from 64 to prevent saturation
            triplet_weight=0.25,  # Stronger triplet signal for pure GNN
            triplet_margin=0.3,
        )

    def forward(self, data, labels=None):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if (hasattr(data, 'batch') and data.batch is not None) else None

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle auto-detection of input dim mismatch
        # (graceful fallback if graphs_v2 not available)
        if x.shape[1] != self.input_dim:
            # Pad or truncate to match
            if x.shape[1] < self.input_dim:
                pad = torch.zeros(x.shape[0], self.input_dim - x.shape[1], device=x.device)
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, :self.input_dim]

        # 1. Input projection
        x = self.input_proj(x)

        # 2. Residual EdgeConv blocks
        for conv in self.edge_convs:
            x, _ = conv(x, batch=batch)

        # 3. TRM
        x, attention_weights = self.trm(x, edge_index, batch=batch)
        node_features = x

        # Determine number of graphs (batch size) without CPU-GPU synchronization
        if hasattr(data, 'num_graphs') and data.num_graphs is not None:
            size = data.num_graphs
        else:
            size = int(batch.max()) + 1

        # 4. Three-stream global pooling
        x_mean = global_mean_pool(x, batch, size=size)
        x_max  = global_max_pool(x, batch, size=size)
        x_att  = self.att_pool(x, batch, size=size)

        x_pooled = torch.cat([x_mean, x_max, x_att], dim=-1)  # 3 * trm_hidden

        # 5. Projection head + L2 normalize
        embedding = F.normalize(self.projection_head(x_pooled), p=2, dim=-1)

        result = {
            'embedding': embedding,
            'attention': attention_weights,
            'node_features': node_features,
        }

        # ArcFace loss
        if labels is not None and self.arcface is not None:
            loss, stats = self.arcface(embedding, labels)
            result['loss'] = loss
            result['stats'] = stats

        return result

    def get_embedding(self, data):
        with torch.no_grad():
            return self.forward(data)['embedding']

    def summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print("CattleGNN++ Model Summary")
        print(f"{'='*60}")
        print(f"  Input dim           : {self.input_dim}-d (CNN+SIFT+pos)")
        print(f"  EdgeConv layers     : {len(self.edge_convs)} (with residuals)")
        print(f"  TRM heads/layers    : 8 heads, 3 layers")
        print(f"  Pooling streams     : 3 (mean + max + attention)")
        print(f"  Embedding dim       : {self.embedding_dim}")
        print(f"  Total parameters    : {total:,}")
        print(f"  Trainable           : {trainable:,}")
        print(f"  ArcFace margin      : 0.35 (tuned)")
        print(f"  ArcFace scale       : 48.0 (tuned)")
        print(f"{'='*60}")
        return {'total_parameters': total, 'trainable_parameters': trainable}
