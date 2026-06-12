"""
CattleGNN v3: State-of-the-Art Pure GNN for Cattle Muzzle Identification
==========================================================================
All innovations over GNN++ (v2):

1. **GATv2** (Brody et al., 2022) — dynamic attention computed AFTER the
   linear transform + nonlinearity, strictly more expressive than GAT.
2. **Virtual Node** — a global summary node connected to every real node,
   enabling long-range information flow without deep stacking.
3. **GraphNorm** — normalisation with a learnable mean-subtraction weight
   α, more stable than BatchNorm for variable-size graphs.
4. **Edge features as first-class citizens** — 5-d geometric edge attrs
   (dx, dy, dist, angle, rel_scale) encoded via MLP and injected into
   GATv2 message passing.
5. **Multi-scale skip connections** — concatenate representations from
   every GATv2 layer + input projection, then fuse down.
6. **Three-stream pooling** (mean + max + attention-weighted).
7. **Projection head** with LayerNorm + GELU for smoother embedding space.

Architecture:
    Input (256-d DISK descriptors)
      → InputProjection (Linear + GraphNorm + GELU)
      → VirtualNode + GATv2 Layer 1 (8 heads, edge features) + GraphNorm + GELU + Dropout
      → VirtualNode + GATv2 Layer 2 + GraphNorm + GELU + Dropout
      → VirtualNode + GATv2 Layer 3 + GraphNorm + GELU + Dropout
      → Multi-scale Skip Concat [input_proj ‖ layer1 ‖ layer2 ‖ layer3]
      → Fusion Linear (concat_dim → 512)
      → Three-stream Pooling (mean + max + attention) → 1536-d
      → Projection Head → 256-d L2-normalised embedding

Estimated parameters: ~4-6 M  (fits 8 GB VRAM with batch_size=128, ~128 nodes/graph).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


# ---------------------------------------------------------------------------
# GraphNorm  (Xi et al., 2021 — "GraphNorm: A Principled Approach …")
# ---------------------------------------------------------------------------

class GraphNorm(nn.Module):
    """
    Graph-level normalisation with a learnable mean-subtraction weight α.

    Unlike BatchNorm (statistics over the whole batch) or LayerNorm
    (statistics per sample), GraphNorm computes per-graph statistics and
    uses a learnable parameter α ∈ [0, 1] that controls how much of the
    graph mean is subtracted. This stabilises training for variable-size
    graphs that are common in muzzle keypoint data.

    Formula:
        x_out = γ * (x - α * μ_G) / (σ_G + ε) + β

    where μ_G, σ_G are the mean / std computed over all nodes in each
    graph of the batch.
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # Learnable mean-subtraction weight — initialised to 1.0 (full subtract)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        Args:
            x:     (N, D) node features.
            batch: (N,)   graph assignment vector.
        Returns:
            (N, D) normalised features.
        """
        # Force float32 for numerically stable normalisation under AMP
        input_dtype = x.dtype
        x = x.float()

        # Per-graph mean:  mean_G[batch]  ->  (N, D)
        num_graphs = int(batch.max().item()) + 1
        count = torch.zeros(num_graphs, 1, device=x.device, dtype=torch.float32)
        count.scatter_add_(0, batch.unsqueeze(1), torch.ones(x.size(0), 1, device=x.device, dtype=torch.float32))
        count = count.clamp(min=1)

        graph_sum = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=torch.float32)
        graph_sum.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        mean = graph_sum / count                          # (G, D)

        # Subtract learnable fraction of the mean
        x_centered = x - self.alpha.float() * mean[batch]  # (N, D)

        # Per-graph variance
        var_sum = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=torch.float32)
        var_sum.scatter_add_(0, batch.unsqueeze(1).expand_as(x_centered), x_centered.pow(2))
        std = (var_sum / count).sqrt()                    # (G, D)

        x_norm = x_centered / (std[batch] + self.eps)
        result = self.gamma.float() * x_norm + self.beta.float()
        return result.to(input_dtype)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Virtual Node MLP
# ---------------------------------------------------------------------------

class VirtualNodeUpdate(nn.Module):
    """
    MLP that updates the virtual (global summary) node each layer.

    Before each GATv2 layer:
        1.  Aggregate all real-node features (mean) → vnode_agg
        2.  Update:  vnode = MLP(vnode + vnode_agg)           (residual)
        3.  Broadcast:  node_i += vnode[graph_of_i]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        vnode: Tensor,
        batch: Tensor,
        num_graphs: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:          (N, D) node features.
            vnode:      (G, D) virtual-node features.
            batch:      (N,)   graph assignment.
            num_graphs: number of graphs in the batch.
        Returns:
            x_out:   (N, D) node features with virtual-node info added.
            vnode:   (G, D) updated virtual-node features.
        """
        # 1. Aggregate real nodes per graph
        agg = global_mean_pool(x, batch, size=num_graphs)  # (G, D)

        # 2. Residual update of virtual node
        vnode = self.norm(self.mlp(vnode + agg) + vnode)    # (G, D)

        # 3. Broadcast back to all nodes
        x_out = x + vnode[batch]                            # (N, D)
        return x_out, vnode


# ---------------------------------------------------------------------------
# Edge Feature Encoder
# ---------------------------------------------------------------------------

class EdgeEncoder(nn.Module):
    """
    Encode raw 5-d geometric edge attributes into a richer representation
    suitable for GATv2's edge_attr input.

    Expected raw features per edge:
        [dx, dy, dist, angle, rel_scale]
    """

    def __init__(self, raw_edge_dim: int = 5, hidden_dim: int = 64,
                 out_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(raw_edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, edge_attr: Tensor) -> Tensor:
        """(E, raw_dim) → (E, out_dim)."""
        return self.mlp(edge_attr)


# ---------------------------------------------------------------------------
# Attention Pooling (shared pattern from GNN++)
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Attention-weighted global pooling.
    Learns which keypoint nodes are most biometrically discriminative.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.Tanh(),
            nn.Linear(in_dim // 4, 1),
        )

    def forward(self, x: Tensor, batch: Tensor,
                size: Optional[int] = None) -> Tensor:
        """
        Args:
            x:    (N, D) node features.
            batch: (N,) batch assignment.
            size: number of graphs (avoids GPU sync if known).
        Returns:
            (B, D) attention-pooled graph representations.
        """
        if size is None:
            size = int(batch.max().item()) + 1

        # Force float32 for scatter operations (AMP compatibility)
        input_dtype = x.dtype
        x_f32 = x.float()

        scores = self.score(x_f32)                          # (N, 1)
        # Numerically stable per-graph softmax
        max_scores = global_max_pool(scores, batch, size=size)
        scores = scores - max_scores[batch]
        exp_scores = scores.exp()

        sum_exp = torch.zeros(size, 1, device=x.device, dtype=torch.float32)
        sum_exp.scatter_add_(0, batch.unsqueeze(1), exp_scores)
        weights = exp_scores / (sum_exp[batch] + 1e-8)

        weighted = weights * x_f32
        pooled = torch.zeros(size, x_f32.size(1), device=x.device, dtype=torch.float32)
        pooled.scatter_add_(0, batch.unsqueeze(1).expand_as(weighted), weighted)
        return pooled.to(input_dtype)


# ---------------------------------------------------------------------------
# CattleGNNv3  —  main model
# ---------------------------------------------------------------------------

class CattleGNNv3(nn.Module):
    """
    State-of-the-art pure GNN for cattle muzzle biometric identification.

    Architecture:
        DISK Descriptors (256-d)
            → Input Projection (Linear + GraphNorm + GELU)
            → 3× (VirtualNode Update + GATv2Conv + GraphNorm + GELU + Dropout)
            → Multi-scale Skip Concatenation
            → Fusion Linear → 512-d
            → Three-stream Pooling (Mean + Max + Attention) → 1536-d
            → Projection Head → 256-d L2-normalised Embedding
    """

    # Handy constants for dataset variants
    INPUT_DIM_DISK = 256
    INPUT_DIM_SIFT = 256

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        edge_attr_dim: int = 5,
        edge_enc_dim: int = 64,
        fusion_dim: int = 512,
        embedding_dim: int = 256,
        projection_hidden: int = 256,
        dropout: float = 0.15,
    ) -> None:
        """
        Args:
            config:            Config dict (overrides explicit kwargs).
            input_dim:         Input node feature dimension (256 for DISK).
            hidden_dim:        Hidden dimension per GATv2 head output.
            num_heads:         Number of attention heads per GATv2 layer.
            num_layers:        Number of GATv2 layers (default 3).
            edge_attr_dim:     Raw edge feature dimension (5: dx,dy,dist,angle,rel_scale).
            edge_enc_dim:      Encoded edge feature dimension for GATv2.
            fusion_dim:        Dimension after multi-scale skip fusion.
            embedding_dim:     Final L2-normalised embedding dimension.
            projection_hidden: Hidden dim in projection head.
            dropout:           Dropout rate.
        """
        super().__init__()

        # ----- Parse config -----
        if config is not None:
            v3 = config.get('gnn_v3', config.get('model', {}))
            input_dim         = v3.get('input_dim', input_dim)
            hidden_dim        = v3.get('hidden_dim', hidden_dim)
            num_heads         = v3.get('num_heads', num_heads)
            num_layers        = v3.get('num_layers', num_layers)
            edge_attr_dim     = v3.get('edge_attr_dim', edge_attr_dim)
            edge_enc_dim      = v3.get('edge_enc_dim', edge_enc_dim)
            fusion_dim        = v3.get('fusion_dim', fusion_dim)
            embedding_dim     = v3.get('embedding_dim', embedding_dim)
            projection_hidden = v3.get('projection_hidden', projection_hidden)
            dropout           = v3.get('dropout', dropout)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # The GATv2Conv output when concat=True → hidden_dim * num_heads
        head_out = hidden_dim * num_heads  # e.g. 256 × 8 = 2048

        # ----- 1. Input Projection -----
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, head_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # GraphNorm applied separately (needs batch vector)
        self.input_norm = GraphNorm(head_out)

        # ----- 2. Edge Encoder -----
        self.edge_encoder = EdgeEncoder(
            raw_edge_dim=edge_attr_dim,
            hidden_dim=edge_enc_dim,
            out_dim=edge_enc_dim,
        )

        # ----- 3. GATv2 Layers + GraphNorm + VirtualNode -----
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()
        self.vnode_updaters = nn.ModuleList()

        for i in range(num_layers):
            # All layers: head_out → head_out (concat=True throughout for
            # multi-scale skip; we handle the final fusion ourselves).
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=head_out,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,          # → output dim = hidden_dim * num_heads
                    edge_dim=edge_enc_dim,
                    dropout=dropout,
                    add_self_loops=True,
                    bias=True,
                    share_weights=False,   # Full GATv2
                )
            )
            self.gat_norms.append(GraphNorm(head_out))
            self.gat_dropouts.append(nn.Dropout(dropout))
            self.vnode_updaters.append(VirtualNodeUpdate(head_out, dropout=dropout))

        # ----- 4. Multi-scale Skip Fusion -----
        #  Concat: input_proj(head_out) + layer1(head_out) + … + layerL(head_out)
        concat_dim = head_out * (1 + num_layers)
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ----- 5. Three-stream Pooling -----
        self.att_pool = AttentionPooling(fusion_dim)
        pool_dim = fusion_dim * 3  # mean + max + attention

        # ----- 6. Projection Head -----
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

        # ----- ArcFace (set up by training script) -----
        self._num_classes: Optional[int] = None
        self.arcface: Optional[nn.Module] = None

        self._init_weights()

    # -----------------------------------------------------------------
    # Weight initialisation
    # -----------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform for Linear layers, ones/zeros for norms."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, GraphNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    # ArcFace setup
    # -----------------------------------------------------------------

    def set_num_classes(self, num_classes: int) -> None:
        """Attach an ArcFace loss head for classification-based training."""
        from .arcface import ArcFaceLoss

        self._num_classes = num_classes
        self.arcface = ArcFaceLoss(
            embedding_dim=self.embedding_dim,
            num_classes=num_classes,
            margin=0.35,
            scale=48.0,
            triplet_weight=0.25,
            triplet_margin=0.3,
        )
        # Move to same device as model (set_num_classes may be called after .to(device))
        device = next(self.parameters()).device
        self.arcface = self.arcface.to(device)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        data: Any,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            data: PyG Batch/Data object with:
                - x:          (N_total, input_dim) node features
                - edge_index: (2, E) edge connections
                - edge_attr:  (E, 5) geometric edge features  [optional]
                - batch:      (N_total,) batch assignment
                - y:          (B,) labels  [optional, used only if labels=None]
            labels: (B,) explicit labels for ArcFace loss.

        Returns:
            dict:
                - 'embedding':      (B, embedding_dim) L2-normalised
                - 'attention':      list of per-layer attention weight tensors
                - 'node_features':  (N, fusion_dim) fused node features before pooling
                - 'loss':           scalar (if labels provided and ArcFace active)
                - 'stats':          dict   (if labels provided and ArcFace active)
        """
        x: Tensor = data.x
        edge_index: Tensor = data.edge_index
        batch: Tensor = (
            data.batch if (hasattr(data, 'batch') and data.batch is not None)
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        # Graceful dim mismatch handling (same pattern as GNN++)
        if x.size(1) != self.input_dim:
            if x.size(1) < self.input_dim:
                pad = torch.zeros(x.size(0), self.input_dim - x.size(1),
                                  device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, :self.input_dim]

        # Batch size (avoid GPU→CPU sync when possible)
        if hasattr(data, 'num_graphs') and data.num_graphs is not None:
            num_graphs: int = data.num_graphs
        else:
            num_graphs = int(batch.max().item()) + 1

        # --- Encode edge features (if available) ---
        edge_attr_enc: Optional[Tensor] = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            ea = data.edge_attr
            if ea.size(1) < self.edge_encoder.mlp[0].in_features:
                pad_e = torch.zeros(ea.size(0),
                                    self.edge_encoder.mlp[0].in_features - ea.size(1),
                                    device=ea.device, dtype=ea.dtype)
                ea = torch.cat([ea, pad_e], dim=1)
            elif ea.size(1) > self.edge_encoder.mlp[0].in_features:
                ea = ea[:, :self.edge_encoder.mlp[0].in_features]
            edge_attr_enc = self.edge_encoder(ea)

        # ============================================================
        # 1.  Input Projection
        # ============================================================
        x = self.input_proj(x)
        x = self.input_norm(x, batch)

        skip_features: List[Tensor] = [x]  # for multi-scale concat

        # ============================================================
        # 2.  Virtual Node initialisation (zeros)
        # ============================================================
        vnode = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)

        # ============================================================
        # 3.  GATv2 layers with VirtualNode + GraphNorm
        # ============================================================
        all_attention: List[Optional[Tensor]] = []

        for i in range(self.num_layers):
            # a) Virtual node update + broadcast
            x, vnode = self.vnode_updaters[i](x, vnode, batch, num_graphs)

            # b) GATv2 convolution (with edge features)
            x, (_, alpha) = self.gat_layers[i](
                x, edge_index,
                edge_attr=edge_attr_enc,
                return_attention_weights=True,
            )
            all_attention.append(alpha)

            # c) GraphNorm + GELU + Dropout
            x = self.gat_norms[i](x, batch)
            x = F.gelu(x)
            x = self.gat_dropouts[i](x)

            skip_features.append(x)

        # ============================================================
        # 4.  Multi-scale Skip Fusion
        # ============================================================
        x_cat = torch.cat(skip_features, dim=-1)   # (N, head_out * (1+L))
        x = self.fusion(x_cat)                      # (N, fusion_dim)

        node_features = x  # preserve for explainability / keypoint matching

        # ============================================================
        # 5.  Three-stream Global Pooling
        # ============================================================
        x_mean = global_mean_pool(x, batch, size=num_graphs)
        x_max = global_max_pool(x, batch, size=num_graphs)
        x_att = self.att_pool(x, batch, size=num_graphs)

        x_pooled = torch.cat([x_mean, x_max, x_att], dim=-1)  # (B, 3*fusion_dim)

        # ============================================================
        # 6.  Projection Head + L2 normalise
        # ============================================================
        embedding = F.normalize(self.projection_head(x_pooled), p=2, dim=-1)

        result: Dict[str, Any] = {
            'embedding': embedding,
            'attention': all_attention,
            'node_features': node_features,
        }

        # ============================================================
        # 7.  ArcFace loss (training only)
        # ============================================================
        if labels is None and hasattr(data, 'y') and data.y is not None:
            labels = data.y

        if labels is not None and self.arcface is not None:
            loss, stats = self.arcface(embedding, labels)
            result['loss'] = loss
            result['stats'] = stats

        return result

    # -----------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------

    def get_embedding(self, data: Any) -> Tensor:
        """Extract L2-normalised embedding (inference mode)."""
        with torch.no_grad():
            return self.forward(data)['embedding']

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Print a human-readable model summary and return stats dict."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        head_out = self.hidden_dim * self.num_heads

        print(f"\n{'=' * 65}")
        print("CattleGNN v3 -- State-of-the-Art Pure GNN")
        print(f"{'=' * 65}")
        print(f"  Input dim            : {self.input_dim}-d")
        print(f"  GATv2 layers         : {self.num_layers}  (heads={self.num_heads}, "
              f"dim/head={self.hidden_dim}, concat->{head_out})")
        print(f"  Virtual Node         : yes (per-layer MLP update)")
        print(f"  Normalisation        : GraphNorm (learnable alpha)")
        print(f"  Edge features        : 5-d -> {self.edge_encoder.mlp[-1].out_features}-d encoder")
        print(f"  Multi-scale skip     : {1 + self.num_layers} streams -> fusion")
        print(f"  Pooling streams      : 3 (mean + max + attention)")
        print(f"  Embedding dim        : {self.embedding_dim}")
        print(f"  Total parameters     : {total:,}")
        print(f"  Trainable            : {trainable:,}")
        print(f"  Size (fp32)          : {total * 4 / 1e6:.1f} MB")
        if self.arcface is not None:
            print(f"  ArcFace classes      : {self._num_classes}")
        print(f"{'=' * 65}")

        return {
            'architecture': 'CattleGNNv3 (GATv2 + VirtualNode + GraphNorm)',
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'total_parameters': total,
            'trainable_parameters': trainable,
            'parameter_size_mb': total * 4 / 1e6,
        }
