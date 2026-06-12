"""
VisGIN: Visibility Graph Neural Network for Cattle Muzzle Identification
=========================================================================
Implements a GNN with custom visibility-based attention, modulating edge 
attentions by the spatial 2D Euclidean distance between keypoint coordinates.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, global_mean_pool, global_max_pool
from src.models.gnn_v3 import GraphNorm, EdgeEncoder, AttentionPooling

class VisibilityGATConv(MessagePassing):
    """
    Visibility Graph Attention Layer.
    Computes dot-product attention modulated by physical Euclidean distance
    and a learnable head-specific spatial decay parameter.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 concat: bool = True, dropout: float = 0.0) -> None:
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_k = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_v = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Spatial decay parameter: larger gamma -> more localized attention
        self.gamma = nn.Parameter(torch.ones(heads, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.constant_(self.gamma, 1.0)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        # q, k, v: [N, heads, out_channels]
        q = self.lin_q(x).view(-1, self.heads, self.out_channels)
        k = self.lin_k(x).view(-1, self.heads, self.out_channels)
        v = self.lin_v(x).view(-1, self.heads, self.out_channels)

        # Propagate messages
        out = self.propagate(edge_index, q=q, k=k, v=v, pos=pos, size=None)

        if self.concat:
            return out.view(-1, self.heads * self.out_channels)
        else:
            return out.mean(dim=1)

    def message(self, q_i: Tensor, k_j: Tensor, v_j: Tensor, 
                pos_i: Tensor, pos_j: Tensor, 
                index: Tensor, ptr: Tensor, size_i: int) -> Tensor:
        # q_i: [E, heads, out_channels] (queries of target nodes)
        # k_j: [E, heads, out_channels] (keys of source nodes)
        # v_j: [E, heads, out_channels] (values of source nodes)
        # pos_i: [E, 2], pos_j: [E, 2] (node coordinates)

        # Standard dot-product attention
        attn = (q_i * k_j).sum(dim=-1) / math.sqrt(self.out_channels)

        # Physical 2D distance squared: [E, 1]
        dist_sq = (pos_i - pos_j).pow(2).sum(dim=-1, keepdim=True)

        # Learnable spatial modulation
        gamma = self.gamma.view(1, -1).clamp(min=1e-4) # ensure non-negative decay
        spatial_decay = gamma * dist_sq # [E, heads]

        # Modulate logit by distance decay
        attn_logits = attn - spatial_decay

        # Softmax over neighborhood
        alpha = softmax(attn_logits, index, ptr, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return v_j * alpha.unsqueeze(-1)


class CattleVisGIN(nn.Module):
    """
    Visibility Graph Neural Network for Cattle Identification.
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        fusion_dim: int = 512,
        embedding_dim: int = 256,
        projection_hidden: int = 256,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        head_out = hidden_dim * num_heads

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, head_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_norm = GraphNorm(head_out)

        # VisGIN Spatial Attention Layers
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(
                VisibilityGATConv(
                    in_channels=head_out,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            self.norms.append(GraphNorm(head_out))
            self.dropouts.append(nn.Dropout(dropout))

        # Multi-scale Skip Fusion
        concat_dim = head_out * (1 + num_layers)
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Three-stream pooling
        self.att_pool = AttentionPooling(fusion_dim)
        pool_dim = fusion_dim * 3

        # Embedding projection head
        self.projection_head = nn.Sequential(
            nn.Linear(pool_dim, projection_hidden),
            nn.LayerNorm(projection_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(projection_hidden, embedding_dim),
        )

        # ArcFace components set up during training
        self._num_classes = None
        self.arcface = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_num_classes(self, num_classes: int) -> None:
        self._num_classes = num_classes
        from src.models.arcface import ArcFaceLoss
        self.arcface = ArcFaceLoss(
            in_features=self.embedding_dim,
            out_features=num_classes,
            margin=0.5,
            scale=64.0,
        )

    def forward(self, batch) -> dict[str, Tensor]:
        x, pos, edge_index = batch.x, batch.pos, batch.edge_index
        batch_assign = batch.batch
        num_graphs = int(batch_assign.max().item()) + 1

        # Project features
        h = self.input_proj(x)
        h = self.input_norm(h, batch_assign)

        history = [h]

        for i in range(self.num_layers):
            h_next = self.conv_layers[i](h, pos, edge_index)
            h_next = self.norms[i](h_next, batch_assign)
            h_next = F.gelu(h_next)
            h = self.dropouts[i](h_next)
            history.append(h)

        # Concatenate multi-scale representations
        h_cat = torch.cat(history, dim=-1)
        h_fused = self.fusion(h_cat)

        # Pooling
        mean_p = global_mean_pool(h_fused, batch_assign, size=num_graphs)
        max_p = global_max_pool(h_fused, batch_assign, size=num_graphs)
        att_p = self.att_pool(h_fused, batch_assign, size=num_graphs)
        pooled = torch.cat([mean_p, max_p, att_p], dim=-1)

        # Final projection to L2 embedding
        embedding = self.projection_head(pooled)
        embedding = F.normalize(embedding, p=2, dim=-1)

        out = {'embedding': embedding}
        if self.training and self.arcface is not None:
            loss, logits = self.arcface(embedding, batch.y)
            out['loss'] = loss
            out['logits'] = logits

        return out
