"""
ProtoN: Prototype Node Graph Neural Network for Cattle Muzzle Identification
=============================================================================
Implements a dual-path message-passing mechanism between muzzle graph nodes
and an identity-level prototype node, following the ProtoN framework.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from src.models.gnn_v3 import GraphNorm, EdgeEncoder, AttentionPooling

class PrototypeNodeUpdate(nn.Module):
    """
    Dual-path update for the identity-level prototype node.
    Refines both the individual keypoint representations and the prototype node.
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
        pnode: Tensor,
        batch: Tensor,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor]:
        # 1. Real nodes send message to prototype node (aggregation)
        agg = global_mean_pool(x, batch, size=num_graphs)
        
        # 2. Update prototype node representation
        pnode = self.norm(self.mlp(pnode + agg) + pnode)
        
        # 3. Prototype node broadcasts message back to real nodes
        x_out = x + pnode[batch]
        return x_out, pnode

class CattleProtoN(nn.Module):
    """
    Prototype Node GNN (ProtoN) architecture for Cattle Muzzle Identification.
    """
    def __init__(
        self,
        num_classes: int,
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
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        head_out = hidden_dim * num_heads

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, head_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_norm = GraphNorm(head_out)

        # Edge Encoder
        self.edge_encoder = EdgeEncoder(
            raw_edge_dim=edge_attr_dim,
            hidden_dim=edge_enc_dim,
            out_dim=edge_enc_dim,
        )

        # GATv2 Layers + Prototype Node Updates
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self.gat_dropouts = nn.ModuleList()
        self.pnode_updaters = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=head_out,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_enc_dim,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.gat_norms.append(GraphNorm(head_out))
            self.gat_dropouts.append(nn.Dropout(dropout))
            self.pnode_updaters.append(PrototypeNodeUpdate(head_out, dropout=dropout))

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

        # Learnable Class Prototypes for global classification & alignment
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.class_prototypes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch) -> dict[str, Tensor]:
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_assign = batch.batch
        num_graphs = int(batch_assign.max().item()) + 1

        # Encode edge attributes
        edge_emb = self.edge_encoder(edge_attr) if edge_attr is not None else None

        # Input projection
        x_proj = self.input_proj(x)
        h = self.input_norm(x_proj, batch_assign)

        # Initialize prototype nodes: use graph mean representation as starting point
        pnode = global_mean_pool(h, batch_assign, size=num_graphs)

        # Keep history for skip connections
        history = [h]

        for i in range(len(self.gat_layers)):
            # 1. Update nodes with prototype node information (dual-path step 1)
            h, pnode = self.pnode_updaters[i](h, pnode, batch_assign, num_graphs)
            
            # 2. Message passing among nodes
            h_next = self.gat_layers[i](h, edge_index, edge_emb)
            h_next = self.gat_norms[i](h_next, batch_assign)
            h_next = F.gelu(h_next)
            h = self.gat_dropouts[i](h_next)
            
            history.append(h)

        # Concatenate multi-scale representations
        h_cat = torch.cat(history, dim=-1)
        h_fused = self.fusion(h_cat)

        # Pooling
        mean_p = global_mean_pool(h_fused, batch_assign, size=num_graphs)
        max_p = global_max_pool(h_fused, batch_assign, size=num_graphs)
        att_p = self.att_pool(h_fused, batch_assign, size=num_graphs)
        pooled = torch.cat([mean_p, max_p, att_p], dim=-1)

        # Project to embedding space
        embedding = self.projection_head(pooled)
        embedding = F.normalize(embedding, p=2, dim=-1)

        return {
            'embedding': embedding,
            'pnode': pnode,
        }

    def compute_loss(self, embedding: Tensor, labels: Tensor, temperature: float = 0.07, align_weight: float = 0.5) -> Tensor:
        """
        Compute ProtoN Hybrid Loss:
          1. Global Classification Loss (Cosine similarity with learnable class prototypes)
          2. Cross-Graph Prototype Alignment Loss (Episodic contrastive learning)
        """
        device = embedding.device
        
        # --- 1. Global Classification Loss (Cosine classification) ---
        # Normalize class prototypes
        norm_prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)
        # Cosine similarity matrix: (Batch, NumClasses)
        logits = torch.mm(embedding, norm_prototypes.t()) / temperature
        cls_loss = F.cross_entropy(logits, labels)

        # --- 2. Cross-Graph Prototype Alignment Loss ---
        # For each class in the batch, pull embeddings of same class together
        # and push different classes apart.
        # Since we use PK Sampler, we have multiple graphs for the same label.
        batch_size = embedding.size(0)
        sim_matrix = torch.mm(embedding, embedding.t()) / temperature
        
        # Mask for positive pairs (same label, but not self)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = labels_eq & ~self_mask
        
        # If no positive pairs in the batch, skip alignment loss
        if not pos_mask.any():
            return cls_loss

        # Standard InfoNCE / SupCon loss
        # Subtract max logit for numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits_stable = sim_matrix - logits_max.detach()
        exp_logits = torch.exp(logits_stable)
        
        # Sum exponentials over all pairs (excluding self)
        sum_exp = exp_logits.sum(dim=1, keepdim=True) - exp_logits * self_mask
        
        # Compute log probability
        log_prob = logits_stable - torch.log(sum_exp + 1e-8)
        
        # Average log prob over positive pairs
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        align_loss = -mean_log_prob_pos.mean()

        return cls_loss + align_weight * align_loss
