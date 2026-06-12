"""
Topological Relation Module (TRM)
==================================
Applies graph attention convolutions to learn topological invariants
from the muzzle pattern's graph structure.

Uses multi-head Graph Attention Networks (GAT) to learn which 
spatial relationships between keypoints are most discriminative
for individual identification.

Key insight: The TRM captures implicit topological rules like
"bead A is always between beads B and C" that are unique per animal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TopologicalRelationModule(nn.Module):
    """
    Topological Relation Module using multi-head Graph Attention.
    
    Architecture:
        Input -> [GATConv -> GraphNorm -> ReLU -> Dropout] × L -> Output
    
    The attention mechanism learns edge importance weights, effectively
    discovering which topological relationships are most informative.
    """
    
    def __init__(self, in_dim, hidden_dim=256, num_heads=4, num_layers=2, 
                 dropout=0.2, concat_heads=True):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension per attention head
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout rate
            concat_heads: Whether to concatenate or average heads
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                layer_in = in_dim
            else:
                if concat_heads:
                    layer_in = hidden_dim * num_heads
                else:
                    layer_in = hidden_dim
            
            # Last layer: average heads instead of concatenating
            is_last = (i == num_layers - 1)
            heads = num_heads if not is_last else num_heads
            concat = concat_heads if not is_last else False
            
            self.layers.append(
                GATConv(
                    in_channels=layer_in,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=True,
                    bias=True,
                )
            )
            
            out_channels = hidden_dim * heads if concat else hidden_dim
            self.norms.append(nn.LayerNorm(out_channels))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output dimension
        self.output_dim = hidden_dim
        
        # Residual projection
        self.residual_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through TRM.
        
        Args:
            x: Node features (N, in_dim)
            edge_index: Edge connections (2, E)
            batch: Batch vector (N,) for batched graphs
            
        Returns:
            out: Refined node features (N, hidden_dim)
            attention_weights: Attention weights from the last layer
        """
        residual = self.residual_proj(x)
        attention_weights = None
        
        for i in range(self.num_layers):
            # GAT convolution
            if i == self.num_layers - 1:
                # Last layer: return attention weights
                x, (edge_idx, alpha) = self.layers[i](
                    x, edge_index, return_attention_weights=True
                )
                attention_weights = alpha
            else:
                x = self.layers[i](x, edge_index)
            
            # Normalization
            x = self.norms[i](x)
            
            x = F.relu(x)
            x = self.dropouts[i](x)
        
        # Residual connection
        out = x + residual
        
        return out, attention_weights
