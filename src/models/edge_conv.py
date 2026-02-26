"""
Dynamic EdgeConv Module
========================
Implements Dynamic Graph Convolution (EdgeConv) as described in
"Dynamic Graph CNN for Learning on Point Clouds" (Wang et al., 2019).

EdgeConv dynamically recomputes K-nearest neighbor graphs in the 
feature space at each layer, enabling the network to capture 
evolving local structures.

Formula: h'_i = max(MLP([h_i || h_j - h_i])) for j in neighbors(i)

This implementation uses pure PyTorch (no torch-cluster dependency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_graph(x, k, batch=None):
    """
    Compute k-nearest neighbor graph in feature space using pure PyTorch.
    
    Args:
        x: Node features (N, D)
        k: Number of nearest neighbors
        batch: Batch vector (N,) assigning each node to a graph
        
    Returns:
        edge_index: (2, N*k) tensor of edge indices
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    
    # Process each graph in the batch separately
    unique_batches = torch.unique(batch)
    edge_indices = []
    
    for b in unique_batches:
        mask = (batch == b)
        indices = torch.where(mask)[0]
        x_b = x[mask]
        n = x_b.size(0)
        
        if n <= 1:
            continue
        
        # Clamp k to n-1
        k_actual = min(k, n - 1)
        
        # Compute pairwise distances
        dist = torch.cdist(x_b, x_b)  # (n, n)
        
        # Set self-distance to infinity to exclude self-loops
        dist.fill_diagonal_(float('inf'))
        
        # Get k nearest neighbors
        _, knn_idx = dist.topk(k_actual, dim=1, largest=False)  # (n, k_actual)
        
        # Build edge index (source -> target)
        src = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, k_actual).reshape(-1)
        tgt = knn_idx.reshape(-1)
        
        # Map back to global indices
        src_global = indices[src]
        tgt_global = indices[tgt]
        
        edge_indices.append(torch.stack([src_global, tgt_global], dim=0))
    
    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    else:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)


class EdgeConvBlock(nn.Module):
    """
    Single EdgeConv block with batch normalization and residual connection.
    
    Architecture:
        Input -> DynamicEdgeConv(MLP) -> BatchNorm -> ReLU -> Dropout -> Output
        With residual connection if dimensions match.
    """
    
    def __init__(self, in_dim, out_dim, k=12, aggr='max', dropout=0.3):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            k: Number of nearest neighbors for dynamic graph
            aggr: Aggregation method ('max', 'mean', 'add')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.k = k
        self.aggr = aggr
        
        # MLP that processes [h_i || h_j - h_i] -> out_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (with projection if dimensions differ)
        self.residual = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim 
            else nn.Identity()
        )
        
    def forward(self, x, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features (N, in_dim)
            batch: Batch vector (N,) for batched graphs
            
        Returns:
            out: Updated node features (N, out_dim)
        """
        # Compute dynamic KNN graph in feature space
        edge_index = knn_graph(x, self.k, batch=batch)
        
        # Get source and target node features
        src, tgt = edge_index[0], edge_index[1]
        
        # Construct edge features: [h_i || h_j - h_i]
        x_src = x[src]  # Features of source nodes
        x_tgt = x[tgt]  # Features of target (neighbor) nodes
        edge_features = torch.cat([x_src, x_tgt - x_src], dim=1)
        
        # Apply MLP to edge features
        edge_out = self.mlp(edge_features)
        
        # Aggregate neighbor messages for each node
        N = x.size(0)
        out_dim = edge_out.size(1)
        
        if self.aggr == 'max':
            h = torch.full((N, out_dim), float('-inf'), device=x.device)
            h.scatter_reduce_(0, src.unsqueeze(1).expand(-1, out_dim), edge_out, reduce='amax')
            # Replace -inf with 0 for nodes with no neighbors
            h = torch.where(h == float('-inf'), torch.zeros_like(h), h)
        elif self.aggr == 'mean':
            h = torch.zeros(N, out_dim, device=x.device)
            count = torch.zeros(N, 1, device=x.device)
            h.scatter_add_(0, src.unsqueeze(1).expand(-1, out_dim), edge_out)
            count.scatter_add_(0, src.unsqueeze(1), torch.ones_like(src, dtype=torch.float).unsqueeze(1))
            h = h / count.clamp(min=1)
        else:  # 'add'
            h = torch.zeros(N, out_dim, device=x.device)
            h.scatter_add_(0, src.unsqueeze(1).expand(-1, out_dim), edge_out)
        
        h = self.bn(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Residual connection
        res = self.residual(x)
        out = h + res
        
        return out


class DynamicEdgeConvBlock(nn.Module):
    """
    Multi-layer Dynamic EdgeConv block stack.
    
    Progressively transforms node features through multiple EdgeConv layers,
    dynamically recomputing the graph at each layer in feature space.
    """
    
    def __init__(self, in_dim, hidden_dims, k=12, aggr='max', dropout=0.3):
        """
        Args:
            in_dim: Input feature dimension (256 for SuperPoint descriptors)
            hidden_dims: List of hidden dimensions for each layer
            k: KNN parameter for dynamic graph construction
            aggr: Aggregation method
            dropout: Dropout rate
        """
        super().__init__()
        
        dims = [in_dim] + list(hidden_dims)
        
        self.layers = nn.ModuleList([
            EdgeConvBlock(dims[i], dims[i + 1], k=k, aggr=aggr, dropout=dropout)
            for i in range(len(dims) - 1)
        ])
        
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x, batch=None):
        """
        Forward pass through all EdgeConv layers.
        
        Args:
            x: Node features (N, in_dim)
            batch: Batch vector
            
        Returns:
            x: Transformed node features (N, output_dim)
            intermediates: List of intermediate representations
        """
        intermediates = [x]
        
        for layer in self.layers:
            x = layer(x, batch=batch)
            intermediates.append(x)
        
        return x, intermediates
