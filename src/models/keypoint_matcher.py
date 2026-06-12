"""
KeypointMatcherGNN: Differentiable Keypoint Matcher for Cattle Identification
=============================================================================
Implements a GNN node encoder followed by a log-space Sinkhorn module to solve
a differentiable optimal transport matching problem between graph keypoints.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from src.models.gnn_v3 import GraphNorm, EdgeEncoder

def log_sinkhorn(scores: Tensor, bin_score: Tensor, iters: int = 20) -> Tensor:
    """
    Log-space Sinkhorn algorithm with a dustbin row and column.
    
    Args:
        scores: [M, N] similarity matrix between two sets of keypoints
        bin_score: [1] learnable score for unmatched keypoints (dustbin value)
        iters: Number of Sinkhorn normalization iterations
    Returns:
        [M+1, N+1] joint log-probability assignment matrix
    """
    M, N = scores.shape
    device = scores.device
    
    # 1. Construct the score matrix with dustbins: shape [M+1, N+1]
    # Rows correspond to Query nodes, Columns to Gallery nodes.
    # The last row and column are the dustbins.
    Z = torch.empty(M + 1, N + 1, device=device, dtype=scores.dtype)
    Z[:M, :N] = scores
    Z[:M, N] = bin_score
    Z[M, :N] = bin_score
    Z[M, N] = bin_score + bin_score.exp() # dustbin-to-dustbin entry

    # 2. Define marginal distributions (mu: row sum targets, nu: col sum targets)
    # Each keypoint wants to match to 1 keypoint, dustbins have capacity to match everything
    log_mu = torch.zeros(M + 1, device=device, dtype=scores.dtype)
    log_nu = torch.zeros(N + 1, device=device, dtype=scores.dtype)
    log_mu[M] = math.log(N)
    log_nu[N] = math.log(M)
    
    # 3. Sinkhorn scaling iterations
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    
    for _ in range(iters):
        # Update u (rows)
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
        # Update v (cols)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)
        
    return Z + u.unsqueeze(1) + v.unsqueeze(0)


class KeypointMatcherGNN(nn.Module):
    """
    GNN Node Encoder + Sinkhorn optimal transport keypoint matcher.
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        edge_attr_dim: int = 5,
        edge_enc_dim: int = 64,
        matching_dim: int = 128,
        sinkhorn_iterations: int = 20,
    ) -> None:
        super().__init__()
        self.sinkhorn_iterations = sinkhorn_iterations
        head_out = hidden_dim * num_heads

        # Node feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, head_out),
            nn.GELU(),
        )
        self.input_norm = GraphNorm(head_out)

        # Edge Encoder
        self.edge_encoder = EdgeEncoder(
            raw_edge_dim=edge_attr_dim,
            hidden_dim=edge_enc_dim,
            out_dim=edge_enc_dim,
        )

        # GATv2 layers to gather context
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(
                GATv2Conv(
                    in_channels=head_out,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_enc_dim,
                    dropout=0.15,
                    add_self_loops=True,
                )
            )
            self.norms.append(GraphNorm(head_out))
            self.dropouts.append(nn.Dropout(0.15))

        # Node matching projection head (maps context to discriminative matching space)
        self.match_proj = nn.Sequential(
            nn.Linear(head_out, matching_dim),
            nn.LayerNorm(matching_dim),
        )

        # Learnable dustbin parameter (threshold score for a match to be rejected)
        self.bin_score = nn.Parameter(torch.tensor([1.0]))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def encode_nodes(self, batch) -> Tensor:
        """
        Encode graph keypoints into descriptor space using GNN.
        Returns:
            [N, matching_dim] node descriptors
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_assign = batch.batch

        edge_emb = self.edge_encoder(edge_attr) if edge_attr is not None else None
        h = self.input_proj(x)
        h = self.input_norm(h, batch_assign)

        for i in range(len(self.conv_layers)):
            h_next = self.conv_layers[i](h, edge_index, edge_emb)
            h_next = self.norms[i](h_next, batch_assign)
            h_next = F.gelu(h_next)
            h = self.dropouts[i](h_next)

        return self.match_proj(h)

    def match_graphs(self, desc_A: Tensor, desc_B: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute optimal transport match matrix between two sets of node descriptors.
        
        Args:
            desc_A: [M, D] descriptors for Graph A
            desc_B: [N, D] descriptors for Graph B
        Returns:
            log_prob_assignment: [M+1, N+1] log probabilities
            match_score: [1] aggregate matching score (sum of matched probabilities)
        """
        # Pairwise similarity: [M, N]
        sim = torch.mm(desc_A, desc_B.t()) / math.sqrt(desc_A.size(-1))
        
        # Run Sinkhorn
        log_prob = log_sinkhorn(sim, self.bin_score, iters=self.sinkhorn_iterations)
        
        # Exponential of real keypoint block yields probabilities: [M, N]
        prob = log_prob[:desc_A.size(0), :desc_B.size(0)].exp()
        
        # Aggregate match score
        match_score = prob.sum()
        
        return log_prob, match_score
