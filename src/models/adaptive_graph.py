"""
Adaptive Graph Construction (ADGC)
==================================
A fixed geometric k-NN graph bakes in exactly the deformation we want the model
to be invariant to: two images of the same muzzle taken at different angles
produce different k-NN neighbourhoods, and spurious edges cross unrelated ridge
regions. ADGC replaces the hard "keep the k nearest spatial neighbours" rule
with a *learned* edge relevance gate.

Given candidate edges (the geometric k-NN edges already built in the graph),
ADGC scores each edge from the endpoint node features and the geometric edge
attributes, producing a gate g_e in [0,1]. Gates are used to (i) reweight edge
attributes so downstream message passing emphasises reliable connections, and
(ii) optionally prune edges whose gate falls below a threshold, yielding a
sparser, per-image-adaptive topology.

This is a single, self-contained module so it can be cited as the paper's
learned graph-construction contribution and ablated against the static k-NN.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaptiveGraphConstruction(nn.Module):
    """Learned edge-relevance gating over a candidate (k-NN) edge set."""

    def __init__(self, node_dim: int, edge_dim: int = 5,
                 hidden: int = 64, prune_threshold: float = 0.1,
                 min_degree: int = 2):
        """
        Args:
            node_dim:        Node feature dimension.
            edge_dim:        Geometric edge-attribute dimension (dx,dy,d,θ,scale).
            hidden:          Hidden width of the edge scorer.
            prune_threshold: Edges with gate < threshold are dropped (train &
                             eval). Set 0 to disable pruning (pure reweighting).
            min_degree:      Never prune a node below this out-degree (keeps the
                             graph connected on small muzzle graphs).
        """
        super().__init__()
        self.prune_threshold = prune_threshold
        self.min_degree = min_degree

        self.scorer = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """Score, reweight, and optionally prune the candidate edges.

        Args:
            x:          (N, node_dim) node features.
            edge_index: (2, E) candidate edges.
            edge_attr:  (E, edge_dim) geometric edge attributes (optional).

        Returns:
            new_edge_index: (2, E') kept edges.
            new_edge_attr:  (E', edge_dim) gate-reweighted attributes.
            gate:           (E',) learned edge gates in [0, 1] (for viz/loss).
        """
        src, dst = edge_index[0], edge_index[1]
        e = edge_index.size(1)
        if e == 0:
            ea = edge_attr if edge_attr is not None else x.new_zeros((0, 1))
            return edge_index, ea, x.new_zeros((0,))

        if edge_attr is None:
            edge_attr = x.new_zeros((e, self.scorer[0].in_features - 2 * x.size(1)))

        feats = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        gate = torch.sigmoid(self.scorer(feats)).squeeze(-1)      # (E,)

        # Reweight edge attributes by relevance.
        weighted_attr = edge_attr * gate.unsqueeze(-1)

        if self.prune_threshold <= 0:
            return edge_index, weighted_attr, gate

        keep = gate >= self.prune_threshold
        keep = self._protect_min_degree(keep, src, gate, x.size(0))

        return edge_index[:, keep], weighted_attr[keep], gate[keep]

    def _protect_min_degree(self, keep: Tensor, src: Tensor,
                            gate: Tensor, num_nodes: int) -> Tensor:
        """Guarantee each source node keeps at least ``min_degree`` edges.

        Vectorised: force-keeps each source node's top-``min_degree`` edges by
        gate, with no Python loop over nodes (critical for training throughput).
        """
        if self.min_degree <= 0 or src.numel() == 0:
            return keep
        E = src.numel()
        device = src.device

        # Only nodes whose current kept-degree is below min_degree are fixed
        # (matches the reference loop: satisfied nodes are left untouched).
        kept_count = torch.zeros(num_nodes, device=device, dtype=torch.long)
        kept_count.scatter_add_(0, src, keep.long())
        deficient_edge = kept_count[src] < self.min_degree      # (E,) original order

        # Order edges by source, and by descending gate within each source.
        order_g = torch.argsort(gate, descending=True)
        order_s = torch.argsort(src[order_g], stable=True)
        final_order = order_g[order_s]                 # grouped by src, gate desc
        sorted_src = src[final_order]

        # Rank of each edge within its source group (0 == highest gate).
        _, counts = torch.unique_consecutive(sorted_src, return_counts=True)
        starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        group_start = torch.repeat_interleave(starts, counts)
        within_rank = torch.arange(E, device=device) - group_start

        force = (within_rank < self.min_degree) & deficient_edge[final_order]
        keep = keep.clone()
        keep[final_order[force]] = True
        return keep
