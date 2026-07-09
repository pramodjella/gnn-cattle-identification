"""
Quantitative Faithfulness Metrics for GNN Explanations
======================================================
Turns qualitative importance heatmaps into measurable evidence that the
explanations reflect what the model actually uses.

Metrics
-------
* **Fidelity+ (comprehensiveness)** — drop in the predicted-class probability
  when the *most important* nodes are removed. Higher is better: if removing
  the highlighted nodes hurts the prediction, they were genuinely used.
* **Fidelity- (sufficiency)**       — drop in the predicted-class probability
  when *only* the most important nodes are kept. Lower is better: if the
  highlighted nodes alone preserve the prediction, they are sufficient.
* **Sparsity**                      — fraction of nodes deemed unimportant.
  Faithful explanations should be sparse (few nodes carry the decision).
* **Explanation agreement**         — Spearman rank correlation between the
  node-importance vectors produced by two independent explainers on the same
  graph. High agreement across methods is strong evidence the model relies on
  real structure, not method-specific artifacts.

The model is assumed to follow the codebase convention ``model(data) -> dict``
with an ``'embedding'`` key. Class scores are derived from the ArcFace class
prototypes (cosine similarity -> softmax), so no external gallery is needed.

References
----------
Pope et al. (2019); Yuan et al. (2021) "On Explainability of GNNs via Subgraph
Explorations"; DeYoung et al. (2020) ERASER (comprehensiveness / sufficiency).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _find_arcface_prototypes(model: torch.nn.Module) -> Optional[Tensor]:
    """Locate the ArcFace class-weight matrix (num_classes, embedding_dim)."""
    for name, param in model.named_parameters():
        if 'arcface' in name.lower() and param.dim() == 2:
            return param.detach()
    # Common fallbacks
    for attr in ('arcface', 'arc_face', 'head'):
        mod = getattr(model, attr, None)
        if mod is not None:
            w = getattr(mod, 'weight', None)
            if isinstance(w, Tensor) and w.dim() == 2:
                return w.detach()
    return None


def _find_arcface_scale(model: torch.nn.Module, default: float = 30.0) -> float:
    """Locate the ArcFace logit scale (s), i.e. the inference temperature.

    The model scores classes as ``softmax(s * cosine)``; using the real scale
    makes fidelity reflect the model's actual (sharp) decision instead of a
    near-uniform softmax over raw cosines.
    """
    for module in model.modules():
        s = getattr(module, 'scale', None)
        if isinstance(s, (int, float)) and s > 1.0:
            return float(s)
    return default


def _subgraph(data, keep_idx: Tensor):
    """Return a copy of ``data`` restricted to ``keep_idx`` nodes.

    Edges are kept only if both endpoints survive, and are re-indexed.
    ``pos``, ``edge_attr`` and ``keypoint_scores`` are carried along if present.
    """
    keep_idx = keep_idx.to(data.x.device)
    n = data.x.size(0)

    remap = torch.full((n,), -1, dtype=torch.long, device=data.x.device)
    remap[keep_idx] = torch.arange(keep_idx.size(0), device=data.x.device)

    out = data.clone()
    out.y = None  # avoid triggering the model's training-time loss branch
    out.x = data.x[keep_idx]
    if getattr(data, 'pos', None) is not None:
        out.pos = data.pos[keep_idx]
    if getattr(data, 'keypoint_scores', None) is not None:
        out.keypoint_scores = data.keypoint_scores[keep_idx]

    ei = data.edge_index
    mask = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
    out.edge_index = torch.stack([remap[ei[0][mask]], remap[ei[1][mask]]])
    if getattr(data, 'edge_attr', None) is not None:
        out.edge_attr = data.edge_attr[mask]

    out.batch = torch.zeros(keep_idx.size(0), dtype=torch.long, device=data.x.device)
    return out


class GraphFaithfulness:
    """Compute Fidelity+/- and sparsity for GNN node explanations."""

    def __init__(self, model: torch.nn.Module,
                 device: Optional[torch.device] = None,
                 prototypes: Optional[Tensor] = None,
                 scale: Optional[float] = None):
        """
        Args:
            model:       Trained GNN (``model(data) -> {'embedding': ...}``).
            device:      Torch device; inferred from model if omitted.
            prototypes:  (C, D) class prototypes. Auto-detected from the
                         ArcFace head when omitted.
            scale:       ArcFace logit scale s (softmax(s * cosine)). Auto-
                         detected from the model when omitted.
        """
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.scale = scale if scale is not None else _find_arcface_scale(model)
        self.prototypes = prototypes if prototypes is not None else _find_arcface_prototypes(model)
        if self.prototypes is not None:
            self.prototypes = F.normalize(self.prototypes.to(self.device), p=2, dim=-1)

    @torch.no_grad()
    def _class_prob(self, data) -> Tensor:
        """Return the softmax class-probability vector for a single graph."""
        data = data.to(self.device)
        if getattr(data, 'y', None) is not None:
            data.y = None  # avoid the model's training-time loss branch
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        emb = self.model(data)['embedding']            # (1, D)
        emb = F.normalize(emb, p=2, dim=-1)
        if self.prototypes is None:
            # Degenerate fallback: use squashed embedding norm as a scalar score.
            return torch.sigmoid(emb.norm(dim=-1))
        logits = (emb @ self.prototypes.t()) * self.scale
        return F.softmax(logits, dim=-1)[0]            # (C,)

    @torch.no_grad()
    def fidelity(self, data, node_importance: Tensor,
                 fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
                 min_keep: int = 3) -> Dict[str, object]:
        """Fidelity+/- curves for a single graph.

        Args:
            data:            PyG graph.
            node_importance: (N,) importance scores.
            fractions:       Fractions of nodes treated as "important".
            min_keep:        Never shrink a graph below this many nodes
                             (avoids degenerate empty-graph forwards).

        Returns:
            dict with per-fraction fidelity_plus / fidelity_minus and their
            averages over fractions (curve summaries).
        """
        data = data.to(self.device)
        imp = node_importance.detach().to(self.device).flatten()
        n = data.x.size(0)

        base_prob = self._class_prob(data)
        target = int(base_prob.argmax().item())
        p_full = float(base_prob[target].item())

        order = torch.argsort(imp, descending=True)   # important first

        fids_plus, fids_minus = [], []
        per_fraction = []
        for frac in fractions:
            k = max(min_keep, int(round(frac * n)))
            k = min(k, n)
            top = order[:k]                            # important nodes
            rest = order[k:]                           # unimportant nodes

            # Fidelity+ : remove important nodes -> keep the rest.
            if rest.numel() >= 1:
                p_remove = float(self._class_prob(_subgraph(data, rest))[target].item())
            else:
                p_remove = 0.0
            fid_plus = p_full - p_remove

            # Fidelity- : keep only important nodes.
            p_keep = float(self._class_prob(_subgraph(data, top))[target].item())
            fid_minus = p_full - p_keep

            fids_plus.append(fid_plus)
            fids_minus.append(fid_minus)
            per_fraction.append({
                'fraction': frac, 'k': int(k),
                'fidelity_plus': fid_plus, 'fidelity_minus': fid_minus,
            })

        return {
            'target_class': target,
            'p_full': p_full,
            'per_fraction': per_fraction,
            'fidelity_plus': float(np.mean(fids_plus)),   # comprehensiveness
            'fidelity_minus': float(np.mean(fids_minus)),  # sufficiency
        }

    @staticmethod
    def sparsity(node_importance: Tensor, threshold: float = 0.5) -> float:
        """Fraction of nodes below ``threshold`` (i.e. deemed unimportant)."""
        imp = node_importance.detach().flatten()
        return float((imp < threshold).float().mean().item())

    def evaluate_dataset(self, graphs: Sequence, explain_fn: Callable[[object], Tensor],
                         fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.5),
                         max_graphs: Optional[int] = None) -> Dict[str, object]:
        """Aggregate faithfulness over many graphs.

        Args:
            graphs:     Iterable of PyG graphs.
            explain_fn: Maps a graph -> (N,) node importance tensor.
            fractions:  Fractions passed to :meth:`fidelity`.
            max_graphs: Cap for speed.

        Returns:
            dict of mean/std Fidelity+/-, mean sparsity, and n.
        """
        fplus, fminus, spars = [], [], []
        for i, g in enumerate(graphs):
            if max_graphs is not None and i >= max_graphs:
                break
            imp = explain_fn(g)
            res = self.fidelity(g, imp, fractions=fractions)
            fplus.append(res['fidelity_plus'])
            fminus.append(res['fidelity_minus'])
            spars.append(self.sparsity(imp))
        return {
            'num_graphs': len(fplus),
            'fidelity_plus_mean': float(np.mean(fplus)) if fplus else 0.0,
            'fidelity_plus_std': float(np.std(fplus)) if fplus else 0.0,
            'fidelity_minus_mean': float(np.mean(fminus)) if fminus else 0.0,
            'fidelity_minus_std': float(np.std(fminus)) if fminus else 0.0,
            'sparsity_mean': float(np.mean(spars)) if spars else 0.0,
        }


def explanation_agreement(importances: Dict[str, Tensor]) -> Dict[str, float]:
    """Spearman rank correlation between node-importance vectors.

    Args:
        importances: mapping ``{method_name: (N,) importance}`` for the SAME
                     graph (identical node ordering).

    Returns:
        dict of pairwise correlations plus the mean across all pairs.
    """
    from itertools import combinations
    try:
        from scipy.stats import spearmanr
    except Exception:  # pragma: no cover - scipy is a hard dep of sklearn anyway
        spearmanr = None

    names = list(importances.keys())
    vecs = {k: v.detach().cpu().numpy().flatten() for k, v in importances.items()}

    def _spearman(a, b):
        if spearmanr is not None:
            r = spearmanr(a, b).correlation
            return float(r) if r == r else 0.0  # guard NaN
        # Fallback: Pearson on ranks
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        if ra.std() < 1e-8 or rb.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    pairwise = {}
    for a, b in combinations(names, 2):
        pairwise[f'{a}__vs__{b}'] = _spearman(vecs[a], vecs[b])

    mean = float(np.mean(list(pairwise.values()))) if pairwise else 0.0
    return {'pairwise': pairwise, 'mean_agreement': mean}


def importance_correlation(node_importance: Tensor, node_property: Tensor) -> float:
    """Spearman correlation of importance with a biological node property.

    Use this to test whether attention aligns with, e.g., local keypoint
    density or detector confidence (the muzzle's dermatoglyphic structure).

    Args:
        node_importance: (N,) explanation importance.
        node_property:   (N,) per-node biological quantity.

    Returns:
        Spearman rho in [-1, 1].
    """
    a = node_importance.detach().cpu().numpy().flatten()
    b = node_property.detach().cpu().numpy().flatten()
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b).correlation
        return float(r) if r == r else 0.0
    except Exception:
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        if ra.std() < 1e-8 or rb.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])
