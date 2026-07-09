"""
Quality Features for Quality-Conditioned Score Calibration (QuaCal)
==================================================================
Per-sample descriptors that (hypothetically) predict *which* branch — the CNN
(texture) or the GNN (topology) — is more reliable for a given input. These
drive the quality-conditioned fusion/calibration gate.

Three groups:
  * Image quality  — blur (Laplacian variance), brightness, contrast.
  * Graph quality  — node count, avg degree, #connected components,
                     edge-length variance.
  * Confidence     — per-branch top-1 margin and cross-branch disagreement
                     (filled in by the caller, which has the score matrices).

All features are cheap and label-free, so the gate is deployable at test time.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor


def image_quality(img: Tensor) -> Dict[str, float]:
    """Blur / brightness / contrast from a (3,H,W) or (H,W) image tensor.

    The tensor may be ImageNet-normalised; we operate on a grayscale proxy, so
    only relative scale matters for the downstream gate.
    """
    import cv2
    x = img.detach().float().cpu()
    if x.dim() == 3:
        x = x.mean(0)                       # to grayscale
    a = x.numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    a8 = (a * 255).astype(np.uint8)
    lap = cv2.Laplacian(a8, cv2.CV_64F)
    return {
        'blur': float(lap.var()),           # low = blurry
        'brightness': float(a.mean()),
        'contrast': float(a.std()),
    }


def graph_quality(data) -> Dict[str, float]:
    """Topological quality of a single graph."""
    n = int(data.x.size(0))
    e = int(data.edge_index.size(1))
    avg_deg = e / max(n, 1)

    # connected components (undirected) via union-find, cheap and dependency-free
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    ei = data.edge_index.cpu().numpy()
    for k in range(e):
        a, b = int(ei[0, k]), int(ei[1, k])
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    n_components = len({find(i) for i in range(n)}) if n else 0

    # edge-length variance from node positions
    elen_var = 0.0
    pos = getattr(data, 'pos', None)
    if pos is not None and e > 0:
        p = pos[:, :2].detach().cpu().numpy()
        d = np.linalg.norm(p[ei[0]] - p[ei[1]], axis=1)
        elen_var = float(d.var())

    return {
        'node_count': float(n),
        'avg_degree': float(avg_deg),
        'n_components': float(n_components),
        'edge_len_var': elen_var,
    }


def branch_confidence(sim_row_cnn: np.ndarray, sim_row_gnn: np.ndarray) -> Dict[str, float]:
    """Top-1 margins per branch + cross-branch disagreement.

    Args:
        sim_row_cnn/gnn: similarity of one probe to the gallery prototypes.
    """
    def margin(s):
        srt = np.sort(s)[::-1]
        return float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])
    pred_cnn = int(np.argmax(sim_row_cnn))
    pred_gnn = int(np.argmax(sim_row_gnn))
    return {
        'cnn_margin': margin(sim_row_cnn),
        'gnn_margin': margin(sim_row_gnn),
        'cnn_top1': float(np.max(sim_row_cnn)),
        'gnn_top1': float(np.max(sim_row_gnn)),
        'disagreement': float(pred_cnn != pred_gnn),
    }


FEATURE_ORDER = [
    'blur', 'brightness', 'contrast',
    'node_count', 'avg_degree', 'n_components', 'edge_len_var',
    'cnn_margin', 'gnn_margin', 'cnn_top1', 'gnn_top1', 'disagreement',
]


def to_vector(feats: Dict[str, float]) -> np.ndarray:
    return np.array([feats.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float64)
