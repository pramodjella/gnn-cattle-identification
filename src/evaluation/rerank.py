"""
k-reciprocal re-ranking (Zhong et al., CVPR 2017) on a similarity matrix.
Unsupervised, label-free, test-time — the topological counterpart to S-norm's
distributional calibration. Used to test whether structure + calibration are
complementary (the premise behind a learned graph calibrator).

Operates on a probe x gallery cosine-similarity matrix; returns a refined
similarity (higher = more similar), so it drops into the same EER pipeline.
"""
from __future__ import annotations
import numpy as np


def _to_dist(sim):
    return 1.0 - sim  # cosine sim -> distance


def k_reciprocal_rerank(sim_pg, sim_gg=None, k1=20, k2=6, lambda_value=0.3):
    """Re-rank a probe x gallery similarity matrix with k-reciprocal encoding.

    Args:
        sim_pg: (P, G) probe-gallery cosine similarities.
        sim_gg: (G, G) gallery-gallery similarities (computed from templates if None).
        k1, k2: neighborhood sizes; lambda_value: mix of Jaccard and original distance.

    Returns:
        (P, G) refined similarity (1 - final_distance).
    """
    P, G = sim_pg.shape
    if sim_gg is None:
        # approximate gallery-gallery from probe-gallery is not possible; caller
        # should pass templates' self-similarity. Fall back to identity-ish.
        sim_gg = np.eye(G)
    # Build a joint (P+G) x (P+G) original distance matrix.
    N = P + G
    D = np.zeros((N, N), dtype=np.float32)
    D[:P, P:] = _to_dist(sim_pg)
    D[P:, :P] = _to_dist(sim_pg).T
    D[P:, P:] = _to_dist(sim_gg)
    np.fill_diagonal(D, 0.0)
    # probe-probe unknown -> large
    D[:P, :P] = 1.0
    np.fill_diagonal(D, 0.0)

    original_dist = D / (np.max(D, axis=0) + 1e-12)
    V = np.zeros((N, N), dtype=np.float32)
    initial_rank = np.argsort(original_dist, axis=1)

    for i in range(N):
        fwd = initial_rank[i, :k1 + 1]
        bwd = initial_rank[fwd, :k1 + 1]
        recip = fwd[np.any(bwd == i, axis=1)]
        recip_exp = recip
        for cand in recip:
            cfwd = initial_rank[cand, :int(np.around(k1 / 2)) + 1]
            cbwd = initial_rank[cfwd, :int(np.around(k1 / 2)) + 1]
            crecip = cfwd[np.any(cbwd == cand, axis=1)]
            if len(np.intersect1d(crecip, recip)) > 2.0 / 3 * len(crecip):
                recip_exp = np.append(recip_exp, crecip)
        recip_exp = np.unique(recip_exp)
        w = np.exp(-original_dist[i, recip_exp])
        V[i, recip_exp] = w / (np.sum(w) + 1e-12)

    if k2 > 1:
        Vq = np.zeros_like(V)
        for i in range(N):
            Vq[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = Vq

    # Jaccard distance
    jaccard = np.zeros((P, N), dtype=np.float32)
    for i in range(P):
        mn = np.minimum(V[i], V).sum(axis=1)
        mx = np.maximum(V[i], V).sum(axis=1)
        jaccard[i] = 1.0 - mn / (mx + 1e-12)

    final = jaccard[:, P:] * (1 - lambda_value) + original_dist[:P, P:] * lambda_value
    return 1.0 - final  # back to similarity
