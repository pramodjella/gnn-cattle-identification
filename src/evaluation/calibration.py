"""
Test-Time Score Calibration Methods for Cross-Domain Re-ID
=========================================================
Candidate mechanisms tested against plain symmetric S-norm. The main-track
question: does any NOVEL variant beat S-norm at recovering cross-domain
verification, label-free and retraining-free?

All operate on a probe x gallery score matrix S (cosine similarities) and
return a recalibrated matrix; identity labels are never used.
"""

from __future__ import annotations

import numpy as np


def snorm(S: np.ndarray) -> np.ndarray:
    """Adaptive symmetric S-norm (baseline to beat)."""
    S = S.astype(np.float64)
    mu_r = S.mean(1, keepdims=True); sd_r = S.std(1, keepdims=True) + 1e-8
    mu_c = S.mean(0, keepdims=True); sd_c = S.std(0, keepdims=True) + 1e-8
    return 0.5 * ((S - mu_r) / sd_r + (S - mu_c) / sd_c)


def asnorm(S: np.ndarray, k: int = 30) -> np.ndarray:
    """Adaptive S-norm: normalise using only the top-k most-similar cohort
    entries (the discriminative region), per row and column."""
    S = S.astype(np.float64)
    k_r = min(k, S.shape[1]); k_c = min(k, S.shape[0])
    top_r = np.sort(S, axis=1)[:, -k_r:]
    mu_r = top_r.mean(1, keepdims=True); sd_r = top_r.std(1, keepdims=True) + 1e-8
    top_c = np.sort(S, axis=0)[-k_c:, :]
    mu_c = top_c.mean(0, keepdims=True); sd_c = top_c.std(0, keepdims=True) + 1e-8
    return 0.5 * ((S - mu_r) / sd_r + (S - mu_c) / sd_c)


def quantile_norm(S: np.ndarray) -> np.ndarray:
    """Distribution-free rank/quantile normalisation (robust to non-Gaussian
    shift): map each score to its rank within the row and column cohort."""
    from scipy.stats import rankdata
    S = S.astype(np.float64)
    R = np.vstack([rankdata(r) / len(r) for r in S])
    C = np.column_stack([rankdata(S[:, j]) / S.shape[0] for j in range(S.shape[1])])
    return 0.5 * (R + C)


def quality_snorm(S: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Quality-conditioned S-norm: scale calibration strength per probe by an
    unsupervised shift indicator (lower peak similarity => more shifted =>
    stronger normalisation). Interpolates raw<->S-norm per row."""
    S = S.astype(np.float64)
    sn = snorm(S)
    peak = S.max(1, keepdims=True)
    q = 1.0 - (peak - peak.min()) / (peak.max() - peak.min() + 1e-8)  # 0..1
    q = q ** gamma
    raw_z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-8)
    return q * sn + (1 - q) * raw_z


def align(S: np.ndarray, ref_mu: float, ref_sd: float) -> np.ndarray:
    """Source-distribution alignment (per-probe Z-norm rescaled to a clean
    source reference). Requires source impostor stats."""
    S = S.astype(np.float64)
    mu_r = S.mean(1, keepdims=True); sd_r = S.std(1, keepdims=True) + 1e-8
    return (S - mu_r) / sd_r * ref_sd + ref_mu


METHODS = {
    'baseline': lambda S, **kw: S,
    's-norm': lambda S, **kw: snorm(S),
    'as-norm(k=20)': lambda S, **kw: asnorm(S, k=20),
    'as-norm(k=40)': lambda S, **kw: asnorm(S, k=40),
    'quantile': lambda S, **kw: quantile_norm(S),
    'quality-snorm': lambda S, **kw: quality_snorm(S),
}
