"""
Image Corruptions for Robustness Evaluation
===========================================
Deployment-realistic corruptions at severities {1,3,5}, applied to muzzle
images to probe branch robustness and drive quality-conditioned fusion.

Corruptions (ImageNet-C style, adapted to muzzle capture failure modes):
  * blur        — Gaussian blur (out-of-focus / motion)
  * brightness  — over/under-exposure + haze (fog)
  * spatter     — occlusion / dirt / mud specks on the muzzle

Operate on a (3,H,W) float tensor in [0,1] (pre-normalisation) and return the
same. The caller applies ImageNet normalisation afterwards.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# per-severity parameters (index by severity 1..5)
_BLUR_SIGMA = {1: 0.6, 2: 1.0, 3: 1.6, 4: 2.4, 5: 3.2}
_BRIGHT_DELTA = {1: 0.15, 2: 0.28, 3: 0.42, 4: 0.55, 5: 0.7}   # additive haze
_SPATTER_FRAC = {1: 0.02, 2: 0.05, 3: 0.10, 4: 0.18, 5: 0.28}  # fraction occluded


def _to_np(img: Tensor):
    return img.detach().float().cpu().numpy().transpose(1, 2, 0)  # HWC


def _to_t(a: np.ndarray, ref: Tensor) -> Tensor:
    return torch.from_numpy(a.transpose(2, 0, 1)).to(ref.dtype)


def blur(img: Tensor, severity: int = 3) -> Tensor:
    import cv2
    a = _to_np(img)
    s = _BLUR_SIGMA[severity]
    k = int(2 * round(3 * s) + 1)
    out = cv2.GaussianBlur(a, (k, k), s)
    return _to_t(np.clip(out, 0, 1), img)


def brightness(img: Tensor, severity: int = 3) -> Tensor:
    """Over-exposure + low-contrast haze (fog-like)."""
    a = _to_np(img)
    d = _BRIGHT_DELTA[severity]
    out = a * (1 - 0.5 * d) + d          # lift blacks + compress range
    return _to_t(np.clip(out, 0, 1), img)


def spatter(img: Tensor, severity: int = 3, seed: int | None = None) -> Tensor:
    """Random dark specks occluding the muzzle (dirt/mud)."""
    a = _to_np(img).copy()
    H, W, _ = a.shape
    rng = np.random.RandomState(seed)
    n_specks = int(_SPATTER_FRAC[severity] * H * W / 25)   # ~5x5 specks
    for _ in range(n_specks):
        y, x = rng.randint(0, H), rng.randint(0, W)
        r = rng.randint(2, 5)
        a[max(0, y - r):y + r, max(0, x - r):x + r] = rng.uniform(0.0, 0.2)
    return _to_t(np.clip(a, 0, 1), img)


CORRUPTIONS = {'blur': blur, 'brightness': brightness, 'spatter': spatter}


def apply(img: Tensor, kind: str, severity: int, seed: int | None = None) -> Tensor:
    if kind == 'clean' or severity == 0:
        return img
    fn = CORRUPTIONS[kind]
    if kind == 'spatter':
        return fn(img, severity, seed=seed)
    return fn(img, severity)
