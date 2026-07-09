"""
Open-Set Biometric Evaluation
=============================
Closed-set Rank-1 answers "given that this animal is enrolled, do we rank it
first?". Real deployments must also answer "is this animal enrolled at all?".
Open-set evaluation measures both: correct identification of *known* probes
while *rejecting* probes from identities that are not in the gallery.

Protocol
--------
* Gallery: a per-identity template (mean embedding) for the KNOWN identities
  only (built from enrolment images, e.g. the train split).
* Probes: test images. A probe is "known" if its identity is enrolled, and
  "unknown" otherwise. Unknown probes must be rejected.
* Score of a probe = max cosine similarity to any gallery template. A probe is
  accepted if that score exceeds an operating threshold t.

Reported metrics
----------------
* **DIR@rank1(FAR)** — Detection & Identification Rate: fraction of known
  probes that are both accepted (score > t) AND matched to the correct identity
  at rank 1, where t is set to achieve a target False Alarm Rate on the unknown
  probes (Phillips et al., 2011).
* **Open-set AUC** — ROC over the accept/reject decision (known vs unknown).

Note on training: an embedding model trained closed-set can still be evaluated
open-set — enrolment (not training) defines the known/unknown partition. For a
strict "generalisation to unseen identities" claim, retrain with the unknown
identities held out of training (see docs runbook).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def build_gallery(embeddings: Tensor, labels: Tensor,
                  known_ids: Sequence[int]) -> Dict[str, Tensor]:
    """Build L2-normalised per-identity mean templates for known identities.

    Args:
        embeddings: (N, D) enrolment embeddings.
        labels:     (N,) identity labels.
        known_ids:  identities to enrol.

    Returns:
        dict with 'templates' (K, D) and 'ids' (K,).
    """
    emb = F.normalize(embeddings, p=2, dim=-1)
    known = set(int(i) for i in known_ids)
    templates, ids = [], []
    for cid in sorted(known):
        mask = labels == cid
        if mask.sum() == 0:
            continue
        templates.append(F.normalize(emb[mask].mean(dim=0), p=2, dim=-1))
        ids.append(cid)
    return {'templates': torch.stack(templates), 'ids': torch.tensor(ids)}


def probe_scores(probe_emb: Tensor, gallery: Dict[str, Tensor]):
    """Max-similarity score and predicted identity for each probe."""
    emb = F.normalize(probe_emb, p=2, dim=-1)
    sims = emb @ gallery['templates'].t()          # (P, K)
    top_sim, top_idx = sims.max(dim=1)
    pred_ids = gallery['ids'][top_idx]
    return top_sim, pred_ids


def dir_at_far(known_scores: Tensor, known_correct: Tensor,
               unknown_scores: Tensor,
               far_targets: Sequence[float] = (0.01, 0.05, 0.1)) -> Dict[str, float]:
    """Detection & Identification Rate at target False Alarm Rates.

    Args:
        known_scores:   (Pk,) max-sim score for known probes.
        known_correct:  (Pk,) bool, whether rank-1 match is the true identity.
        unknown_scores: (Pu,) max-sim score for unknown probes.
        far_targets:    desired false-alarm rates on unknown probes.

    Returns:
        {'DIR@FAR=x': rate, ...} plus the thresholds used.
    """
    known_scores = known_scores.cpu().numpy()
    known_correct = known_correct.cpu().numpy().astype(bool)
    unknown_scores = unknown_scores.cpu().numpy()

    out = {}
    for far in far_targets:
        # Threshold t s.t. P(unknown accepted) == far  ->  (1-far) quantile.
        t = float(np.quantile(unknown_scores, 1.0 - far)) if len(unknown_scores) else 1.0
        accepted_and_correct = (known_scores > t) & known_correct
        dir_rate = float(accepted_and_correct.mean()) if len(known_scores) else 0.0
        out[f'DIR@FAR={far}'] = dir_rate
        out[f'threshold@FAR={far}'] = t
    return out


def openset_auc(known_scores: Tensor, unknown_scores: Tensor) -> float:
    """ROC-AUC of the accept/reject decision (known=positive)."""
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    s = np.concatenate([known_scores.cpu().numpy(), unknown_scores.cpu().numpy()])
    if len(np.unique(y)) < 2:
        return 0.0
    return float(roc_auc_score(y, s))


def evaluate_openset(gallery_emb: Tensor, gallery_lbl: Tensor,
                     probe_emb: Tensor, probe_lbl: Tensor,
                     known_ids: Sequence[int],
                     far_targets: Sequence[float] = (0.01, 0.05, 0.1)) -> Dict[str, object]:
    """Full open-set evaluation.

    Args:
        gallery_emb/lbl: enrolment embeddings & labels (e.g. train split).
        probe_emb/lbl:   probe embeddings & labels (e.g. test split).
        known_ids:       identities enrolled in the gallery; probes of any
                         other identity are treated as unknown/impostor.
        far_targets:     false-alarm rates for DIR reporting.
    """
    known = set(int(i) for i in known_ids)
    gallery = build_gallery(gallery_emb, gallery_lbl, sorted(known))

    top_sim, pred_ids = probe_scores(probe_emb, gallery)

    is_known = torch.tensor([int(l) in known for l in probe_lbl])
    known_scores = top_sim[is_known]
    unknown_scores = top_sim[~is_known]
    known_correct = (pred_ids[is_known] == probe_lbl[is_known])

    res = {
        'num_known_ids': len(known),
        'num_unknown_ids': int(len(set(int(l) for l in probe_lbl) - known)),
        'num_known_probes': int(is_known.sum()),
        'num_unknown_probes': int((~is_known).sum()),
        'openset_auc': openset_auc(known_scores, unknown_scores),
        'rank1_on_known_probes': float(known_correct.float().mean()) if known_correct.numel() else 0.0,
    }
    res.update(dir_at_far(known_scores, known_correct, unknown_scores, far_targets))
    return res
