"""
Main-Track Feasibility Probe: Score Calibration on MegaDescriptor (Wildlife)
===========================================================================
Tests the ONE question that decides whether the main-track pivot is real:

  Does label-free test-time score calibration (S-norm) recover the cross-domain
  verification gap of the SOTA foundation model (MegaDescriptor) on WILDLIFE
  re-identification — the same way it did for cattle muzzles?

If yes, the finding ("cross-domain degradation is a score-calibration problem,
recoverable label-free") is modality-agnostic and holds on the field's own
foundation model — a genuine main-track thesis. If no, stay with the domain paper.

Protocol (per dataset):
  * Embed all images with MegaDescriptor-L-384 (timm, hf-hub:BVRA/...).
  * Domain-aware split: gallery from one domain (e.g. earliest dates / one
    location), probes from a DIFFERENT domain -> cross-domain.
  * Metrics: verification EER and ROC-AUC, baseline cosine vs. + S-norm.
  * Also report a within-domain (random-split) EER as the "no-shift" reference.

Usage:
  python scripts/wildlife_probe.py --dataset MacaqueFaces --root data/wildlife
  (see wildlife_datasets for available dataset names / downloaders)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_megadescriptor(device):
    import timm
    model = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True)
    model.eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    dim = model.num_features
    print(f"[probe] MegaDescriptor-L-384 loaded (embed dim={dim})")
    return model, tf


@torch.no_grad()
def embed_paths(model, tf, paths, device, batch=32, corrupt='clean', severity=0):
    """Embed images, optionally applying a corruption (domain shift) first."""
    from PIL import Image
    from src.evaluation.corruptions import apply as apply_corr
    import torchvision.transforms as T
    embs, buf = [], []

    def _load(p):
        img = Image.open(p).convert('RGB')
        if corrupt != 'clean' and severity > 0:
            # corrupt in [0,1] tensor space at native res, then run tf
            t = T.ToTensor()(img)
            t = apply_corr(t, corrupt, severity, seed=abs(hash(p)) % 10000)
            img = T.ToPILImage()(t.clamp(0, 1))
        return tf(img)

    def flush(b):
        x = torch.stack([_load(p) for p in b]).to(device)
        e = model(x)
        if e.dim() > 2:
            e = e.mean(dim=(2, 3)) if e.dim() == 4 else e.mean(1)
        embs.append(F.normalize(e, p=2, dim=-1).float().cpu())
    for p in paths:
        buf.append(p)
        if len(buf) == batch:
            flush(buf); buf = []
    if buf:
        flush(buf)
    return torch.cat(embs)


def snorm_rect(S):
    """Adaptive symmetric S-norm on a rectangular probe x gallery score matrix."""
    S = S.astype(np.float64)
    mu_r = S.mean(1, keepdims=True); sd_r = S.std(1, keepdims=True) + 1e-8
    mu_c = S.mean(0, keepdims=True); sd_c = S.std(0, keepdims=True) + 1e-8
    return 0.5 * ((S - mu_r) / sd_r + (S - mu_c) / sd_c)


def align_cal(S, ref_mu, ref_sd):
    """Source-distribution alignment (novel): remap each shifted probe's score
    distribution onto the clean SOURCE reference so the operating threshold
    transfers. Per-probe Z-norm rescaled to the source (gallery-gallery)
    impostor statistics. Unsupervised, label-free.
    """
    S = S.astype(np.float64)
    mu_r = S.mean(1, keepdims=True); sd_r = S.std(1, keepdims=True) + 1e-8
    return (S - mu_r) / sd_r * ref_sd + ref_mu


def verif_eer_from_scores(S, probe_lbl, gal_lbl):
    """EER from a probe x gallery score matrix (genuine = same id)."""
    from sklearn.metrics import roc_curve
    gen, imp = [], []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            (gen if probe_lbl[i] == gal_lbl[j] else imp).append(S[i, j])
    y = [1] * len(gen) + [0] * len(imp)
    fpr, tpr, _ = roc_curve(y, gen + imp)
    fnr = 1 - tpr; k = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[k] + fnr[k]) / 2)


# ── S-norm (same as the cattle cross-dataset harness) ─────────────────────────
def snorm(sim):
    S = sim.copy().astype(np.float64)
    np.fill_diagonal(S, np.nan)
    mu = np.nanmean(S, axis=1, keepdims=True)
    sd = np.nanstd(S, axis=1, keepdims=True) + 1e-8
    Z = 0.5 * ((S - mu) / sd + (S - mu.T) / sd.T)
    np.fill_diagonal(Z, -np.inf)
    return Z


def eer_auc(emb, lbl):
    from sklearn.metrics import roc_curve, auc
    sim = (emb @ emb.t()).numpy()
    return _eer_auc_from_sim(sim, lbl.numpy())


def _eer_auc_from_sim(sim, lbl):
    from sklearn.metrics import roc_curve, auc
    n = len(lbl); gen, imp = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (gen if lbl[i] == lbl[j] else imp).append(sim[i, j])
    y = [1] * len(gen) + [0] * len(imp)
    fpr, tpr, _ = roc_curve(y, gen + imp)
    fnr = 1 - tpr
    eer = float((fpr[np.nanargmin(np.abs(fpr - fnr))] + fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2)
    return eer, float(auc(fpr, tpr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MacaqueFaces')
    ap.add_argument('--root', default='data/wildlife')
    ap.add_argument('--max-per-id', type=int, default=20)
    ap.add_argument('--min-per-id', type=int, default=4)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from wildlife_datasets import datasets as wd

    ds_cls = getattr(wd, args.dataset)
    root = os.path.join(args.root, args.dataset)
    if not os.path.exists(root):
        print(f"[probe] downloading {args.dataset} -> {root}")
        ds_cls.get_data(root)
    ds = ds_cls(root)
    df = ds.df.copy()

    # keep identities with enough images
    vc = df['identity'].value_counts()
    keep = vc[(vc >= args.min_per_id)].index
    df = df[df['identity'].isin(keep)]
    df = df.groupby('identity').head(args.max_per_id).reset_index(drop=True)
    paths = [os.path.join(root, p) for p in df['path']]
    labels = df['identity'].astype('category').cat.codes.to_numpy()
    print(f"[probe] {len(df)} images, {len(set(labels))} identities")

    model, tf = load_megadescriptor(device)

    # Split each identity 50/50 into gallery / probe.
    rng = np.random.RandomState(0)
    gal_idx, prb_idx = [], []
    for c in set(labels):
        ids = np.where(labels == c)[0]; rng.shuffle(ids)
        h = max(1, len(ids) // 2)
        gal_idx += list(ids[:h]); prb_idx += list(ids[h:])
    gal_idx, prb_idx = np.array(gal_idx), np.array(prb_idx)
    gpaths = [paths[i] for i in gal_idx]; ppaths = [paths[i] for i in prb_idx]
    gal_lbl, prb_lbl = labels[gal_idx], labels[prb_idx]

    print("[probe] embedding gallery (clean) ...")
    g_emb = embed_paths(model, tf, gpaths, device)
    # per-identity gallery templates
    gids = sorted(set(gal_lbl)); templ, tlbl = [], []
    for c in gids:
        m = gal_lbl == c
        templ.append(F.normalize(g_emb[m].mean(0), p=2, dim=-1)); tlbl.append(c)
    templ = torch.stack(templ); tlbl = np.array(tlbl)

    # SOURCE reference: clean gallery-gallery impostor score statistics.
    GG = (g_emb @ g_emb.t()).numpy()
    imp_ref = [GG[i, j] for i in range(len(gal_lbl)) for j in range(i + 1, len(gal_lbl))
               if gal_lbl[i] != gal_lbl[j]]
    ref_mu, ref_sd = float(np.mean(imp_ref)), float(np.std(imp_ref))

    print("\n" + "=" * 74)
    print(f"  DATASET: {args.dataset} | MegaDescriptor-L-384 | {len(set(labels))} ids")
    print("  CROSS-DOMAIN protocol: clean gallery vs corrupted probes (EER %, lower=better)")
    print("=" * 74)
    print(f"  {'shift':<16}{'baseline':>11}{'+S-norm':>11}{'+Align(ours)':>14}")

    for corr, sev in [('clean', 0), ('brightness', 3), ('spatter', 3), ('spatter', 5),
                      ('blur', 5)]:
        p_emb = embed_paths(model, tf, ppaths, device, corrupt=corr, severity=sev)
        S = (p_emb @ templ.t()).numpy()
        base = verif_eer_from_scores(S, prb_lbl, tlbl)
        sn = verif_eer_from_scores(snorm_rect(S), prb_lbl, tlbl)
        al = verif_eer_from_scores(align_cal(S, ref_mu, ref_sd), prb_lbl, tlbl)
        tag = 'clean (no shift)' if corr == 'clean' else f'{corr} sev{sev}'
        print(f"  {tag:<16}{base*100:>10.2f}%{sn*100:>10.2f}%{al*100:>13.2f}%")
    print("=" * 74)
    print("  Novelty check: does +Align (source-distribution matching) beat plain S-norm?")


if __name__ == '__main__':
    main()
