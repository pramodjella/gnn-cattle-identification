"""
Natural (non-synthetic) cross-domain shift on MegaDescriptor + S-norm
=====================================================================
The corruption probe uses a controlled synthetic shift. This script uses a
REAL, ecologically-valid domain shift — the field's preferred protocol — by
splitting gallery / probe on dataset metadata:
  * MacaqueFaces : by DATE (early dates enrol, later dates probe) -> temporal.
  * IPanda50     : by VIDEO (some videos enrol, other videos probe) -> session.

Reports verification EER for (a) a RANDOM split (in-domain reference) and
(b) the NATURAL cross-domain split, each baseline vs. + S-norm. The question:
does a real domain shift open a gap, and does label-free S-norm recover it?

Usage:  python scripts/wildlife_natural_shift.py --dataset MacaqueFaces
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
from scripts.wildlife_probe import load_megadescriptor, embed_paths, verif_eer_from_scores
from src.evaluation.calibration import snorm


def load_backbone(name, device):
    """Load a foundation backbone; returns (model, transform)."""
    if name == 'megadescriptor':
        return load_megadescriptor(device)
    if name == 'dinov2':
        import timm
        m = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        m.eval().to(device)
        cfg = timm.data.resolve_data_config({}, model=m)
        tf = timm.data.create_transform(**cfg)
        print(f"[probe] DINOv2 ViT-B/14 loaded (embed dim={m.num_features})")
        return m, tf
    raise ValueError(name)


def domain_of(df, dataset):
    """Return a per-row domain label for the natural split."""
    if dataset == 'MacaqueFaces':
        d = df['date'].astype(str)
        med = np.median(d.astype('category').cat.codes)
        return (d.astype('category').cat.codes > med).astype(int).to_numpy()  # 0=early,1=late
    if dataset == 'IPanda50':
        vid = df['path'].str.extract(r'_(v\d+)_')[0].fillna('v0')
        # per row: hash video to 0/1 deterministically
        return (vid.astype('category').cat.codes % 2).to_numpy()
    # fallback: random domain
    return np.random.RandomState(0).randint(0, 2, len(df))


def _snorm_rect(S):
    mu_r = S.mean(1, keepdims=True); sd_r = S.std(1, keepdims=True) + 1e-8
    mu_c = S.mean(0, keepdims=True); sd_c = S.std(0, keepdims=True) + 1e-8
    return 0.5 * ((S - mu_r) / sd_r + (S - mu_c) / sd_c)


def eer_pair(g_emb, g_lbl, p_emb, p_lbl, boot=0):
    """Baseline and S-norm EER for a gallery/probe split; optional bootstrap CI
    on the recovery (baseline EER - S-norm EER), resampling probe rows."""
    gids = sorted(set(g_lbl))
    templ = torch.stack([F.normalize(g_emb[g_lbl == c].mean(0), p=2, dim=-1) for c in gids])
    tlbl = np.array(gids)
    S = (p_emb @ templ.t()).numpy()
    base = verif_eer_from_scores(S, p_lbl, tlbl)
    sn = verif_eer_from_scores(_snorm_rect(S), p_lbl, tlbl)
    ci = None
    if boot:
        rng = np.random.RandomState(0); P = S.shape[0]; diffs = []
        for _ in range(boot):
            idx = rng.randint(0, P, P)
            b = verif_eer_from_scores(S[idx], p_lbl[idx], tlbl)
            s = verif_eer_from_scores(_snorm_rect(S[idx]), p_lbl[idx], tlbl)
            diffs.append(b - s)
        diffs = np.sort(diffs)
        ci = (float(np.mean(diffs)), float(diffs[int(0.025*boot)]), float(diffs[int(0.975*boot)]))
    return base, sn, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MacaqueFaces')
    ap.add_argument('--root', default='data/wildlife')
    ap.add_argument('--max-per-id', type=int, default=30)
    ap.add_argument('--backbone', default='megadescriptor', choices=['megadescriptor', 'dinov2'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from wildlife_datasets import datasets as wd
    ds_cls = getattr(wd, args.dataset)
    root = os.path.join(args.root, args.dataset)
    if not os.path.exists(root):
        ds_cls.get_data(root)
    df = ds_cls(root).df.reset_index(drop=True)
    df['domain'] = domain_of(df, args.dataset)

    # keep identities present in BOTH domains with enough images
    def enough(g):
        return (g['domain'] == 0).sum() >= 2 and (g['domain'] == 1).sum() >= 2
    keep = df.groupby('identity').filter(enough)['identity'].unique()
    df = df[df['identity'].isin(keep)].groupby('identity').head(args.max_per_id).reset_index(drop=True)
    paths = [os.path.join(root, p) for p in df['path']]
    lbl = df['identity'].astype('category').cat.codes.to_numpy()
    dom = df['domain'].to_numpy()
    print(f"[probe] {len(df)} imgs, {len(set(lbl))} ids in both domains "
          f"(domain0={int((dom==0).sum())}, domain1={int((dom==1).sum())})")

    model, tf = load_backbone(args.backbone, device)
    emb = embed_paths(model, tf, paths, device)

    # NATURAL cross-domain split: gallery=domain0, probe=domain1 (+ bootstrap CI)
    g0, p1 = dom == 0, dom == 1
    nat_base, nat_sn, ci = eer_pair(emb[g0], lbl[g0], emb[p1], lbl[p1], boot=400)

    # RANDOM (in-domain) reference: shuffle domain assignment
    rng = np.random.RandomState(0)
    rdom = np.zeros(len(lbl), int)
    for c in set(lbl):
        idx = np.where(lbl == c)[0]; rng.shuffle(idx); rdom[idx[:len(idx)//2]] = 1
    rg, rp = rdom == 0, rdom == 1
    rnd_base, rnd_sn, _ = eer_pair(emb[rg], lbl[rg], emb[rp], lbl[rp])

    print("\n" + "=" * 60)
    print(f"  {args.dataset} | {args.backbone} | natural shift = "
          f"{'DATE' if args.dataset=='MacaqueFaces' else 'VIDEO'}")
    print("=" * 60)
    print(f"  {'split':<22}{'baseline EER':>14}{'+S-norm EER':>14}")
    print(f"  {'random (in-domain)':<22}{rnd_base*100:>13.2f}%{rnd_sn*100:>13.2f}%")
    print(f"  {'natural (cross-dom)':<22}{nat_base*100:>13.2f}%{nat_sn*100:>13.2f}%")
    print("=" * 60)
    gap = nat_base - rnd_base
    rec = (nat_base - nat_sn) / nat_base * 100 if nat_base > 1e-6 else 0
    print(f"  domain gap (natural - random): {gap*100:+.2f} pts EER")
    print(f"  S-norm recovery on natural shift: {rec:+.1f}% relative")
    if ci:
        md, lo, hi = ci
        sig = 'SIGNIFICANT (CI excludes 0)' if lo > 0 else 'n.s.'
        print(f"  recovery (base-snorm) EER = {md*100:+.3f}%  95% CI [{lo*100:+.3f},{hi*100:+.3f}] -> {sig}")


if __name__ == '__main__':
    main()
