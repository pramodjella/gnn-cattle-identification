"""
Do topology (k-reciprocal) and calibration (S-norm) combine?  (GraphCal de-risk)
==============================================================================
Tests whether structure-based re-ranking and distribution-based calibration are
complementary on cross-domain re-ID. If the combination beats either alone, a
learned graph calibrator that unifies them has headroom.

Usage: python scripts/rerank_experiment.py --dataset MacaqueFaces --backbone dinov2
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.wildlife_probe import load_backbone, embed_paths, verif_eer_from_scores
from scripts.wildlife_natural_shift import domain_of, _snorm_rect
from src.evaluation.rerank import k_reciprocal_rerank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MacaqueFaces')
    ap.add_argument('--root', default='data/wildlife')
    ap.add_argument('--backbone', default='dinov2', choices=['megadescriptor', 'dinov2'])
    ap.add_argument('--max-per-id', type=int, default=25)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from wildlife_datasets import datasets as wd
    ds_cls = getattr(wd, args.dataset)
    root = os.path.join(args.root, args.dataset)
    if not os.path.exists(root):
        ds_cls.get_data(root)
    df = ds_cls(root).df.reset_index(drop=True)
    df['domain'] = domain_of(df, args.dataset)
    def enough(g): return (g['domain'] == 0).sum() >= 2 and (g['domain'] == 1).sum() >= 2
    keep = df.groupby('identity').filter(enough)['identity'].unique()
    df = df[df['identity'].isin(keep)].groupby('identity').head(args.max_per_id).reset_index(drop=True)
    paths = [os.path.join(root, p) for p in df['path']]
    lbl = df['identity'].astype('category').cat.codes.to_numpy()
    dom = df['domain'].to_numpy()

    model, tf = load_backbone(args.backbone, device)
    emb = embed_paths(model, tf, paths, device)

    g0, p1 = dom == 0, dom == 1
    g_emb, g_lbl = emb[g0], lbl[g0]
    p_emb, p_lbl = emb[p1], lbl[p1]
    gids = sorted(set(g_lbl))
    templ = torch.stack([F.normalize(g_emb[g_lbl == c].mean(0), p=2, dim=-1) for c in gids])
    tlbl = np.array(gids)

    sim_pg = (p_emb @ templ.t()).numpy()
    sim_gg = (templ @ templ.t()).numpy()

    def eer(S): return verif_eer_from_scores(S, p_lbl, tlbl)

    base = eer(sim_pg)
    sn = eer(_snorm_rect(sim_pg))
    kr = eer(k_reciprocal_rerank(sim_pg, sim_gg))
    sn_then_kr = eer(k_reciprocal_rerank(_snorm_rect(sim_pg), sim_gg))
    kr_then_sn = eer(_snorm_rect(k_reciprocal_rerank(sim_pg, sim_gg)))

    print("\n" + "=" * 58)
    print(f"  {args.dataset} | {args.backbone} | cross-domain (natural)")
    print("=" * 58)
    print(f"  baseline (cosine)        EER {base*100:6.2f}%")
    print(f"  + S-norm                 EER {sn*100:6.2f}%")
    print(f"  + k-reciprocal           EER {kr*100:6.2f}%")
    print(f"  + S-norm then k-recip    EER {sn_then_kr*100:6.2f}%")
    print(f"  + k-recip then S-norm    EER {kr_then_sn*100:6.2f}%")
    print("=" * 58)
    best_single = min(sn, kr)
    best_combo = min(sn_then_kr, kr_then_sn)
    verdict = ("COMPLEMENTARY: combo beats best single by "
               f"{(best_single-best_combo)*100:.2f}pt" if best_combo < best_single - 1e-6
               else "NOT complementary: combo <= best single")
    print(f"  best single={best_single*100:.2f}%  best combo={best_combo*100:.2f}%  -> {verdict}")


if __name__ == '__main__':
    main()
