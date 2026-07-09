"""
Calibration bake-off: can any variant beat plain S-norm?  (main-track crux)
==========================================================================
Embeds a wildlife dataset once with MegaDescriptor, builds a clean gallery vs
corrupted-probe cross-domain protocol, and compares every calibration method
in src/evaluation/calibration.py by verification EER.

Usage:  python scripts/calibration_bakeoff.py --dataset FriesianCattle2017
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
from src.evaluation.calibration import METHODS
from scripts.wildlife_probe import load_backbone, embed_paths, verif_eer_from_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='FriesianCattle2017')
    ap.add_argument('--root', default='data/wildlife')
    ap.add_argument('--shifts', nargs='+', default=['clean:0', 'spatter:3', 'spatter:5', 'blur:5'])
    ap.add_argument('--backbone', default='megadescriptor', choices=['megadescriptor', 'dinov2'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from wildlife_datasets import datasets as wd
    ds_cls = getattr(wd, args.dataset)
    root = os.path.join(args.root, args.dataset)
    if not os.path.exists(root):
        ds_cls.get_data(root)
    df = ds_cls(root).df.copy()
    vc = df['identity'].value_counts()
    df = df[df['identity'].isin(vc[vc >= 4].index)].groupby('identity').head(20).reset_index(drop=True)
    paths = [os.path.join(root, p) for p in df['path']]
    labels = df['identity'].astype('category').cat.codes.to_numpy()

    rng = np.random.RandomState(0)
    gal_idx, prb_idx = [], []
    for c in set(labels):
        ids = np.where(labels == c)[0]; rng.shuffle(ids); h = max(1, len(ids) // 2)
        gal_idx += list(ids[:h]); prb_idx += list(ids[h:])
    gal_idx, prb_idx = np.array(gal_idx), np.array(prb_idx)
    gpaths = [paths[i] for i in gal_idx]; ppaths = [paths[i] for i in prb_idx]
    gal_lbl, prb_lbl = labels[gal_idx], labels[prb_idx]

    model, tf = load_backbone(args.backbone, device)
    g_emb = embed_paths(model, tf, gpaths, device)
    gids = sorted(set(gal_lbl))
    templ = torch.stack([F.normalize(g_emb[gal_lbl == c].mean(0), p=2, dim=-1) for c in gids])
    tlbl = np.array(gids)

    def eer_from_S(S):
        return verif_eer_from_scores(S, prb_lbl, tlbl)

    def bootstrap_diff(S_a, S_b, n=300):
        """Bootstrap CI of EER(a) - EER(b) by resampling probes (rows)."""
        rng = np.random.RandomState(0); P = S_a.shape[0]; diffs = []
        for _ in range(n):
            idx = rng.randint(0, P, P)
            da = verif_eer_from_scores(S_a[idx], prb_lbl[idx], tlbl)
            db = verif_eer_from_scores(S_b[idx], prb_lbl[idx], tlbl)
            diffs.append(da - db)
        diffs = np.sort(diffs)
        return float(np.mean(diffs)), float(diffs[int(0.025*n)]), float(diffs[int(0.975*n)])

    names = list(METHODS.keys())
    print(f"\n  DATASET: {args.dataset} | MegaDescriptor-L | {len(gids)} ids")
    print("  EER (%) by calibration method across cross-domain shifts (lower=better)")
    header = "  {:<14}".format('shift') + "".join(f"{n:>15}" for n in names)
    print(header); print("  " + "-" * (len(header) - 2))
    results = {}
    for spec in args.shifts:
        corr, sev = spec.split(':'); sev = int(sev)
        p_emb = embed_paths(model, tf, ppaths, device, corrupt=corr, severity=sev)
        S = (p_emb @ templ.t()).numpy()
        row = {}
        for n in names:
            eer = verif_eer_from_scores(METHODS[n](S), prb_lbl, tlbl)
            row[n] = eer
        results[spec] = row
        print("  {:<14}".format(spec) + "".join(f"{row[n]*100:>14.2f}%" for n in names))
        # significance: is quality-snorm better than plain s-norm? (EER diff, 95% CI)
        if not spec.startswith('clean'):
            Ssn = METHODS['s-norm'](S); Sqs = METHODS['quality-snorm'](S)
            md, lo, hi = bootstrap_diff(Ssn, Sqs)  # s-norm minus quality-snorm; >0 => qsnorm better
            sig = 'SIGNIFICANT' if lo > 0 else ('sig(qs worse)' if hi < 0 else 'n.s.')
            results[spec]['_qs_vs_snorm_eerdiff'] = {'mean': md, 'ci': [lo, hi]}
            print(f"                   qs vs s-norm EER diff = {md*100:+.3f}% "
                  f"(95% CI [{lo*100:+.3f},{hi*100:+.3f}]) -> {sig}")

    # winner tally under real shift (exclude clean)
    print("  " + "-" * (len(header) - 2))
    shifted = [s for s in args.shifts if not s.startswith('clean')]
    print("  Best method per shifted case (vs s-norm):")
    for s in shifted:
        r = {k: v for k, v in results[s].items() if not k.startswith('_')}
        best = min(r, key=r.get)
        beats = "BEATS s-norm" if r[best] < r['s-norm'] - 1e-6 and best != 's-norm' else "s-norm best/tied"
        print(f"    {s:<12} winner={best:<16} ({r[best]*100:.2f}%)  [{beats}]")

    # Accumulate into a cumulative evidence JSON for the empirical paper.
    from src.utils import save_stats, load_stats
    out_path = PROJECT_ROOT / 'outputs/stats/calibration_bakeoff.json'
    acc = {}
    if out_path.exists():
        try:
            acc = load_stats(str(out_path))
        except Exception:
            acc = {}
    acc[args.dataset] = {'num_ids': len(gids), 'results': results, 'methods': names}
    save_stats(acc, str(out_path))
    print(f"\n  Accumulated -> outputs/stats/calibration_bakeoff.json ({len(acc)} datasets)")


if __name__ == '__main__':
    main()
