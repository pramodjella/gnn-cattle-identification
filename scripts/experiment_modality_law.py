"""
Modality-law de-risk (main-track thesis).
=========================================
Question: does a LABEL-FREE structural statistic of the target embedding graph
predict WHEN per-input structural test-time signals (quality-conditioned
calibration; k-reciprocal re-ranking) help cross-domain re-ID? Thesis: the value
of structural signals is high on face-like modalities and low on repetitive
patterns, and this is predicted by neighbourhood-consistency of the graph.

Protocol per dataset (foundation backbone): keep identities with >= min images,
50/50 gallery/probe split; gallery embedded CLEAN, probe embedded under a domain
shift (spatter, matching the calibration bakeoff). For the probe x gallery cosine
matrix we compute EER for baseline / S-norm / quality-snorm / k-reciprocal->S-norm,
the structural-signal BENEFIT (S-norm EER - structural EER; +ve = structure helps)
with probe-bootstrap 95% CIs, and three LABEL-FREE statistics of the combined
target graph. Correlate statistic vs benefit across datasets (Spearman).

Everything is persisted to outputs/stats/modality_law.json (reproducible).

Usage:
  python scripts/experiment_modality_law.py --datasets MacaqueFaces CZoo IPanda50 \
      --backbone megadescriptor --corrupt spatter --severity 3
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import save_stats
from src.evaluation.calibration import quality_snorm
from src.evaluation.rerank import k_reciprocal_rerank
from scripts.wildlife_probe import (
    load_backbone, embed_paths, snorm_rect, verif_eer_from_scores)

# face-like vs repetitive-pattern prior (for reporting only; NOT used in the statistic)
MODALITY = {'MacaqueFaces': 'face', 'CZoo': 'face', 'CTai': 'face',
            'IPanda50': 'pattern', 'SeaTurtleID2022': 'pattern', 'NyalaData': 'pattern',
            'FriesianCattle2017': 'pattern', 'ZindiTurtleRecall': 'pattern'}


def load_split(dataset, root, min_per_id, max_per_id, seed=0):
    """Load a wildlife dataset and make a 50/50 per-identity gallery/probe split."""
    from wildlife_datasets import datasets as wd
    ds_root = os.path.join(root, dataset)
    ds = getattr(wd, dataset)(ds_root)
    df = ds.df.copy()
    vc = df['identity'].value_counts()
    df = df[df['identity'].isin(vc[vc >= min_per_id].index)]
    df = df.groupby('identity').head(max_per_id).reset_index(drop=True)
    paths = [os.path.join(ds_root, p) for p in df['path']]
    labels = df['identity'].astype('category').cat.codes.to_numpy()
    rng = np.random.RandomState(seed)
    gal, prb = [], []
    for c in set(labels):
        ids = np.where(labels == c)[0]; rng.shuffle(ids)
        h = max(1, len(ids) // 2)
        gal += list(ids[:h]); prb += list(ids[h:])
    gal, prb = np.array(gal), np.array(prb)
    return ([paths[i] for i in gal], labels[gal],
            [paths[i] for i in prb], labels[prb])


def structural_stats(emb, k=20, cap=2500, seed=0):
    """Three LABEL-FREE statistics of an embedding graph (no identity labels).
    - reciprocal_consistency: mean Jaccard overlap of k-NN sets between a node and
      its k-NN (high => locally coherent manifold; the quantity k-reciprocal uses).
    - reciprocity_rate: fraction of directed k-NN edges that are mutual.
    - top1_margin: mean(top-1 sim - mean of top 2..k) — neighbourhood sharpness.
    """
    E = emb.numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb)
    if len(E) > cap:
        idx = np.random.RandomState(seed).choice(len(E), cap, replace=False)
        E = E[idx]
    N = len(E); k = min(k, N - 1)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    knn = np.argpartition(-S, kth=k, axis=1)[:, :k]                 # (N,k) top-k
    M = np.zeros((N, N), dtype=np.float32)
    M[np.repeat(np.arange(N), k), knn.reshape(-1)] = 1.0
    inter = M @ M.T                                                 # intersection counts
    ii = np.repeat(np.arange(N), k); jj = knn.reshape(-1)
    interij = inter[ii, jj]
    consistency = float((interij / (2 * k - interij + 1e-9)).mean())
    reciprocity = float(M[jj, ii].mean())                          # i in knn(j)?
    Ssort = np.sort(S, axis=1)[:, ::-1]
    margin = float((Ssort[:, 0] - Ssort[:, 1:k].mean(1)).mean())
    return {'reciprocal_consistency': consistency,
            'reciprocity_rate': reciprocity, 'top1_margin': margin}


def bootstrap_benefit(S_snorm, S_struct, prb_lbl, gal_lbl, n_boot=400, seed=0):
    """95% CI of (EER_snorm - EER_struct) by resampling probes."""
    rng = np.random.RandomState(seed)
    P = S_snorm.shape[0]; diffs = []
    for _ in range(n_boot):
        r = rng.randint(0, P, P)
        e_s = verif_eer_from_scores(S_snorm[r], prb_lbl[r], gal_lbl)
        e_t = verif_eer_from_scores(S_struct[r], prb_lbl[r], gal_lbl)
        diffs.append(e_s - e_t)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_dataset(name, args, model, tf, device):
    gpaths, gal_lbl, ppaths, prb_lbl = load_split(
        name, args.root, args.min_per_id, args.max_per_id)
    g_emb = embed_paths(model, tf, gpaths, device)
    p_emb = embed_paths(model, tf, ppaths, device, corrupt=args.corrupt, severity=args.severity)
    # Per-identity mean templates (matches the calibration-bakeoff enrollment protocol —
    # the few-column setting where the quality/structural signal lives).
    gids = sorted(set(gal_lbl))
    templ = torch.stack([F.normalize(g_emb[gal_lbl == c].mean(0), p=2, dim=-1) for c in gids])
    tlbl = np.array(gids)
    S = (p_emb @ templ.t()).numpy()

    e_base = verif_eer_from_scores(S, prb_lbl, tlbl)
    S_sn = snorm_rect(S); e_sn = verif_eer_from_scores(S_sn, prb_lbl, tlbl)
    S_qs = quality_snorm(S); e_qs = verif_eer_from_scores(S_qs, prb_lbl, tlbl)
    sim_gg = (templ @ templ.t()).numpy()
    S_kr = snorm_rect(k_reciprocal_rerank(S, sim_gg))
    e_kr = verif_eer_from_scores(S_kr, prb_lbl, tlbl)

    # Label-free statistic on the raw probe+gallery image embeddings (characterises the
    # modality's embedding geometry, independent of enrollment).
    stat = structural_stats(torch.cat([g_emb, p_emb]))
    qs_lo, qs_hi = bootstrap_benefit(S_sn, S_qs, prb_lbl, tlbl)
    kr_lo, kr_hi = bootstrap_benefit(S_sn, S_kr, prb_lbl, tlbl)
    return {
        'modality_prior': MODALITY.get(name, '?'),
        'n_gallery': len(gal_lbl), 'n_probe': len(prb_lbl), 'n_ids': int(len(gids)),
        'eer': {'baseline': e_base, 's_norm': e_sn, 'quality_snorm': e_qs, 'k_recip_snorm': e_kr},
        'benefit_qs': e_sn - e_qs, 'benefit_qs_ci': [qs_lo, qs_hi],
        'benefit_kr': e_sn - e_kr, 'benefit_kr_ci': [kr_lo, kr_hi],
        'stats': stat,
    }


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    if len(x) < 2 or np.std(rx) == 0 or np.std(ry) == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['MacaqueFaces', 'CZoo', 'IPanda50'])
    ap.add_argument('--root', default='data/wildlife')
    ap.add_argument('--backbone', default='megadescriptor', choices=['megadescriptor', 'dinov2'])
    ap.add_argument('--corrupt', default='spatter')
    ap.add_argument('--severity', type=int, default=3)
    ap.add_argument('--min-per-id', type=int, default=4)
    ap.add_argument('--max-per-id', type=int, default=15)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tf = load_backbone(args.backbone, device)

    out = {'backbone': args.backbone, 'shift': f'{args.corrupt}:{args.severity}', 'datasets': {}}
    for name in args.datasets:
        print(f"\n=== {name} ===", flush=True)
        try:
            r = run_dataset(name, args, model, tf, device)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}", flush=True); continue
        out['datasets'][name] = r
        print(f"  {r['modality_prior']:8s} consistency={r['stats']['reciprocal_consistency']:.3f}  "
              f"qs_benefit={r['benefit_qs']*100:+.2f} CI[{r['benefit_qs_ci'][0]*100:+.2f},{r['benefit_qs_ci'][1]*100:+.2f}]  "
              f"kr_benefit={r['benefit_kr']*100:+.2f} CI[{r['benefit_kr_ci'][0]*100:+.2f},{r['benefit_kr_ci'][1]*100:+.2f}]", flush=True)

    # correlation across datasets: does the label-free statistic predict the benefit?
    names = list(out['datasets'])
    if len(names) >= 2:
        cons = [out['datasets'][n]['stats']['reciprocal_consistency'] for n in names]
        recp = [out['datasets'][n]['stats']['reciprocity_rate'] for n in names]
        for bname in ('benefit_qs', 'benefit_kr'):
            b = [out['datasets'][n][bname] for n in names]
            out.setdefault('correlation', {})[f'consistency_vs_{bname}'] = spearman(cons, b)
            out['correlation'][f'reciprocity_vs_{bname}'] = spearman(recp, b)
        print("\n=== correlation (label-free statistic vs structural benefit) ===")
        print(json.dumps(out['correlation'], indent=1))
    save_stats(out, str(PROJECT_ROOT / f'outputs/stats/modality_law_{args.backbone}.json'))
    print(f"\nSaved -> outputs/stats/modality_law_{args.backbone}.json")


if __name__ == '__main__':
    main()
