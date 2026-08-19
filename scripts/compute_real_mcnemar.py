"""
Real McNemar tests from per-probe predictions (replaces the simulated ones).
==========================================================================
The previous scripts/statistical_tests.py SIMULATED paired predictions from
aggregate Rank-1 scalars, which structurally forced n01=0 and produced invalid
p-values. This script embeds the test split with each trained model, records
per-probe closed-set Rank-1 correctness, and computes exact McNemar tests
(binomial, exact) on the real discordant pairs.

Outputs: outputs/stats/mcnemar_real.json
Usage:   python scripts/compute_real_mcnemar.py
"""
import sys, json
from pathlib import Path
from itertools import combinations
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config, save_stats
from src.training.image_dataset import create_hybrid_loaders
from scripts.experiment_pathway_intervention import (
    load_hybrid, load_cnn, embed_cnn, embed, branch_correct)
from scripts.experiment_quality_fusion import load_proton, embed_proton


def mcnemar_exact(a, b):
    """Exact (binomial) McNemar on two boolean correctness vectors."""
    from scipy.stats import binomtest
    n01 = int(np.sum(~a & b))      # a wrong, b right
    n10 = int(np.sum(a & ~b))      # a right, b wrong
    n = n01 + n10
    if n == 0:
        return {'n01': n01, 'n10': n10, 'p_value': 1.0, 'note': 'no discordant pairs'}
    p = binomtest(n10, n, 0.5).pvalue
    chi2 = (abs(n10 - n01) - 1) ** 2 / n          # continuity-corrected, for reference
    return {'n01': n01, 'n10': n10, 'n_discordant': n,
            'chi2_cc': float(chi2), 'p_value': float(p),
            'significant_0.05': bool(p < 0.05)}


def main():
    cfg = load_config()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = create_hybrid_loaders(str(PROJECT_ROOT / cfg['dataset']['processed_dir']),
                                    str(PROJECT_ROOT / cfg['dataset']['graph_dir']), cfg)
    test = loaders['test']

    print('[1/3] CNN ...', flush=True)
    cnn = load_cnn(dev)
    ce, lbl = embed_cnn(cnn, test, dev)
    del cnn; torch.cuda.empty_cache()

    print('[2/3] Hybrid ...', flush=True)
    hyb = load_hybrid(cfg, dev)
    he, _ = embed(hyb, test, dev, 'full')
    del hyb; torch.cuda.empty_cache()

    print('[3/3] ProtoN ...', flush=True)
    pro = load_proton(cfg, dev)
    pe, _ = embed_proton(pro, test, dev)
    del pro; torch.cuda.empty_cache()

    y = lbl.numpy()
    correct = {}
    for name, emb in [('CNN', ce), ('Hybrid', he), ('ProtoN', pe)]:
        S = (emb @ emb.t()).numpy()
        correct[name] = branch_correct(S, y)
        print(f"  {name:8s} Rank-1 = {correct[name].mean()*100:.2f}%")

    out = {'n_probes': int(len(y)),
           'rank1': {k: float(v.mean()) for k, v in correct.items()},
           'method': 'exact binomial McNemar on per-probe closed-set Rank-1 correctness',
           'tests': {}}
    for a, b in combinations(correct, 2):
        r = mcnemar_exact(correct[a], correct[b])
        out['tests'][f'{a}_vs_{b}'] = r
        print(f"  {a} vs {b}: n10={r['n10']} n01={r['n01']} p={r['p_value']:.3e}")

    np.save(PROJECT_ROOT / 'outputs/stats/per_probe_correct.npy',
            {k: v for k, v in correct.items()}, allow_pickle=True)
    save_stats(out, str(PROJECT_ROOT / 'outputs/stats/mcnemar_real.json'))
    print('\nSaved -> outputs/stats/mcnemar_real.json')


if __name__ == '__main__':
    main()
