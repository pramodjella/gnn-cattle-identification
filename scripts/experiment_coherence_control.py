"""
Coherence-matched control: is graph-attribution "faithfulness" a coherence artefact?
====================================================================================
De-risk for a methods paper. Our Stage-2 result showed graph Grad-CAM passes the
conventional top-vs-RANDOM ablation test but fails top-vs-BOTTOM: removing the least
important nodes is nearly as damaging as removing the most. Notably random removal is
far LESS damaging than either extreme (dcos 0.048 vs ~0.10 at k=30).

HYPOTHESIS: the top-vs-random gap is not about importance at all. Top and bottom sets are
spatially/structurally COHERENT (they are extremes of a smoothly varying attribution),
whereas a uniformly random subset is scattered. Deleting a coherent block damages a graph
more than deleting the same number of scattered nodes -- regardless of importance.

DECISIVE TEST: add `random_block` -- remove a random spatially CONTIGUOUS set of the same
size (a random seed keypoint plus its k-1 nearest neighbours in image space). If
random_block ~ top, importance explains nothing and the standard protocol is confounded.

Also reports a coherence diagnostic per strategy: the mean pairwise distance among removed
keypoints (lower = more coherent), to verify the premise directly.

Outputs: outputs/stats/coherence_control.json
Usage:   python scripts/experiment_coherence_control.py [--num-graphs N]
"""
import sys, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config, save_stats
from src.evaluation.faithfulness import _subgraph
from src.models.explainability import GradCAMGraph
from scripts.evaluate_explainability import (
    load_gnn, gradcam_importance, _last_gat_layer_name, _prep)
from scripts.experiment_causal_ablation import load_test_graphs, embed, bootstrap_ci

STRATEGIES = ['top', 'bottom', 'random', 'random_block']


def node_positions(data):
    """(N,2) keypoint positions, or None."""
    p = getattr(data, 'pos', None)
    return p[:, :2].cpu().numpy() if p is not None else None


def select_removal(data, importance, frac, strategy, rng):
    """Indices to remove under each strategy (all remove the SAME count k)."""
    n = data.x.size(0)
    k = max(1, int(round(frac * n)))
    if n - k < 2:
        k = max(1, n - 2)
    order = torch.argsort(importance, descending=True)
    if strategy == 'top':
        return order[:k].cpu().numpy(), k
    if strategy == 'bottom':
        return order[-k:].cpu().numpy(), k
    if strategy == 'random':
        return rng.choice(n, size=k, replace=False), k
    # random_block: a random seed + its k-1 nearest spatial neighbours (coherent, importance-blind)
    pos = node_positions(data)
    if pos is None:
        return rng.choice(n, size=k, replace=False), k
    seed = int(rng.integers(n))
    d = np.linalg.norm(pos - pos[seed], axis=1)
    return np.argsort(d)[:k], k


def coherence(data, idx):
    """Mean pairwise distance among removed keypoints (lower = more coherent)."""
    pos = node_positions(data)
    if pos is None or len(idx) < 2:
        return float('nan')
    p = pos[np.asarray(idx)]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    iu = np.triu_indices(len(p), k=1)
    return float(d[iu].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gnn_v3')
    ap.add_argument('--num-graphs', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    cfg = load_config(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs, meta = load_test_graphs(cfg)
    model = load_gnn(args.model, cfg, dev, meta['num_classes'])
    gc = GradCAMGraph(model, target_layer_name=_last_gat_layer_name(model))

    idx = rng.choice(len(graphs), size=min(args.num_graphs, len(graphs)), replace=False)
    subset = [graphs[i] for i in idx]
    labels = torch.tensor([int(g.y) for g in subset])
    print(f"[INFO] {len(subset)} test graphs", flush=True)

    full = torch.stack([embed(model, g, dev) for g in subset])
    imps = [gradcam_importance(gc, g, dev).detach().cpu() for g in subset]
    gc.remove_hooks()
    Sf = (full @ full.t()).numpy(); np.fill_diagonal(Sf, -1e9)
    full_pred = labels[Sf.argmax(1)]

    out = {'model': args.model, 'n': len(subset),
           'hypothesis': 'random_block ~ top would show the top-vs-random test is a coherence artefact',
           'conditions': {}}
    for frac in (0.10, 0.20, 0.30):
        for strat in STRATEGIES:
            embs, cohs = [], []
            for g, imp in zip(subset, imps):
                rem, k = select_removal(g, imp, frac, strat, rng)
                cohs.append(coherence(g, rem))
                keep = torch.tensor([i for i in range(g.x.size(0)) if i not in set(map(int, rem))],
                                    dtype=torch.long)
                embs.append(embed(model, _subgraph(g, keep), dev))
            ab = torch.stack(embs)
            dcos = (1 - (ab * full).sum(1)).clamp(min=0).numpy()
            Sa = (ab @ full.t()).numpy(); np.fill_diagonal(Sa, -1e9)
            flip = (labels[Sa.argmax(1)] != full_pred).numpy().astype(float)
            lo, hi = bootstrap_ci(dcos)
            key = f'{strat}_{int(frac*100)}'
            out['conditions'][key] = {
                'dcosine': float(dcos.mean()), 'dcosine_ci': [lo, hi],
                'top1_flip': float(flip.mean()),
                'mean_pairwise_dist_removed': float(np.nanmean(cohs))}
            print(f"  {key:16s} dcos={dcos.mean():.4f} [{lo:.4f},{hi:.4f}]  "
                  f"flip={flip.mean()*100:5.1f}%  coherence(dist)={np.nanmean(cohs):.4f}", flush=True)

    # verdict
    v = {}
    for frac in (10, 20, 30):
        t = out['conditions'][f'top_{frac}']; rb = out['conditions'][f'random_block_{frac}']
        r = out['conditions'][f'random_{frac}']
        overlap = rb['dcosine_ci'][1] >= t['dcosine_ci'][0] and t['dcosine_ci'][1] >= rb['dcosine_ci'][0]
        v[f'k{frac}'] = {
            'top_dcos': t['dcosine'], 'random_block_dcos': rb['dcosine'], 'random_dcos': r['dcosine'],
            'block_matches_top(CIs overlap)': bool(overlap),
            'block_exceeds_random': bool(rb['dcosine'] > r['dcosine'])}
    out['verdict'] = v
    print('\n=== VERDICT ===')
    print(json.dumps(v, indent=1))
    save_stats(out, str(PROJECT_ROOT / 'outputs/stats/coherence_control.json'))
    print('\nSaved -> outputs/stats/coherence_control.json')


if __name__ == '__main__':
    main()
