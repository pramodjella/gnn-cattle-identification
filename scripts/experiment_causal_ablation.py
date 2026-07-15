"""
Part 2, Stage 2 of the Research Extension Plan: Causal ablation of explanations.
================================================================================
Tests whether Grad-CAM node importance is *causally* faithful by removing nodes
and measuring the effect on the identity embedding:
  * top-k%    -- remove the k% MOST important nodes (should hurt most)
  * random-k% -- remove a random k% (control)
  * bottom-k% -- remove the k% LEAST important nodes (should hurt least)
for k in {10, 20, 30}. Per-condition measures:
  * dcosine     -- 1 - cos(full_embedding, ablated_embedding)         [mean]
  * top1_flip   -- fraction of probes whose nearest-neighbour identity changes
  * rank1_drop  -- closed-set Rank-1 drop vs full graphs (percentage points)
  * eer_incr    -- EER increase vs full graphs (percentage points)
A causally faithful explainer => top-k >> random-k >~ bottom-k on every measure.

Outputs: outputs/stats/causal_ablation.json
Usage: python scripts/experiment_causal_ablation.py [--model gnn_v3] [--num-graphs 120]
"""
import os, sys, argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config, save_stats
from src.evaluation.metrics import BiometricMetrics
from src.evaluation.faithfulness import _subgraph
from src.models.explainability import GradCAMGraph

# reuse loaders / attribution helpers from the existing explainability script
from scripts.evaluate_explainability import (
    load_gnn, gradcam_importance, _last_gat_layer_name, _prep)


def load_test_graphs(config):
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']
    with open(graph_dir / 'label_mapping.json') as f:
        num_classes = len(json.load(f))
    graphs = torch.load(graph_dir / 'test_graphs.pt', weights_only=False)
    return graphs, {'num_classes': num_classes}


@torch.no_grad()
def embed(model, data, device):
    out = model(_prep(data, device))
    return F.normalize(out['embedding'], p=2, dim=-1).squeeze(0).cpu()


def ablate(data, importance, frac, strategy, rng):
    """Return subgraph with `frac` of nodes removed per `strategy`."""
    n = data.x.size(0)
    k = max(1, int(round(frac * n)))
    if n - k < 2:                       # keep the graph non-degenerate
        k = max(1, n - 2)
    order = torch.argsort(importance, descending=True)  # most->least important
    if strategy == 'top':
        remove = order[:k]
    elif strategy == 'bottom':
        remove = order[-k:]
    else:                               # random
        remove = torch.as_tensor(rng.choice(n, size=k, replace=False))
    keep = torch.tensor([i for i in range(n) if i not in set(remove.tolist())],
                        dtype=torch.long)
    return _subgraph(data, keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gnn_v3', choices=['gnn_v3', 'gnn_v4'])
    ap.add_argument('--num-graphs', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graphs, meta = load_test_graphs(config)
    model = load_gnn(args.model, config, device, meta['num_classes'])
    gradcam = GradCAMGraph(model, target_layer_name=_last_gat_layer_name(model))

    idx = rng.choice(len(graphs), size=min(args.num_graphs, len(graphs)), replace=False)
    subset = [graphs[i] for i in idx]
    labels = torch.tensor([int(g.y) for g in subset])

    # full-graph reference embeddings + importances
    full_emb = torch.stack([embed(model, g, device) for g in subset])
    imps = [gradcam_importance(gradcam, g, device).detach().cpu() for g in subset]
    gradcam.remove_hooks()

    M = BiometricMetrics()
    full_summary = M.compute_all_metrics(full_emb, labels)['summary']
    full_r1, full_eer = full_summary['rank_1_accuracy'], full_summary['eer']

    # nearest-neighbour identity in the FULL gallery (for flip detection)
    Sfull = (full_emb @ full_emb.t()).numpy(); np.fill_diagonal(Sfull, -1e9)
    full_pred = labels[Sfull.argmax(1)]

    fracs = [0.10, 0.20, 0.30]
    strategies = ['top', 'random', 'bottom']
    results = {'model': args.model, 'n': len(subset),
               'full': {'rank1': full_r1, 'eer': full_eer}, 'conditions': {}}

    for frac in fracs:
        for strat in strategies:
            abl_emb = torch.stack([
                embed(model, ablate(g, imp, frac, strat, rng), device)
                for g, imp in zip(subset, imps)])
            dcos = (1 - (abl_emb * full_emb).sum(1)).clamp(min=0).mean().item()
            # flip: nearest neighbour of ablated probe in the full gallery
            Sab = (abl_emb @ full_emb.t()).numpy(); np.fill_diagonal(Sab, -1e9)
            abl_pred = labels[Sab.argmax(1)]
            flip = float((abl_pred != full_pred).float().mean())
            s = M.compute_all_metrics(abl_emb, labels)['summary']
            key = f'{strat}_{int(frac*100)}'
            results['conditions'][key] = {
                'dcosine': dcos, 'top1_flip': flip,
                'rank1_drop': (full_r1 - s['rank_1_accuracy']) * 100,
                'eer_incr': (s['eer'] - full_eer) * 100,
                'rank1': s['rank_1_accuracy'], 'eer': s['eer']}
            print(f"  {key:10s} dcos={dcos:.4f} flip={flip*100:5.1f}%  "
                  f"R1drop={results['conditions'][key]['rank1_drop']:+5.1f}  "
                  f"EER+={results['conditions'][key]['eer_incr']:+5.2f}")

    save_stats(results, str(PROJECT_ROOT / 'outputs/stats/causal_ablation.json'))
    print("\nSaved -> outputs/stats/causal_ablation.json")


if __name__ == '__main__':
    main()
