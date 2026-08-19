"""
Multi-explainer sweep: is the coherence confound explainer-specific?
====================================================================
Our control showed an importance-BLIND coherent removal (`random_block`) matches the
"most important" condition for graph Grad-CAM, i.e. the standard top-vs-random ablation
test measures spatial coherence rather than attribution quality.

This sweeps four attributions to test how general that is, including the decisive one:

  gradcam         graph Grad-CAM (as before)
  rollout         multi-layer GATv2 attention rollout
  random          i.i.d. uniform node scores  -- no information, NOT smooth
  smoothed_random i.i.d. uniform scores AVERAGED over each node's spatial k-NN
                  -- still ZERO information, but SMOOTH over the graph

PREDICTION that decides the paper: `smoothed_random` should PASS the conventional
top-vs-random test despite carrying no information at all, because smoothing alone makes
its extremes spatially coherent. Plain `random` should show no gap. If so, the standard
protocol certifies noise, and the confound is a property of the TEST, not of any explainer.

Outputs: outputs/stats/explainer_sweep.json
Usage:   python scripts/experiment_explainer_sweep.py [--num-graphs N]
"""
import sys, argparse, json
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config, save_stats
from src.evaluation.faithfulness import _subgraph
from src.models.explainability import GradCAMGraph
from scripts.evaluate_explainability import (
    load_gnn, gradcam_importance, rollout_importance, _last_gat_layer_name, _prep)
from src.models.explainability import GNNExplainerWrapper
from scripts.experiment_causal_ablation import load_test_graphs, embed, bootstrap_ci
from scripts.experiment_coherence_control import select_removal, coherence, node_positions

EXPLAINERS = ['gradcam', 'rollout', 'random', 'smoothed_random']  # + 'gnnexplainer' via --with-gnnexplainer
STRATEGIES = ['top', 'bottom', 'random', 'random_block']


def smoothed_random_importance(data, rng, k=8):
    """Zero-information scores that are SMOOTH over the graph: uniform noise averaged
    over each node's k nearest spatial neighbours."""
    n = data.x.size(0)
    raw = rng.random(n)
    pos = node_positions(data)
    if pos is None:
        return torch.tensor(raw, dtype=torch.float)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    nb = np.argsort(d, axis=1)[:, :min(k, n)]
    return torch.tensor(raw[nb].mean(1), dtype=torch.float)


def get_importance(kind, model, g, dev, gc, rng, gnnexp=None):
    if kind == 'gnnexplainer':
        node_imp, _ = gnnexp.explain_graph(_prep(g, dev))
        return node_imp.detach().cpu()
    if kind == 'gradcam':
        return gradcam_importance(gc, g, dev).detach().cpu()
    if kind == 'rollout':
        return rollout_importance(model, g, dev).detach().cpu()
    if kind == 'random':
        return torch.tensor(rng.random(g.x.size(0)), dtype=torch.float)
    return smoothed_random_importance(g, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gnn_v3')
    ap.add_argument('--num-graphs', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--with-gnnexplainer', action='store_true',
                    help='add GNNExplainer (slow: gradient mask optimisation per graph)')
    ap.add_argument('--gnnexplainer-epochs', type=int, default=60)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    cfg = load_config(); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs, meta = load_test_graphs(cfg)
    # NOTE: load_gnn() falls back to gnn_v3's checkpoint_dir when config has no key for the
    # requested model, which SILENTLY loads the wrong weights. Load explicitly instead.
    from src.models.gnn_v3 import CattleGNNv3
    if args.model == 'gnn_v4':
        # gnn_v4 is architecturally distinct (see scripts/train_gnn_v4_enhanced.py):
        # hidden=128 x 8 heads = 1024-d head_out, 4 layers.
        _v4 = cfg.get('gnn_v4', {})
        model = CattleGNNv3(input_dim=256, hidden_dim=_v4.get('hidden_dim', 128),
                            num_heads=_v4.get('num_heads', 8),
                            num_layers=_v4.get('num_layers', 4))
    else:
        model = CattleGNNv3(config=cfg)
    if hasattr(model, 'set_num_classes'):
        model.set_num_classes(meta['num_classes'])
    _ckp = PROJECT_ROOT / f'outputs/{args.model}/best_model.pt'
    _st = torch.load(_ckp, map_location=dev, weights_only=False)
    model.load_state_dict(_st['model_state_dict'])
    model = model.to(dev).eval()
    print(f'[INFO] loaded {args.model} epoch={_st.get("epoch")} from {_ckp}', flush=True)
    gc = GradCAMGraph(model, target_layer_name=_last_gat_layer_name(model))

    idx = rng.choice(len(graphs), size=min(args.num_graphs, len(graphs)), replace=False)
    subset = [graphs[i] for i in idx]
    labels = torch.tensor([int(g.y) for g in subset])
    full = torch.stack([embed(model, g, dev) for g in subset])
    Sf = (full @ full.t()).numpy(); np.fill_diagonal(Sf, -1e9)
    full_pred = labels[Sf.argmax(1)]
    print(f"[INFO] {len(subset)} graphs\n", flush=True)

    explainers = list(EXPLAINERS)
    gnnexp = None
    if args.with_gnnexplainer:
        gnnexp = GNNExplainerWrapper(model, epochs=args.gnnexplainer_epochs)
        explainers.append('gnnexplainer')

    out = {'model': args.model, 'n': len(subset), 'frac': 0.30, 'explainers': {}}
    FRAC = 0.30
    for kind in explainers:
        imps = [get_importance(kind, model, g, dev, gc, rng, gnnexp) for g in subset]
        res = {}
        for strat in STRATEGIES:
            embs, cohs = [], []
            for g, imp in zip(subset, imps):
                rem, _ = select_removal(g, imp, FRAC, strat, rng)
                cohs.append(coherence(g, rem))
                keep = torch.tensor([i for i in range(g.x.size(0)) if i not in set(map(int, rem))],
                                    dtype=torch.long)
                embs.append(embed(model, _subgraph(g, keep), dev))
            ab = torch.stack(embs)
            dcos = (1 - (ab * full).sum(1)).clamp(min=0).numpy()
            Sa = (ab @ full.t()).numpy(); np.fill_diagonal(Sa, -1e9)
            flip = (labels[Sa.argmax(1)] != full_pred).numpy().astype(float)
            lo, hi = bootstrap_ci(dcos)
            res[strat] = {'dcosine': float(dcos.mean()), 'dcosine_ci': [lo, hi],
                          'top1_flip': float(flip.mean()),
                          'coherence': float(np.nanmean(cohs))}
        t, r, b = res['top'], res['random'], res['bottom']
        res['passes_top_vs_random'] = bool(t['dcosine_ci'][0] > r['dcosine_ci'][1])
        res['passes_top_vs_bottom'] = bool(t['dcosine_ci'][0] > b['dcosine_ci'][1])
        out['explainers'][kind] = res
        print(f"{kind:16s} top={t['dcosine']:.4f} random={r['dcosine']:.4f} bottom={b['dcosine']:.4f} "
              f"block={res['random_block']['dcosine']:.4f} | "
              f"top>random: {'PASS' if res['passes_top_vs_random'] else 'fail'} | "
              f"top>bottom: {'PASS' if res['passes_top_vs_bottom'] else 'fail'}", flush=True)
    gc.remove_hooks()

    _sfx = '' if args.model == 'gnn_v3' else '_' + args.model
    save_stats(out, str(PROJECT_ROOT / f'outputs/stats/explainer_sweep{_sfx}.json'))
    print(f'\nSaved -> outputs/stats/explainer_sweep{_sfx}.json')


if __name__ == '__main__':
    main()
