"""
Quantitative Explainability Evaluation
======================================
Moves the paper's explainability claims from "here is a nice heatmap" to
"our explanations are measurably faithful and consistent across methods".

For a trained pure-GNN model it computes, over a sample of test graphs:

  * Fidelity+  (comprehensiveness) and Fidelity- (sufficiency)
  * Explanation sparsity
  * Cross-method agreement (Spearman) between Attention Rollout, Graph
    Grad-CAM and (optionally) GNNExplainer

Results feed directly into a new "Explainability" results table and figure.

Usage:
    python scripts/evaluate_explainability.py --model gnn_v3 --num-graphs 40
    python scripts/evaluate_explainability.py --model gnn_v3 --with-gnnexplainer

Outputs:
    outputs/stats/explainability_faithfulness.json
    outputs/figures/explainability/faithfulness_summary.png
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats
from src.models.explainability import AttentionRollout, GradCAMGraph, GNNExplainerWrapper
from src.evaluation.faithfulness import (
    GraphFaithfulness, explanation_agreement,
)


def load_gnn(model_name, config, device, num_classes):
    """Load a pure-GNN checkpoint (gnn_v3 / gnn_v4)."""
    if model_name in ('gnn_v3', 'gnn_v4'):
        from src.models.gnn_v3 import CattleGNNv3
        model = CattleGNNv3(config=config)
        ckpt_dir = PROJECT_ROOT / config.get(model_name, config.get('gnn_v3', {})).get(
            'checkpoint_dir', f'outputs/{model_name}')
    else:
        raise ValueError(f"Unsupported model for faithfulness eval: {model_name}")

    if hasattr(model, 'set_num_classes'):
        model.set_num_classes(num_classes)

    ckpt_path = ckpt_dir / 'best_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    print(f"[INFO] Loaded {model_name} (epoch {ckpt.get('epoch', '?')}, "
          f"val R1={ckpt.get('val_r1', ckpt.get('best_val_acc', 0)):.4f})")
    return model


def _prep(data, device):
    """Move to device, add batch vector, drop y (avoids training loss branch)."""
    data = data.clone().to(device)
    if getattr(data, 'y', None) is not None:
        data.y = None
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    return data


def rollout_importance(model, data, device):
    """Node importance from multi-layer GATv2 attention rollout."""
    data = _prep(data, device)
    with torch.no_grad():
        out = model(data)
    attn = out.get('attention', None)
    if attn is None:
        return torch.ones(data.x.size(0), device=device)
    if not isinstance(attn, (list, tuple)):
        attn = [attn]
    ro = AttentionRollout(add_residual=True)
    return ro.compute(list(attn), data.edge_index, data.x.size(0))


def gradcam_importance(gradcam, data, device):
    """Node importance from Graph Grad-CAM."""
    return gradcam.attribute(_prep(data, device))


def _last_gat_layer_name(model):
    """Resolve the last GATv2 layer path for Grad-CAM hooks."""
    if hasattr(model, 'gat_layers') and len(model.gat_layers) > 0:
        return f'gat_layers.{len(model.gat_layers) - 1}'
    raise AttributeError("Model has no 'gat_layers' to hook for Grad-CAM.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gnn_v3', choices=['gnn_v3', 'gnn_v4'])
    ap.add_argument('--split', default='test')
    ap.add_argument('--num-graphs', type=int, default=40,
                    help='Number of test graphs to evaluate (CPU-friendly default).')
    ap.add_argument('--with-gnnexplainer', action='store_true',
                    help='Also run GNNExplainer (slow, gradient mask optimisation).')
    ap.add_argument('--gnnexplainer-epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']
    with open(graph_dir / 'label_mapping.json') as f:
        num_classes = len(json.load(f))

    model = load_gnn(args.model, config, device, num_classes)

    graphs = torch.load(graph_dir / f'{args.split}_graphs.pt', weights_only=False)
    idx = np.random.choice(len(graphs), size=min(args.num_graphs, len(graphs)), replace=False)
    sample = [graphs[i] for i in idx]
    print(f"[INFO] Evaluating faithfulness on {len(sample)} {args.split} graphs")

    faith = GraphFaithfulness(model, device=device)
    if faith.prototypes is None:
        print("[WARN] No ArcFace prototypes found; class scores fall back to "
              "embedding norm (fidelity still directional but less calibrated).")

    gradcam = GradCAMGraph(model, target_layer_name=_last_gat_layer_name(model))
    gnnexp = None
    if args.with_gnnexplainer:
        gnnexp = GNNExplainerWrapper(model, epochs=args.gnnexplainer_epochs)

    # ── Per-graph loop: importances + faithfulness + agreement ────────────────
    methods = ['attention_rollout', 'grad_cam']
    if gnnexp is not None:
        methods.append('gnn_explainer')

    fid = {m: {'plus': [], 'minus': [], 'sparsity': []} for m in methods}
    agreements = []

    for i, g in enumerate(sample):
        importances = {}

        try:
            importances['attention_rollout'] = rollout_importance(model, g, device)
        except Exception as e:
            print(f"  [g{i}] rollout failed: {e}")

        try:
            importances['grad_cam'] = gradcam_importance(gradcam, g, device)
        except Exception as e:
            print(f"  [g{i}] grad-cam failed: {e}")

        if gnnexp is not None:
            try:
                node_imp, _ = gnnexp.explain_graph(g)
                importances['gnn_explainer'] = node_imp
            except Exception as e:
                print(f"  [g{i}] gnnexplainer failed: {e}")

        # Faithfulness per available method.
        for m, imp in importances.items():
            try:
                res = faith.fidelity(g, imp)
                fid[m]['plus'].append(res['fidelity_plus'])
                fid[m]['minus'].append(res['fidelity_minus'])
                fid[m]['sparsity'].append(GraphFaithfulness.sparsity(imp))
            except Exception as e:
                print(f"  [g{i}] fidelity({m}) failed: {e}")

        # Cross-method agreement (needs >=2 methods on the same graph).
        if len(importances) >= 2:
            ag = explanation_agreement(importances)
            agreements.append(ag['mean_agreement'])

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(sample)} graphs")

    gradcam.remove_hooks()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _ms(v):
        return {'mean': float(np.mean(v)) if v else 0.0,
                'std': float(np.std(v)) if v else 0.0, 'n': len(v)}

    summary = {'model': args.model, 'split': args.split,
               'num_graphs': len(sample), 'methods': {}}
    for m in methods:
        summary['methods'][m] = {
            'fidelity_plus': _ms(fid[m]['plus']),
            'fidelity_minus': _ms(fid[m]['minus']),
            'sparsity': _ms(fid[m]['sparsity']),
        }
    summary['cross_method_agreement'] = _ms(agreements)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EXPLAINABILITY FAITHFULNESS SUMMARY")
    print("=" * 72)
    print(f"  {'Method':<20} {'Fidelity+':>14} {'Fidelity-':>14} {'Sparsity':>12}")
    print(f"  {'':<20} {'(higher=better)':>14} {'(lower=better)':>14} {'':>12}")
    print("  " + "-" * 62)
    for m in methods:
        s = summary['methods'][m]
        print(f"  {m:<20} "
              f"{s['fidelity_plus']['mean']:>7.4f}±{s['fidelity_plus']['std']:<5.3f} "
              f"{s['fidelity_minus']['mean']:>7.4f}±{s['fidelity_minus']['std']:<5.3f} "
              f"{s['sparsity']['mean']:>6.3f}")
    print("  " + "-" * 62)
    print(f"  Cross-method agreement (Spearman): "
          f"{summary['cross_method_agreement']['mean']:.3f} "
          f"± {summary['cross_method_agreement']['std']:.3f}")
    print("=" * 72)

    save_stats(summary, str(PROJECT_ROOT / 'outputs/stats/explainability_faithfulness.json'))
    print("  Saved -> outputs/stats/explainability_faithfulness.json")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_dir = PROJECT_ROOT / 'outputs/figures/explainability'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    x = np.arange(len(methods))
    w = 0.35
    plus = [summary['methods'][m]['fidelity_plus']['mean'] for m in methods]
    plus_e = [summary['methods'][m]['fidelity_plus']['std'] for m in methods]
    minus = [summary['methods'][m]['fidelity_minus']['mean'] for m in methods]
    minus_e = [summary['methods'][m]['fidelity_minus']['std'] for m in methods]
    ax.bar(x - w / 2, plus, w, yerr=plus_e, capsize=4, label='Fidelity+ (higher better)', color='#2a9d8f')
    ax.bar(x + w / 2, minus, w, yerr=minus_e, capsize=4, label='Fidelity- (lower better)', color='#e76f51')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods])
    ax.set_ylabel('Predicted-class probability change')
    ax.set_title(f'Explanation Faithfulness — {args.model} '
                 f'(agreement={summary["cross_method_agreement"]["mean"]:.2f})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out_png = fig_dir / 'faithfulness_summary.png'
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {out_png}")


if __name__ == '__main__':
    main()
