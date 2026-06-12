"""
Evaluate GNN+ from saved checkpoint (recover results after crash)
=================================================================
The GNN+ training completed successfully (best_model.pt saved)
but crashed during the final evaluation step.
This script loads the checkpoint and runs the full test evaluation.
"""

import os
import sys
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs
from src.models.gnn_model import CattleGNN
from src.models.arcface import ArcFaceLoss
from src.training.dataset import create_data_loaders
from src.evaluation.metrics import BiometricMetrics


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = str(PROJECT_ROOT / 'outputs/gnn_plus/best_model.pt')
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"\n{'='*55}")
    print("  GNN+ EVALUATION (from checkpoint)")
    print(f"{'='*55}")
    print(f"  Best epoch : {ckpt['epoch']}")
    print(f"  Best val R1: {ckpt['val_r1']:.4f}")

    # ── Load test data ─────────────────────────────────────────────────────
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    loaders = create_data_loaders(graph_dir, config)

    train_graphs = torch.load(os.path.join(graph_dir, 'train_graphs.pt'), weights_only=False)
    labels = [d.y.item() for d in train_graphs]
    num_classes = len(set(labels))
    print(f"  Classes: {num_classes}")

    # ── Load model ─────────────────────────────────────────────────────────
    config_plus = dict(config)
    config_plus['model'] = dict(config['model'])
    config_plus['model']['edge_conv'] = dict(config['model']['edge_conv'])
    config_plus['model']['edge_conv']['k_dynamic'] = config.get('gnn_plus', {}).get('edge_conv_k_dynamic', 12)

    model = CattleGNN(config=config_plus)
    model.set_num_classes(num_classes)
    model = model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Model loaded OK")

    # ── Evaluate on test set ───────────────────────────────────────────────
    metrics = BiometricMetrics()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loaders['test']:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            all_emb.append(out['embedding'].cpu())
            all_lbl.append(batch.y.cpu())

    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    print(f"  Test embeddings: {emb.shape}")

    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    # Extract history from checkpoint
    history = ckpt.get('history', {})
    best_val_r1 = ckpt['val_r1']

    ensure_dirs(str(PROJECT_ROOT / 'outputs/stats'))
    save_stats({
        'model': 'GNN+',
        'architecture': 'CattleGNN + ArcFace + EdgeConv(k=12) + GraphAug',
        'best_epoch': ckpt['epoch'],
        'best_val_r1': best_val_r1,
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies'].get('rank_5', 0),
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'history': history,
    }, str(PROJECT_ROOT / 'outputs/stats/gnn_plus_results.json'))

    print(f"\n  GNN+ Test Rank-1: {results['identification']['rank_accuracies']['rank_1']*100:.1f}%")
    print(f"  GNN+ Best Val R1: {best_val_r1*100:.1f}%")
    print(f"\n  Results saved to outputs/stats/gnn_plus_results.json")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
