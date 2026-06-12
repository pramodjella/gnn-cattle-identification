"""
Script: Evaluate Pre-Trained Hybrid CNN-GNN Model
==================================================
Loads the existing best_model.pt from outputs/hybrid/ and runs
final test set evaluation to produce hybrid_results.json.
Use this when the Hybrid model is already trained but its results
JSON is missing (e.g. training was interrupted after Phase 2).

Output: outputs/stats/hybrid_results.json
"""

import os
import sys
import json
import torch
from pathlib import Path
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.models.hybrid_model import HybridCNNGNN
from src.training.image_dataset import create_hybrid_loaders
from src.evaluation.metrics import BiometricMetrics


def main():
    config = load_config()
    set_seed(config['project']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hybrid_cfg = config.get('hybrid', {})
    use_amp    = hybrid_cfg.get('use_amp', True)
    ckpt_dir   = str(PROJECT_ROOT / hybrid_cfg.get('checkpoint_dir', 'outputs/hybrid'))

    ensure_dirs(str(PROJECT_ROOT / 'outputs/stats'))

    print(f"\n{'='*65}")
    print("  HYBRID MODEL EVALUATION  (Pre-trained checkpoint)")
    print(f"{'='*65}")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Checkpoint: {ckpt_dir}/best_model.pt")

    # ── Data ────────────────────────────────────────────────────────────────
    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    loaders = create_hybrid_loaders(preprocessed_dir, graph_dir, config)

    with open(os.path.join(preprocessed_dir, 'train_split.json')) as f:
        train_data = json.load(f)
    num_classes = len(set(
        item.get('animal_id', item.get('label', str(i)))
        for i, item in enumerate(train_data)
    ))
    print(f"  Classes: {num_classes} | Test samples: {len(loaders['test'].dataset)}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] No checkpoint found at {ckpt_path}")
        print("  → Please run: python scripts/train_hybrid.py first")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_classes = ckpt.get('num_classes', num_classes)

    # ── Build model ──────────────────────────────────────────────────────────
    model = HybridCNNGNN(
        num_classes=checkpoint_classes,
        config=config,
        pretrained=False,  # weights loaded from checkpoint
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"  Loaded from epoch {ckpt.get('epoch', '?')} | "
          f"Val R1: {ckpt.get('val_r1', 0):.4f}")

    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # ── Evaluate on test split ───────────────────────────────────────────────
    metrics = BiometricMetrics()
    all_emb, all_lbl = [], []

    with torch.no_grad():
        for images, graphs, labels in loaders['test']:
            images = images.to(device, non_blocking=True)
            graphs = graphs.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(images, graphs)
            all_emb.append(out['embedding'].float().cpu())
            all_lbl.append(labels)

    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)

    print(f"\n  Embeddings: {emb.shape} | Labels: {lbl.shape}")
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    # ── Also evaluate on val ─────────────────────────────────────────────────
    all_emb_val, all_lbl_val = [], []
    with torch.no_grad():
        for images, graphs, labels in loaders['val']:
            images = images.to(device, non_blocking=True)
            graphs = graphs.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(images, graphs)
            all_emb_val.append(out['embedding'].float().cpu())
            all_lbl_val.append(labels)
    emb_val = torch.cat(all_emb_val)
    lbl_val = torch.cat(all_lbl_val)
    sim_val = torch.mm(emb_val, emb_val.t())
    sim_val.fill_diagonal_(-1e9)
    val_r1 = (lbl_val[sim_val.argmax(dim=1)] == lbl_val).float().mean().item()
    print(f"\n  Val Rank-1: {val_r1:.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    history = ckpt.get('history', {})
    save_stats({
        'model': 'Hybrid CNN-GNN (EfficientNet-B3 + EdgeConv + TRM + ArcFace)',
        'architecture': 'Cached feature map training + end-to-end fine-tuning',
        'best_epoch': ckpt.get('epoch', -1),
        'best_val_r1': ckpt.get('val_r1', val_r1),
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies'].get('rank_5', 0),
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
        'history': history,
    }, str(PROJECT_ROOT / 'outputs/stats/hybrid_results.json'))

    print(f"\n[DONE] Hybrid results saved to outputs/stats/hybrid_results.json")
    print(f"   Test Rank-1: {results['identification']['rank_accuracies']['rank_1']*100:.1f}%")
    print(f"   Test EER:    {results['verification']['eer']*100:.2f}%")
    print(f"   ROC AUC:     {results['verification']['roc_auc']:.4f}")


if __name__ == '__main__':
    main()
