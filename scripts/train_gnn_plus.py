"""
Script: Train GNN+ (Improved CattleGNN with ArcFace)
======================================================
Improved version of the base GNN model for the three-way comparison:
  - ArcFace loss instead of triplet loss
  - Deeper EdgeConv (k_dynamic=12)
  - Graph augmentation (KeypointDropout + FeatureJitter)
  - 150 epochs with cosine annealing

Output: outputs/gnn_plus/best_model.pt
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.models.gnn_model import CattleGNN
from src.models.arcface import ArcFaceLoss
from src.training.dataset import create_data_loaders
from src.training.augmentation import GraphAugmentation
from src.evaluation.metrics import BiometricMetrics


def validate_rank1(model, val_loader, device):
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            all_emb.append(out['embedding'].cpu())
            all_lbl.append(batch.y.cpu())
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    sim = torch.mm(emb, emb.t())
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)
    return (lbl[nn_idx] == lbl).float().mean().item()


def main():
    config = load_config()
    set_seed(config['project']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_cfg = config.get('gnn_plus', {})
    arc_cfg = config.get('arcface', {})

    epochs        = gnn_cfg.get('epochs', 150)
    batch_size    = gnn_cfg.get('batch_size', 128)
    lr            = gnn_cfg.get('learning_rate', 4e-4)
    wd            = gnn_cfg.get('weight_decay', 1e-4)
    use_amp       = gnn_cfg.get('use_amp', True)
    ckpt_dir      = str(PROJECT_ROOT / gnn_cfg.get('checkpoint_dir', 'outputs/gnn_plus'))
    patience      = gnn_cfg.get('early_stopping', {}).get('patience', 30)
    min_delta     = gnn_cfg.get('early_stopping', {}).get('min_delta', 0.001)

    ensure_dirs(ckpt_dir, str(PROJECT_ROOT / 'outputs/stats'),
                str(PROJECT_ROOT / 'outputs/results'))

    print(f"\n{'='*65}")
    print("  GNN+ TRAINING  (ArcFace + deeper EdgeConv + graph augmentation)")
    print(f"{'='*65}")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Checkpoint: {ckpt_dir}")

    # ── Data ────────────────────────────────────────────────────────────────
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    # Use graph augmentation for GNN+
    loaders = create_data_loaders(graph_dir, config, augment_train=True)
    labels = [d.y.item() for d in torch.load(
        os.path.join(graph_dir, 'train_graphs.pt'), weights_only=False)]
    num_classes = len(set(labels))
    print(f"  Classes: {num_classes} | Train: {len(loaders['train'].dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    # Slightly deeper k_dynamic=12 for GNN+
    config_plus = dict(config)
    config_plus['model'] = dict(config['model'])
    config_plus['model']['edge_conv'] = dict(config['model']['edge_conv'])
    config_plus['model']['edge_conv']['k_dynamic'] = gnn_cfg.get('edge_conv_k_dynamic', 12)

    model = CattleGNN(config=config_plus).to(device)
    model.set_num_classes(num_classes)
    model = model.to(device)

    # ArcFace loss
    arcface = ArcFaceLoss(
        embedding_dim=config['model']['embedding_dim'],
        num_classes=num_classes,
        margin=arc_cfg.get('margin', 0.5),
        scale=arc_cfg.get('scale', 64.0),
        triplet_weight=arc_cfg.get('triplet_weight', 0.1),
        triplet_margin=arc_cfg.get('triplet_margin', 0.3),
    ).to(device)

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    params = list(model.parameters()) + list(arcface.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    # AMP
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in arcface.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  AMP: {use_amp} ({amp_dtype}) | Scaler: {use_scaler}")

    # ── Training Loop ────────────────────────────────────────────────────────
    best_r1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_r1': [], 'lr': [], 'epoch_time': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        arcface.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loaders['train']:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
                loss, stats = arcface(out['embedding'], batch.y)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        val_r1 = validate_rank1(model, loaders['val'], device)
        epoch_time = time.time() - t0

        vram = f" | VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val R1: {val_r1:.4f} "
              f"| LR: {optimizer.param_groups[0]['lr']:.6f} | {epoch_time:.1f}s{vram}", flush=True)

        history['train_loss'].append(avg_loss)
        history['val_r1'].append(val_r1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_time)

        # Save best
        if val_r1 > best_r1 + min_delta:
            best_r1 = val_r1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'val_r1': val_r1,
                'model_state_dict': model.state_dict(),
                'arcface_state_dict': arcface.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  >> New best! R1: {best_r1:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early stopping] Epoch {epoch}, patience={patience}")
                break

    # ── Final Evaluation ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"GNN+ Training Complete! Best R1: {best_r1:.4f} @ epoch {best_epoch}")
    print(f"{'='*65}")

    # Full test evaluation
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

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
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    save_stats({
        'model': 'GNN+',
        'best_val_r1': best_r1,
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
        'history': history,
    }, str(PROJECT_ROOT / 'outputs/stats/gnn_plus_results.json'))

    print(f"\n✅ GNN+ results saved to outputs/stats/gnn_plus_results.json")


if __name__ == '__main__':
    main()

