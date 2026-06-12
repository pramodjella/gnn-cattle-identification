"""
Script: Train GNN v4 — High Capacity Enhanced Model  [TUNED for 98%+]
======================================================================
Key changes over previous version:
  1. dropout=0.15 (from 0.30) — was catastrophically over-regularized
     (20.56M params barely beat 5M v3 at 91.60% vs 91.39% due to this)
  2. weight_decay=1e-5 (from 1e-3) — was far too aggressive for GNN
  3. Proper ArcFace label_smoothing=0.05 (not fake embedding penalty)
  4. batch_size=32 KEPT — 1024-d head_out (128*8 heads) needs ~6GB at bs=32
  5. min_delta=0.0005 (from 0.001) — finer improvement detection
  6. Config-driven hyperparameters
  7. VRAM safety check at startup

Output: outputs/gnn_v4/best_model.pt
        outputs/stats/gnn_v4_enhanced_results.json
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
from torch.optim.lr_scheduler import OneCycleLR

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.models.gnn_v3 import CattleGNNv3
from src.training.dataset import create_data_loaders
from src.evaluation.metrics import BiometricMetrics


def validate_rank1(model, val_loader, device):
    """Compute Rank-1 accuracy on validation set."""
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


def apply_graph_augmentation(batch, epoch, max_epochs):
    """Apply stochastic graph augmentation during training."""
    import random
    
    if random.random() < 0.3:  # 30% chance to augment
        # Feature noise injection (scaled down over training)
        noise_scale = 0.05 * (1 - epoch / max_epochs)  # Anneal noise
        if batch.x is not None:
            noise = torch.randn_like(batch.x) * noise_scale
            batch.x = batch.x + noise
    
    if random.random() < 0.2:  # 20% chance of node dropout
        if batch.x is not None and batch.x.size(0) > 20:
            # Randomly zero out 5-10% of node features
            mask = torch.rand(batch.x.size(0), 1, device=batch.x.device) > 0.05
            batch.x = batch.x * mask
    
    if random.random() < 0.15:  # 15% chance of edge dropout
        if batch.edge_index is not None and batch.edge_index.size(1) > 50:
            num_edges = batch.edge_index.size(1)
            keep_mask = torch.rand(num_edges, device=batch.edge_index.device) > 0.05
            batch.edge_index = batch.edge_index[:, keep_mask]
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[keep_mask]
    
    return batch


def main():
    config = load_config()
    set_seed(config['project']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # VRAM check: GNN v4 needs ~6GB at batch=32 (1024-d head_out * 8 heads)
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU VRAM total: {total_vram_gb:.1f} GB")
        if total_vram_gb < 7.5:
            print(f"  [WARNING] <8GB VRAM detected. Reducing batch to 16.")

    # Read from config — falls back to tuned defaults
    v4_cfg = config.get('gnn_v4', {})
    epochs       = v4_cfg.get('epochs', 300)
    # VRAM-safe batch: 1024-d head (128 hidden * 8 heads), 4 layers, ~6GB at bs=32
    # Do NOT increase beyond 32 on 8GB GPU
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1024**3
               if torch.cuda.is_available() else 0)
    batch_size   = v4_cfg.get('batch_size', 32 if vram_gb >= 7.5 else 16)
    lr           = v4_cfg.get('learning_rate', 4e-4)
    wd           = v4_cfg.get('weight_decay', 1e-5)   # KEY FIX: 1e-3 -> 1e-5
    dropout      = v4_cfg.get('dropout', 0.15)         # KEY FIX: 0.30 -> 0.15
    use_amp      = v4_cfg.get('use_amp', True)
    ckpt_dir     = str(PROJECT_ROOT / v4_cfg.get('checkpoint_dir', 'outputs/gnn_v4'))
    patience     = v4_cfg.get('patience', 80)
    min_delta    = v4_cfg.get('min_delta', 0.0005)    # 0.001 -> 0.0005
    label_smooth = 0.05                               # Proper ArcFace label smoothing
    # Architecture (same as before — keep high capacity)
    hidden_dim   = v4_cfg.get('hidden_dim', 128)
    num_heads    = v4_cfg.get('num_heads', 8)          # 8 heads * 128 = 1024-d
    num_layers   = v4_cfg.get('num_layers', 4)

    ensure_dirs(ckpt_dir, str(PROJECT_ROOT / 'outputs/stats'),
                str(PROJECT_ROOT / 'outputs/results'))

    print(f"\n{'='*65}")
    print("  GNN v4 TRAINING (High Capacity)  [TUNED for 98%+]")
    print(f"{'='*65}")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} (VRAM-safe) | LR: {lr} | WD: {wd}")
    print(f"  Dropout: {dropout} (was 0.30) | Label Smoothing: {label_smooth}")
    print(f"  Arch: hidden={hidden_dim}, heads={num_heads}, layers={num_layers} | head_out={hidden_dim*num_heads}-d")
    print(f"  Scheduler: OneCycleLR | Patience: {patience} | min_delta: {min_delta}")
    print(f"  Checkpoint: {ckpt_dir}")

    # -- Data --
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    config['training']['batch_size'] = batch_size
    loaders = create_data_loaders(graph_dir, config, augment_train=True)
    labels = [d.y.item() for d in torch.load(
        os.path.join(graph_dir, 'train_graphs.pt'), weights_only=False)]
    num_classes = len(set(labels))
    
    print(f"  Classes: {num_classes} | Train: {len(loaders['train'].dataset)}")
    steps_per_epoch = len(loaders['train'])

    # -- Model (high capacity — same architecture, fixed regularization) --
    model = CattleGNNv3(
        input_dim=256,
        hidden_dim=hidden_dim,   # 128 * 8 heads = 1024-d head_out
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.set_num_classes(num_classes)
    # Override ArcFace with proper label smoothing
    from src.models.arcface import ArcFaceLoss
    model.arcface = ArcFaceLoss(
        embedding_dim=model.embedding_dim,
        num_classes=num_classes,
        margin=0.35,
        scale=48.0,
        triplet_weight=0.15,
        triplet_margin=0.3,
        label_smoothing=label_smooth,
    ).to(device)
    model_summary = model.summary()

    # -- Optimizer --
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,              # 10% warmup (30 epochs)
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos',
    )

    # AMP
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  AMP: {use_amp} ({amp_dtype}) | Scaler: {use_scaler}")

    # -- Training Loop --
    best_r1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_r1': [], 'lr': [], 'epoch_time': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loaders['train']:
            batch = batch.to(device, non_blocking=True)
            
            # Apply graph augmentation
            batch = apply_graph_augmentation(batch, epoch, epochs)
            
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
                loss = out.get('loss', None)

                if loss is None:
                    loss, _ = model.arcface(out['embedding'], batch.y)
                # label_smoothing is now handled inside ArcFaceLoss properly

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        val_r1 = validate_rank1(model, loaders['val'], device)
        epoch_time = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        vram = f" | VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val R1: {val_r1:.4f} "
              f"| LR: {current_lr:.6f} | {epoch_time:.1f}s{vram}", flush=True)

        history['train_loss'].append(avg_loss)
        history['val_r1'].append(val_r1)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        # Save best
        if val_r1 > best_r1 + min_delta:
            best_r1 = val_r1
            best_epoch = epoch
            patience_counter = 0
            save_dict = {
                'epoch': epoch, 'val_r1': val_r1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': {
                    'architecture': 'CattleGNNv4',
                    'features': 'DISK',
                    'dropout': dropout,
                    'weight_decay': wd,
                    'label_smoothing': label_smooth,
                },
            }
            torch.save(save_dict, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  >> New best! R1: {best_r1:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 60:
                print(f"\n[Early stopping] Epoch {epoch}, patience={patience}")
                break

    # -- Final Evaluation --
    print(f"\n{'='*65}")
    print(f"GNN v4 Training Complete! Best R1: {best_r1:.4f} @ epoch {best_epoch}")
    print(f"{'='*65}")

    # Load best model for full evaluation
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), 
                       map_location=device, weights_only=False)
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
        'model': 'GNN_v4_tuned',
        'architecture': 'CattleGNNv4 (GATv2 + VirtualNode + GraphNorm - High Capacity) [TUNED]',
        'features': 'Kornia-DISK',
        'best_val_r1': best_r1,
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'history': history,
        'model_summary': model_summary,
        'hyperparameters': {
            'lr': lr, 'weight_decay': wd, 'dropout': dropout,
            'label_smoothing': label_smooth, 'scheduler': 'OneCycleLR',
            'batch_size': batch_size, 'epochs': epochs,
            'hidden_dim': hidden_dim, 'num_heads': num_heads, 'num_layers': num_layers,
            'head_out_dim': hidden_dim * num_heads,
        },
    }, str(PROJECT_ROOT / 'outputs/stats/gnn_v4_enhanced_results.json'))

    print(f"\nGNN v4 enhanced results saved to outputs/stats/gnn_v4_enhanced_results.json")


if __name__ == '__main__':
    main()
