"""
Script: Train ProtoN — Prototype Node Graph Neural Network
=========================================================
Implements the dual-path ProtoN training loop with the hybrid alignment loss.
Saves model checkpoint to outputs/proton/best_model.pt.
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
from src.models.proton import CattleProtoN
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
    if random.random() < 0.3:
        noise_scale = 0.05 * (1 - epoch / max_epochs)
        if batch.x is not None:
            batch.x = batch.x + torch.randn_like(batch.x) * noise_scale
    if random.random() < 0.2:
        if batch.x is not None and batch.x.size(0) > 20:
            mask = torch.rand(batch.x.size(0), 1, device=batch.x.device) > 0.05
            batch.x = batch.x * mask
    if random.random() < 0.15:
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

    # Read from config (with fallbacks to previous values)
    proton_cfg = config.get('proton', {})
    epochs        = proton_cfg.get('epochs', 200)
    batch_size    = proton_cfg.get('batch_size', 128)
    lr            = proton_cfg.get('learning_rate', 4e-4)
    wd            = proton_cfg.get('weight_decay', 5e-5)   # Reduced from 5e-4
    dropout       = proton_cfg.get('dropout', 0.12)         # Reduced from 0.20
    align_weight  = proton_cfg.get('align_weight', 0.2)    # KEY: reduced from 0.5
    temperature   = proton_cfg.get('temperature', 0.07)
    num_layers    = proton_cfg.get('num_layers', 4)         # Increased from 3
    num_heads     = proton_cfg.get('num_heads', 4)
    hidden_dim    = proton_cfg.get('hidden_dim', 128)
    warmup_epochs = proton_cfg.get('warmup_epochs', 20)
    use_amp       = proton_cfg.get('use_amp', True)
    ckpt_dir      = str(PROJECT_ROOT / proton_cfg.get('checkpoint_dir', 'outputs/proton'))
    patience      = proton_cfg.get('patience', 50)
    min_delta     = proton_cfg.get('min_delta', 0.0005)

    ensure_dirs(ckpt_dir, str(PROJECT_ROOT / 'outputs/stats'), str(PROJECT_ROOT / 'outputs/results'))

    print(f"\n{'='*65}")
    print("  ProtoN (Prototype Node GNN) TRAINING  [TUNED for 98%+]")
    print(f"{'='*65}")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | WD: {wd}")
    print(f"  Align Weight: {align_weight} (was 0.5) | Temp: {temperature} | Dropout: {dropout}")
    print(f"  Num Layers: {num_layers} (was 3) | Patience: {patience} | min_delta: {min_delta}")
    print(f"  Checkpoint: {ckpt_dir}")

    # -- Data --
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    loaders = create_data_loaders(graph_dir, config, augment_train=True)
    labels = [d.y.item() for d in torch.load(os.path.join(graph_dir, 'train_graphs.pt'), weights_only=False)]
    num_classes = len(set(labels))

    print(f"  Classes: {num_classes} | Train: {len(loaders['train'].dataset)}")
    steps_per_epoch = len(loaders['train'])

    # -- Model --
    model = CattleProtoN(
        num_classes=num_classes,
        input_dim=256,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,          # 10% warmup
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos',
    )

    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

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
            # Enhanced augmentation: stronger noise + edge drop
            batch = apply_graph_augmentation(batch, epoch, epochs)
            # Additional: 15% DropEdge per literature recommendation
            if batch.edge_index is not None and batch.edge_index.size(1) > 0:
                import random
                if random.random() < 0.4:
                    n_edges = batch.edge_index.size(1)
                    keep = torch.rand(n_edges, device=batch.edge_index.device) > 0.15
                    batch.edge_index = batch.edge_index[:, keep]
                    if batch.edge_attr is not None:
                        batch.edge_attr = batch.edge_attr[keep]
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
                loss = model.compute_loss(
                    out['embedding'], 
                    batch.y, 
                    temperature=temperature, 
                    align_weight=align_weight
                )

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
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val R1: {val_r1:.4f} | LR: {current_lr:.6f} | {epoch_time:.1f}s", flush=True)

        history['train_loss'].append(avg_loss)
        history['val_r1'].append(val_r1)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

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
                    'architecture': 'CattleProtoN',
                    'features': 'DISK',
                    'dropout': dropout,
                    'weight_decay': wd,
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
    print(f"ProtoN Training Complete! Best R1: {best_r1:.4f} @ epoch {best_epoch}")
    print(f"{'='*65}")

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
        'model': 'ProtoN [TUNED]',
        'architecture': 'ProtoN (Prototype Node GNN + Cross-Graph Alignment)',
        'features': 'Kornia-DISK',
        'best_val_r1': best_r1,
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'history': history,
        'hyperparameters': {
            'lr': lr, 'weight_decay': wd, 'dropout': dropout,
            'align_weight': align_weight, 'temperature': temperature,
            'batch_size': batch_size, 'epochs': epochs,
            'num_layers': num_layers, 'num_heads': num_heads,
        },
    }, str(PROJECT_ROOT / 'outputs/stats/proton_results.json'))

    print(f"\nProtoN results saved to outputs/stats/proton_results.json")

if __name__ == '__main__':
    main()
