"""
Script: Train GNN v3 (State-of-the-Art Pure GNN)
==================================================
Trains CattleGNNv3 with:
  - GATv2 (dynamic attention) + Virtual Nodes + GraphNorm
  - Edge features as first-class citizens
  - Sub-center ArcFace (k=3) for intra-class variation
  - Enhanced graph augmentation (SubgraphCrop + FeatureMixup)
  - Cosine annealing with warm restarts
  - Stochastic Weight Averaging (SWA) in final phase

Output: outputs/gnn_v3/best_model.pt
"""

import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.models.gnn_v3 import CattleGNNv3
from src.models.arcface import ArcFaceLoss
from src.training.dataset import create_data_loaders
from src.training.augmentation import EnhancedGraphAugmentation
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


def main():
    config = load_config()
    set_seed(config['project']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GNN v3 config (with fallbacks)
    v3_cfg = config.get('gnn_v3', {})
    arc_cfg = config.get('arcface', {})
    
    epochs        = v3_cfg.get('epochs', 200)
    batch_size    = v3_cfg.get('batch_size', 128)
    lr            = v3_cfg.get('learning_rate', 3e-4)
    wd            = v3_cfg.get('weight_decay', 1e-4)
    use_amp       = v3_cfg.get('use_amp', True)
    ckpt_dir      = str(PROJECT_ROOT / v3_cfg.get('checkpoint_dir', 'outputs/gnn_v3'))
    patience      = v3_cfg.get('patience', 40)
    min_delta     = v3_cfg.get('min_delta', 0.001)
    warmup_epochs = v3_cfg.get('warmup_epochs', 10)
    swa_start     = v3_cfg.get('swa_start_epoch', 150)  # Start SWA at this epoch
    swa_lr        = v3_cfg.get('swa_lr', 1e-5)

    ensure_dirs(ckpt_dir, str(PROJECT_ROOT / 'outputs/stats'),
                str(PROJECT_ROOT / 'outputs/results'))

    print(f"\n{'='*65}")
    print("  GNN v3 TRAINING  (GATv2 + VirtualNode + GraphNorm + Edge Features)")
    print(f"{'='*65}")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  SWA: starts at epoch {swa_start} | SWA LR: {swa_lr}")
    print(f"  Checkpoint: {ckpt_dir}")

    # ── Data ────────────────────────────────────────────────────────────────
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    # Enhanced augmentation for v3
    loaders = create_data_loaders(graph_dir, config, augment_train=True)
    labels = [d.y.item() for d in torch.load(
        os.path.join(graph_dir, 'train_graphs.pt'), weights_only=False)]
    num_classes = len(set(labels))
    print(f"  Classes: {num_classes} | Train: {len(loaders['train'].dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = CattleGNNv3(config=config).to(device)
    model.set_num_classes(num_classes)
    model_summary = model.summary()
    
    # ArcFace is now inside the model via set_num_classes
    arcface = model.arcface

    # ── Optimizer / Scheduler ────────────────────────────────────────────────
    # Separate learning rates: GATv2 layers get slightly lower LR
    gat_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'gat_layers' in name:
            gat_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': gat_params, 'lr': lr * 0.5},        # GATv2 layers: lower LR
        {'params': other_params, 'lr': lr},             # Everything else
    ], weight_decay=wd)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=1e-6)
    
    # SWA (Stochastic Weight Averaging)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=5)

    # AMP
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  AMP: {use_amp} ({amp_dtype}) | Scaler: {use_scaler}")

    # ── Learning Rate Warmup ─────────────────────────────────────────────────
    def get_warmup_factor(epoch: int) -> float:
        """Linear warmup for the first `warmup_epochs` epochs."""
        if epoch <= warmup_epochs:
            return epoch / warmup_epochs
        return 1.0

    # ── Training Loop ────────────────────────────────────────────────────────
    best_r1 = 0.0
    best_epoch = 0
    patience_counter = 0
    swa_started = False
    history = {'train_loss': [], 'val_r1': [], 'lr': [], 'epoch_time': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Apply warmup factor
        warmup_factor = get_warmup_factor(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = pg.get('initial_lr', pg['lr']) * warmup_factor if epoch <= warmup_epochs else pg['lr']

        for batch in loaders['train']:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
                loss = out.get('loss', None)
                
                if loss is None:
                    # Fallback: compute ArcFace loss separately
                    loss, _ = arcface(out['embedding'], batch.y)

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

            total_loss += loss.item()
            num_batches += 1

        # Scheduler step
        if epoch >= swa_start:
            if not swa_started:
                print(f"\n  >> SWA started at epoch {epoch}")
                swa_started = True
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Validate with SWA model after SWA starts, otherwise regular model
        if swa_started:
            # Update SWA BN stats
            torch.optim.swa_utils.update_bn(loaders['train'], swa_model, device=device)
            val_r1 = validate_rank1(swa_model, loaders['val'], device)
        else:
            val_r1 = validate_rank1(model, loaders['val'], device)
        
        epoch_time = time.time() - t0

        vram = f" | VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
        swa_tag = " [SWA]" if swa_started else ""
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val R1: {val_r1:.4f} "
              f"| LR: {optimizer.param_groups[0]['lr']:.6f} | {epoch_time:.1f}s{vram}{swa_tag}", flush=True)

        history['train_loss'].append(avg_loss)
        history['val_r1'].append(val_r1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_time)

        # Save best
        if val_r1 > best_r1 + min_delta:
            best_r1 = val_r1
            best_epoch = epoch
            patience_counter = 0
            save_dict = {
                'epoch': epoch, 'val_r1': val_r1,
                'model_state_dict': (swa_model.module.state_dict() if swa_started 
                                     else model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': {
                    'architecture': 'CattleGNNv3',
                    'features': 'DISK',
                    'swa': swa_started,
                },
            }
            torch.save(save_dict, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  >> New best! R1: {best_r1:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > warmup_epochs + 20:
                print(f"\n[Early stopping] Epoch {epoch}, patience={patience}")
                break

    # ── Final Evaluation ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"GNN v3 Training Complete! Best R1: {best_r1:.4f} @ epoch {best_epoch}")
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
        'model': 'GNN_v3',
        'architecture': 'CattleGNNv3 (GATv2 + VirtualNode + GraphNorm)',
        'features': 'Kornia-DISK',
        'best_val_r1': best_r1,
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
        'history': history,
        'model_summary': model_summary,
    }, str(PROJECT_ROOT / 'outputs/stats/gnn_v3_results.json'))

    print(f"\n✅ GNN v3 results saved to outputs/stats/gnn_v3_results.json")


if __name__ == '__main__':
    main()
