"""
Script: Train CNN Baseline (EfficientNet-B4 + ArcFace)  [TUNED for 98%+]
=========================================================================
Upgraded training script with:
  - EfficientNet-B4 backbone (from B3)
  - Embedding dim 512 (from 256)
  - Mixup augmentation (alpha=0.2) — label-preserving mix
  - Stochastic Weight Averaging (SWA) from epoch 100
  - Test-Time Augmentation (TTA) at evaluation
  - ArcFace scale=42, margin=0.40 (literature-optimal for 260 classes)
  - Label smoothing=0.05
  - Patience=30, epochs=150

Output: outputs/cnn/best_model.pt
"""

import os
import sys
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, update_bn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.models.cnn_model import CNNMuzzleModel
from src.training.image_dataset import create_image_loaders

from src.evaluation.metrics import BiometricMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Mixup Utility
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2, device='cuda'):
    """
    Mixup augmentation: interpolate pairs of samples and labels.
    For ArcFace we use the PRIMARY label (lam > 0.5 → use original label).
    This is 'hard' mixup — we keep the dominant label for ArcFace.
    """
    if alpha > 0:
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)  # always use dominant label

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y  # keep original labels (dominant label approach)


# ─────────────────────────────────────────────────────────────────────────────
# Validation with TTA
# ─────────────────────────────────────────────────────────────────────────────

def validate_rank1_tta(model, val_loader, device, image_size=256, use_tta=True):
    """
    Validate Rank-1 accuracy, optionally with Test-Time Augmentation.
    TTA averages embeddings from 2 views: original + horizontal flip.
    Handles AveragedModel wrapper transparently via .module unwrapping.
    """
    # Unwrap AveragedModel — get_embedding is a custom method not forwarded
    base_model = getattr(model, 'module', model)
    base_model.eval()

    if not use_tta:
        # Standard single-pass evaluation
        all_emb, all_lbl = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                emb = base_model.get_embedding(images)
                all_emb.append(emb.cpu())
                all_lbl.append(labels)
        emb = torch.cat(all_emb)
        lbl = torch.cat(all_lbl)
    else:
        # TTA: 2-view (original + horizontal flip) → average embeddings
        all_emb, all_lbl = [], []

        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            images_flipped = torch.flip(images, dims=[-1])  # horizontal flip

            with torch.no_grad():
                emb1 = base_model.get_embedding(images)
                emb2 = base_model.get_embedding(images_flipped)
                emb = F.normalize(emb1 + emb2, p=2, dim=-1)

            all_emb.append(emb.cpu())
            all_lbl.append(labels)

        emb = torch.cat(all_emb)
        lbl = torch.cat(all_lbl)

    sim = torch.mm(emb, emb.t())
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)
    return (lbl[nn_idx] == lbl).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    set_seed(config['project']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cnn_cfg = config.get('cnn', {})
    # CLI overrides for controlled ablations (leave config untouched):
    #   --no-mixup           disable mixup
    #   --ckpt-dir <path>    redirect checkpoints (avoids clobbering the paper's model)
    import sys as _sys
    if '--no-mixup' in _sys.argv:
        cnn_cfg = dict(cnn_cfg); cnn_cfg['use_mixup'] = False
    if '--ckpt-dir' in _sys.argv:
        cnn_cfg = dict(cnn_cfg); cnn_cfg['checkpoint_dir'] = _sys.argv[_sys.argv.index('--ckpt-dir') + 1]

    epochs          = cnn_cfg.get('epochs', 104)  # Cycle 4 restart (ep105) causes NaN with ArcFace s=128
    batch_size      = cnn_cfg.get('batch_size', 16)
    backbone_lr     = cnn_cfg.get('backbone_lr', 3e-5)
    head_lr         = cnn_cfg.get('head_lr', 1e-3)
    wd              = cnn_cfg.get('weight_decay', 5e-5)
    dropout         = cnn_cfg.get('dropout', 0.35)
    embedding_dim   = cnn_cfg.get('embedding_dim', 512)
    backbone        = cnn_cfg.get('backbone', 'efficientnet_b4')
    use_amp         = cnn_cfg.get('use_amp', True)
    use_mixup       = cnn_cfg.get('use_mixup', True)
    mixup_alpha     = cnn_cfg.get('mixup_alpha', 0.2)
    use_swa         = False   # Disabled: AveragedModel diverges with ArcFace s=128 (NaN at ep113)
    swa_start       = 9999    # Never reached
    swa_lr          = cnn_cfg.get('swa_lr', 5e-6)
    use_tta         = cnn_cfg.get('use_tta', True)
    arcface_scale   = cnn_cfg.get('arcface_scale', 128.0)
    arcface_margin  = cnn_cfg.get('arcface_margin', 0.35)
    label_smoothing = cnn_cfg.get('label_smoothing', 0.05)
    ckpt_dir        = str(PROJECT_ROOT / cnn_cfg.get('checkpoint_dir', 'outputs/cnn'))
    patience        = cnn_cfg.get('early_stopping', {}).get('patience', 30)
    min_delta       = cnn_cfg.get('early_stopping', {}).get('min_delta', 0.0005)
    image_size      = config.get('preprocessing', {}).get('image_size', 256)

    ensure_dirs(ckpt_dir, str(PROJECT_ROOT / 'outputs/stats'))

    print(f"\n{'='*70}")
    print(f"  CNN TUNED TRAINING  ({backbone} + ArcFace)  [Target: 98%+]")
    print(f"{'='*70}")
    print(f"  Device : {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Backbone: {backbone} | Emb: {embedding_dim}-d | ArcFace: s={arcface_scale}, m={arcface_margin}")
    print(f"  Epochs : {epochs} | Batch: {batch_size} | LR backbone: {backbone_lr} | head: {head_lr}")
    print(f"  Mixup  : {use_mixup} (α={mixup_alpha}) | SWA: {use_swa} (start ep {swa_start}) | TTA: {use_tta}")

    # ── Data ────────────────────────────────────────────────────────────────
    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    loaders = create_image_loaders(preprocessed_dir, config)

    with open(os.path.join(preprocessed_dir, 'train_split.json')) as f:
        train_data = json.load(f)
    num_classes = len(set(
        item.get('animal_id', item.get('label', str(i)))
        for i, item in enumerate(train_data)
    ))
    print(f"  Classes: {num_classes} | Train samples: {len(loaders['train'].dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = CNNMuzzleModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout=dropout,
        pretrained=True,
        backbone=backbone,
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
        label_smoothing=label_smoothing,
    ).to(device)
    model.summary()

    # Differential LR parameter groups
    param_groups = model.get_parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-7)

    # SWA model — collects weight averages from ep swa_start onward
    swa_model = AveragedModel(model) if use_swa else None
    
    # AMP
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)

    # ── Training Loop ────────────────────────────────────────────────────────
    best_r1 = 0.0
    best_epoch = 0
    patience_counter = 0
    in_swa_phase = False
    history = {'train_loss': [], 'val_r1': [], 'lr': [], 'epoch_time': [], 'swa_active': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Switch to SWA phase
        if use_swa and epoch == swa_start and not in_swa_phase:
            print(f"\n  [SWA] Switching to SWA phase at epoch {epoch}")
            in_swa_phase = True

        for images, labels in loaders['train']:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixup augmentation (hard mixup — keep dominant label for ArcFace)
            if use_mixup and not in_swa_phase:
                images, labels = mixup_data(images, labels, alpha=mixup_alpha, device=device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                result = model(images, labels)
                loss = result['loss']

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

        # LR scheduling + SWA weight collection
        if in_swa_phase and use_swa:
            # Collect averaged weights from base model
            swa_model.update_parameters(model)
            # Manual constant SWA LR (avoids SWALR NaN bug with ArcFace)
            for pg in optimizer.param_groups:
                pg['lr'] = swa_lr
            # Evaluate with averaged weights + pre-SWA BN stats (stable from ep99)
            # NOTE: Do NOT call swa_model.train() or run any forward pass on swa_model
            # as it shares BN buffers with base model and would corrupt training stats
            val_r1 = validate_rank1_tta(swa_model, loaders['val'], device, image_size, use_tta=False)
        else:
            scheduler.step()
            val_r1 = validate_rank1_tta(model, loaders['val'], device, image_size, use_tta=use_tta)

        avg_loss = total_loss / max(num_batches, 1)

        # NaN early stop — cosine cycle restart can cause gradient explosion with ArcFace s=128
        if math.isnan(avg_loss):
            print(f"\n[WARN] NaN loss detected at epoch {epoch}. Stopping early."
                  f" Best checkpoint preserved at Val R1={best_r1:.4f}.")
            break
        epoch_time = time.time() - t0

        vram = f" | VRAM: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
        swa_tag = " [SWA]" if in_swa_phase else ""
        print(f"Epoch {epoch:3d}/{epochs}{swa_tag} | Loss: {avg_loss:.4f} | Val R1: {val_r1:.4f} "
              f"| LR: {optimizer.param_groups[0]['lr']:.2e} | {epoch_time:.1f}s{vram}", flush=True)

        history['train_loss'].append(avg_loss)
        history['val_r1'].append(val_r1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_time)
        history['swa_active'].append(in_swa_phase)

        if val_r1 > best_r1 + min_delta:
            best_r1 = val_r1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'val_r1': val_r1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'num_classes': num_classes,
                'config': {
                    'backbone': backbone, 'embedding_dim': embedding_dim,
                    'arcface_scale': arcface_scale, 'arcface_margin': arcface_margin,
                },
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  >> New best! R1: {best_r1:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early stopping] Epoch {epoch}")
                break

    # ── SWA BN Update ─────────────────────────────────────────────────────────
    if use_swa and in_swa_phase:
        print("\n  [SWA] Updating BatchNorm statistics...")
        train_loader_bn = loaders['train']
        update_bn(train_loader_bn, swa_model, device=device)
        swa_r1 = validate_rank1_tta(swa_model, loaders['val'], device, image_size, use_tta=use_tta)
        print(f"  [SWA] SWA model Val R1: {swa_r1:.4f} (vs best single: {best_r1:.4f})")
        if swa_r1 > best_r1:
            print(f"  [SWA] SWA is better! Saving SWA model.")
            best_r1 = swa_r1
            torch.save({
                'epoch': epochs, 'val_r1': swa_r1, 'swa': True,
                'model_state_dict': swa_model.module.state_dict(),
                'history': history, 'num_classes': num_classes,
                'config': {
                    'backbone': backbone, 'embedding_dim': embedding_dim,
                    'arcface_scale': arcface_scale, 'arcface_margin': arcface_margin,
                },
            }, os.path.join(ckpt_dir, 'best_model.pt'))

    # ── Final Evaluation ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CNN Training Complete! Best R1: {best_r1:.4f} @ epoch {best_epoch}")
    print(f"{'='*70}")

    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    metrics = BiometricMetrics()
    all_emb, all_lbl = [], []

    # Full TTA evaluation on test set
    with torch.no_grad():
        for images, labels in loaders['test']:
            images = images.to(device, non_blocking=True)
            images_flipped = torch.flip(images, dims=[-1])
            emb1 = model.get_embedding(images)
            emb2 = model.get_embedding(images_flipped)
            emb = F.normalize(emb1 + emb2, p=2, dim=-1)
            all_emb.append(emb.float().cpu())
            all_lbl.append(labels)

    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    save_stats({
        'model': f'CNN ({backbone} + ArcFace) [TUNED]',
        'backbone': backbone,
        'embedding_dim': embedding_dim,
        'arcface_scale': arcface_scale,
        'arcface_margin': arcface_margin,
        'label_smoothing': label_smoothing,
        'best_val_r1': best_r1,
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
        'history': history,
        'tuning': {
            'mixup': use_mixup, 'swa': use_swa, 'tta': use_tta,
            'label_smoothing': label_smoothing,
        },
    }, str(PROJECT_ROOT / 'outputs/stats/cnn_results.json'))

    print(f"\n✅ CNN results saved to outputs/stats/cnn_results.json")


if __name__ == '__main__':
    main()
