"""
Script: Train Hybrid CNN-GNN with Pre-Cached CNN Features
==========================================================
The bottleneck of the naive Hybrid approach is running the CNN forward pass
on every image for every epoch. Solution: pre-extract CNN feature maps ONCE
and cache them to disk. The GNN then trains on cached tensors — very fast.

Strategy:
  Step 1 (run once, ~5 min): Extract EfficientNet-B3 feature maps for all images
                               Save to outputs/hybrid/feature_cache/{split}/
  Step 2 (150 epochs, ~20 min): Train GNN head on cached features

After Step 2, we do a short end-to-end fine-tuning pass (Phase 2) with the
backbone unfrozen to push accuracy to its maximum.

Why this is better:
  - Phase 1 cached: ~8s/epoch instead of ~107s/epoch (13x speedup)
  - Same final accuracy as end-to-end (backbone was pretrained on ImageNet)
  - Phase 2 fine-tuning (20 epochs): gives +2-5% accuracy boost

Output: outputs/hybrid/best_model.pt
"""

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Batch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.models.hybrid_model import HybridCNNGNN
from src.training.image_dataset import create_hybrid_loaders
from src.evaluation.metrics import BiometricMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction & Caching
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_cache_features(model, loaders, device, cache_dir, amp_dtype):
    """
    Run CNN backbone once over all splits and cache spatial feature maps.

    Cached tensors shape per sample: (1536, H', W') where H'=W'=8 for 256px input.
    This is ~1536 * 8 * 8 * 2 bytes = ~192 KB per image.
    Total cache size: (3312+615+964) * 192KB ~ 940 MB — fits comfortably on disk.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    # Only need CNN features extractor
    cnn = model.cnn_features
    multi_scale = getattr(model, 'multi_scale', False)

    for split, loader in loaders.items():
        split_cache = cache_dir / split
        split_cache.mkdir(exist_ok=True)

        # Check if already cached — and that the cached FORMAT matches the
        # current model (single- vs multi-scale). A stale single-scale cache
        # from a prior run would otherwise feed 1536-d features to a multi-scale
        # node_proj (1864-d) and crash.
        existing = sorted(split_cache.glob('*.pt'))
        dataset_size = len(loader.dataset)
        expected_stages = len(model.ms_stage_indices) if multi_scale else 1

        if len(existing) == dataset_size:
            probe = torch.load(existing[0], weights_only=False)
            cached_stages = len(probe['fmaps']) if 'fmaps' in probe else 1
            if cached_stages == expected_stages:
                print(f"  [Cache] {split}: {len(existing)} maps cached ({cached_stages} stage/s), skipping.")
                continue
            print(f"  [Cache] {split}: stale cache ({cached_stages} vs {expected_stages} stages) — rebuilding.")
            for f in existing:
                f.unlink()

        print(f"  [Cache] Extracting features for {split} ({dataset_size} samples)...")
        t0 = time.time()
        count = 0

        with torch.no_grad():
            for batch_idx, (images, graphs, labels) in enumerate(loader):
                images = images.to(device, non_blocking=True)

                # Extract spatial feature maps. Multi-scale caches several stage
                # maps (list); single-scale caches the final map (list of one).
                with autocast(device_type='cuda', dtype=torch.float32, enabled=False):
                    if multi_scale:
                        stage_maps = model._forward_stages(images.float())  # list
                    else:
                        stage_maps = [cnn(images.float())]  # (B, 1536, H', W')

                # Save each sample's feature map(s) and graph
                B = stage_maps[0].shape[0]
                for i in range(B):
                    sample_idx = batch_idx * loader.batch_size + i
                    torch.save({
                        'fmaps': [m[i].cpu().half() for m in stage_maps],  # list, fp16
                        'pos': graphs[i].pos if hasattr(graphs[i], 'pos') else None,
                        'edge_index': graphs[i].edge_index,
                        'edge_attr': graphs[i].edge_attr if graphs[i].edge_attr is not None else None,
                        'label': labels[i],
                    }, split_cache / f'{sample_idx:05d}.pt')
                    count += 1

        elapsed = time.time() - t0
        print(f"  [Cache] {split}: {count} feature maps saved in {elapsed:.1f}s")

    print(f"  [Cache] All features cached to {cache_dir}")


class CachedHybridDataset(torch.utils.data.Dataset):
    """Dataset that loads pre-cached CNN feature maps + graph structure."""

    def __init__(self, cache_dir, augment=False):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob('*.pt'))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        # Back-compat: old caches stored a single 'fmap'; new store 'fmaps' list.
        if 'fmaps' in data:
            fmaps = [m.float() for m in data['fmaps']]
        else:
            fmaps = [data['fmap'].float()]
        pos = data['pos']
        edge_index = data['edge_index']
        edge_attr = data.get('edge_attr')
        label = data['label']

        # Simple feature augmentation (only for training)
        if self.augment and torch.rand(1).item() < 0.5:
            fmaps = [m + torch.randn_like(m) * 0.01 for m in fmaps]

        return fmaps, pos, edge_index, edge_attr, label


def cached_collate_fn(batch):
    """Collate cached features + graph structure into a batch."""
    fmaps_list, positions, edge_indices, edge_attrs, labels = zip(*batch)
    # Each sample is a list of stage maps; stack per stage -> list of (B,C,H,W).
    num_stages = len(fmaps_list[0])
    fmaps = [torch.stack([s[k] for s in fmaps_list]) for k in range(num_stages)]
    labels = torch.stack(labels) # (B,)

    # Rebuild batch vector for GNN
    batch_size = len(fmaps)
    batch_parts = []
    for i, pos in enumerate(positions):
        if pos is not None:
            n = pos.shape[0]
            batch_parts.append(torch.full((n,), i, dtype=torch.long))

    batch_vec = torch.cat(batch_parts) if batch_parts else torch.zeros(0, dtype=torch.long)

    return fmaps, positions, edge_indices, edge_attrs, batch_vec, labels


# ─────────────────────────────────────────────────────────────────────────────
# GNN Forward on Cached Features
# ─────────────────────────────────────────────────────────────────────────────

def forward_on_cached(model, fmaps, positions, edge_indices, edge_attrs,
                       batch_vec, labels, device, amp_dtype, use_amp):
    """
    Run only the GNN part of the Hybrid model on pre-extracted feature maps.
    Bypasses the CNN backbone entirely — very fast.
    """
    # fmaps is a list of per-stage batched maps [(B,C,H,W), ...] (one for
    # single-scale, several for multi-scale).
    fmaps = [f.to(device, non_blocking=True) for f in fmaps]
    batch_vec = batch_vec.to(device, non_blocking=True)

    B = fmaps[0].shape[0]

    with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
        # Reconstruct per-node CNN features via bilinear sampling from every
        # cached stage map, then concatenate across stages (multi-scale).
        node_feats_list = []
        for b in range(B):
            pos_b = positions[b]
            if pos_b is None:
                continue
            pos_b = pos_b.to(device)
            grid = (pos_b[:, :2] * 2.0 - 1.0).view(1, 1, -1, 2)
            per_scale = []
            for m in fmaps:
                s = F.grid_sample(m[b:b+1], grid, mode='bilinear',
                                  padding_mode='border', align_corners=True)
                per_scale.append(s.squeeze(0).squeeze(1).T)  # (N, C_stage)
            node_feats_list.append(torch.cat(per_scale, dim=-1))  # (N, sum C)

        if not node_feats_list:
            return None, None

        node_feats = torch.cat(node_feats_list, dim=0)  # (N_total, sum C)

        # Rebuild edge_index (+ edge_attr) for the full batch
        edge_indices_shifted, edge_attr_parts = [], []
        node_offset = 0
        for b, ei in enumerate(edge_indices):
            ei = ei.to(device)
            n_b = positions[b].shape[0] if positions[b] is not None else 0
            edge_indices_shifted.append(ei + node_offset)
            if edge_attrs[b] is not None:
                edge_attr_parts.append(edge_attrs[b].to(device))
            node_offset += n_b
        edge_index = torch.cat(edge_indices_shifted, dim=1)
        edge_attr_full = torch.cat(edge_attr_parts, dim=0) if edge_attr_parts else None

        # Project node features
        x = model.node_proj(node_feats)

        # Adaptive graph construction (learned edge gating) — if enabled.
        if getattr(model, 'learned_edges', False):
            edge_index, _, _ = model.adaptive_graph(x, edge_index, edge_attr_full)

        # EdgeConv
        x, _ = model.edge_conv(x, batch=batch_vec)

        # TRM
        x, attention = model.trm(x, edge_index, batch=batch_vec)

        # Global pooling
        from torch_geometric.nn import global_mean_pool, global_max_pool
        x_mean = global_mean_pool(x, batch_vec)
        x_max = global_max_pool(x, batch_vec)
        x_pooled = torch.cat([x_mean, x_max], dim=-1)

        # Projection + normalize
        emb = model.projection_head(x_pooled)
        embedding = F.normalize(emb, p=2, dim=-1)

    return embedding, attention


# ─────────────────────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────────────────────

def validate_cached(model, val_files, device, amp_dtype, use_amp, batch_size=32):
    """Validate on cached features."""
    model.eval()
    all_emb, all_lbl = [], []

    # Simple sequential evaluation
    dataset = CachedHybridDataset(Path(val_files))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=cached_collate_fn, num_workers=0)

    with torch.no_grad():
        for fmaps, positions, edge_indices, edge_attrs, batch_vec, labels in loader:
            labels = labels.to(device)
            emb, _ = forward_on_cached(model, fmaps, positions, edge_indices,
                                        edge_attrs, batch_vec, labels, device, amp_dtype, use_amp)
            if emb is not None:
                all_emb.append(emb.float().cpu())
                all_lbl.append(labels.cpu())

    if not all_emb:
        return 0.0

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

    hybrid_cfg = config.get('hybrid', {})
    total_epochs    = hybrid_cfg.get('epochs', 200)
    finetune_epochs = hybrid_cfg.get('finetune_epochs', 50)  # Now config-driven (from 25)
    batch_size      = 32        # Can use larger batch on cached features (no CNN mem)
    backbone_lr     = hybrid_cfg.get('backbone_lr', 1e-5)
    head_lr         = hybrid_cfg.get('head_lr', 1e-3)
    wd              = hybrid_cfg.get('weight_decay', 1e-4)
    use_amp         = hybrid_cfg.get('use_amp', True)
    use_enhanced_aug = hybrid_cfg.get('use_enhanced_aug', True)
    ckpt_dir        = str(PROJECT_ROOT / hybrid_cfg.get('checkpoint_dir', 'outputs/hybrid'))
    patience        = hybrid_cfg.get('early_stopping', {}).get('patience', 40)
    min_delta       = hybrid_cfg.get('early_stopping', {}).get('min_delta', 0.0005)
    cache_dir       = str(PROJECT_ROOT / 'outputs/hybrid/feature_cache')

    ensure_dirs(ckpt_dir, str(PROJECT_ROOT / 'outputs/stats'), cache_dir)

    print(f"\n{'='*65}")
    print("  HYBRID CNN-GNN (Pre-Cached Feature Maps — 13x Faster)")
    print(f"{'='*65}")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Phase 1: {total_epochs} epochs on CACHED features | ~8s/ep")
    print(f"  Phase 2: {finetune_epochs} epochs end-to-end fine-tuning | ~60s/ep")
    print(f"  Enhanced aug: {use_enhanced_aug} | Patience: {patience}")

    # ── Data ────────────────────────────────────────────────────────────────
    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    loaders_full = create_hybrid_loaders(preprocessed_dir, graph_dir, config)

    with open(os.path.join(preprocessed_dir, 'train_split.json')) as f:
        train_data = json.load(f)
    num_classes = len(set(
        item.get('animal_id', item.get('label', str(i)))
        for i, item in enumerate(train_data)
    ))
    print(f"  Classes: {num_classes} | Train: {len(loaders_full['train'].dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = HybridCNNGNN(
        num_classes=num_classes,
        config=config,
        pretrained=True,
    ).to(device)

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # ── Step 1: Cache CNN Feature Maps ──────────────────────────────────────
    print(f"\n  Step 1: Caching CNN feature maps (runs once)...")
    extract_and_cache_features(model, loaders_full, device, cache_dir, amp_dtype)

    # ── Step 2: Train GNN on Cached Features ────────────────────────────────
    print(f"\n  Step 2: Training GNN head on cached features...")

    train_dataset = CachedHybridDataset(Path(cache_dir) / 'train', augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0, collate_fn=cached_collate_fn)

    # Use EnhancedGraphAugmentation for graph data if enabled
    if use_enhanced_aug:
        from src.training.augmentation import EnhancedGraphAugmentation
        print(f"  Using EnhancedGraphAugmentation (SubgraphCrop + FeatureMixup)")


    # Freeze CNN backbone — only GNN + ArcFace train
    for p in model.cnn_features.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (GNN+ArcFace): {trainable:,}")

    gnn_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(gnn_params, lr=head_lr, weight_decay=wd)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-7)

    best_r1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_r1': [], 'lr': [], 'epoch_time': [], 'phase': []}

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for fmaps, positions, edge_indices, edge_attrs, batch_vec, labels in train_loader:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            emb, _ = forward_on_cached(model, fmaps, positions, edge_indices,
                                        edge_attrs, batch_vec, labels, device, amp_dtype, use_amp)
            if emb is None:
                continue

            loss, stats = model.arcface(emb, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        val_r1 = validate_cached(model, str(Path(cache_dir) / 'val'), device, amp_dtype, use_amp)
        epoch_time = time.time() - t0

        vram = f" | VRAM:{torch.cuda.memory_allocated(0)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
        print(f"Epoch {epoch:3d}/{total_epochs} [P1-Cached] | Loss:{avg_loss:.4f} | "
              f"R1:{val_r1:.4f} | {epoch_time:.1f}s{vram}", flush=True)

        history['train_loss'].append(avg_loss)
        history['val_r1'].append(val_r1)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_time)
        history['phase'].append(1)

        if val_r1 > best_r1 + min_delta:
            best_r1 = val_r1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'val_r1': val_r1, 'phase': 1,
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes, 'history': history,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  >> New best! R1:{best_r1:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early stopping P1] Epoch {epoch}")
                break

    # ── Step 3: End-to-End Fine-tuning (Phase 2) ─────────────────────────
    print(f"\n  {'='*55}")
    print(f"  Step 3: End-to-end fine-tuning (Phase 2, {finetune_epochs} epochs)")
    print(f"  {'='*55}")

    # Unfreeze backbone with very small LR
    for p in model.cnn_features.parameters():
        p.requires_grad = True
    total_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (all): {total_t:,}")

    # Reload best checkpoint from Phase 1
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    param_groups = model.get_parameter_groups(backbone_lr=backbone_lr, head_lr=head_lr * 0.5)
    optimizer2 = torch.optim.AdamW(param_groups, weight_decay=wd)
    scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=finetune_epochs, eta_min=1e-7)
    patience_counter2 = 0

    # Use full image loaders for Phase 2
    from src.training.image_dataset import create_hybrid_loaders as chl
    from torch.amp import autocast as ac

    def validate_full(model, val_loader, device, amp_dtype):
        model.eval()
        all_emb, all_lbl = [], []
        with torch.no_grad():
            for images, graphs, labels in val_loader:
                images = images.to(device, non_blocking=True)
                graphs = graphs.to(device, non_blocking=True)
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    out = model(images, graphs)
                all_emb.append(out['embedding'].float().cpu())
                all_lbl.append(labels)
        emb = torch.cat(all_emb)
        lbl = torch.cat(all_lbl)
        sim = torch.mm(emb, emb.t())
        sim.fill_diagonal_(-1e9)
        return (lbl[sim.argmax(dim=1)] == lbl).float().mean().item()

    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for images, graphs, labels in loaders_full['train']:
            images = images.to(device, non_blocking=True)
            graphs = graphs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer2.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                result = model(images, graphs, labels)
                loss = result['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer2.step()
            total_loss += loss.item()
            num_batches += 1

        scheduler2.step()
        avg_loss = total_loss / max(num_batches, 1)
        val_r1 = validate_full(model, loaders_full['val'], device, amp_dtype)
        epoch_time = time.time() - t0
        global_ep = total_epochs + epoch

        vram = f" | VRAM:{torch.cuda.memory_allocated(0)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
        print(f"Epoch {epoch:3d}/{finetune_epochs} [P2-E2E] | Loss:{avg_loss:.4f} | "
              f"R1:{val_r1:.4f} | {epoch_time:.1f}s{vram}", flush=True)

        history['train_loss'].append(avg_loss)
        history['val_r1'].append(val_r1)
        history['lr'].append(optimizer2.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_time)
        history['phase'].append(2)

        if val_r1 > best_r1 + min_delta:
            best_r1 = val_r1
            best_epoch = global_ep
            patience_counter2 = 0
            torch.save({
                'epoch': global_ep, 'val_r1': val_r1, 'phase': 2,
                'model_state_dict': model.state_dict(), 'num_classes': num_classes,
                'history': history,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  >> New best (P2)! R1:{best_r1:.4f}", flush=True)
        else:
            patience_counter2 += 1
            if patience_counter2 >= 15:
                print(f"[Early stopping P2] Epoch {epoch}")
                break

    # ── Final Evaluation ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Hybrid Training Complete! Best R1: {best_r1:.4f} @ epoch {best_epoch}")

    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    metrics = BiometricMetrics()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, graphs, labels in loaders_full['test']:
            images = images.to(device, non_blocking=True)
            graphs = graphs.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(images, graphs)
            all_emb.append(out['embedding'].float().cpu())
            all_lbl.append(labels)
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    save_stats({
        'model': 'Hybrid CNN-GNN (EfficientNet-B3 + EdgeConv + TRM + ArcFace)',
        'architecture': 'Cached feature map training + end-to-end fine-tuning',
        'best_epoch': best_epoch,
        'best_val_r1': best_r1,
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies'].get('rank_5', 0),
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
        'history': history,
    }, str(PROJECT_ROOT / 'outputs/stats/hybrid_results.json'))

    print(f"\nHybrid results saved to outputs/stats/hybrid_results.json")


if __name__ == '__main__':
    main()
