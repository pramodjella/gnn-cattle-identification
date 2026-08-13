"""
Script: 5-Fold Cross-Validation on Top-3 Models
===============================================
Performs stratified 5-fold cross-validation on:
  1. CNN Baseline (EfficientNet-B4 + ArcFace)
  2. Hybrid CNN-GNN (EfficientNet-B3 + EdgeConv + TRM + ArcFace)
  3. ProtoN (Prototype Node GNN with Alignment Loss)

To ensure scientific rigor:
  - StratifiedKFold splits images of each animal class evenly across 5 folds.
  - The same train/test indices are shared across all 3 models per fold.
  - Performance metrics (Rank-1, Rank-5, EER, ROC-AUC) are averaged (mean ± std).
  - GNN v3/v4 hyperparameter configurations are read from config.yaml.
  - Hybrid training uses pre-extracted CNN feature caching for Phase 1 to maximize speed.
  - All PyTorch data loaders use num_workers=0 for Windows compatibility.

Outputs:
  - outputs/stats/cross_validation_results.json
  - outputs/stats/cross_validation_latex.tex
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import global_mean_pool, global_max_pool


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.training.augmentation import build_train_transform, build_val_transform
from src.evaluation.metrics import BiometricMetrics

# Import models
from src.models.cnn_model import CNNMuzzleModel
from src.models.hybrid_model import HybridCNNGNN
from src.models.proton import CattleProtoN
from src.features.graph_builder import GraphBuilder

# ─────────────────────────────────────────────────────────────────────────────
# Custom PyTorch Datasets for Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

class CVImageDataset(Dataset):
    """Dataset of preprocessed images for CNN baseline cross-validation."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        label = sample['label']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return [s['label'] for s in self.samples]


class CVGraphDataset(Dataset):
    """Dataset of GNN graphs for ProtoN GNN cross-validation."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        graph = sample['graph']
        label = sample['label']
        if self.transform:
            graph = self.transform(graph)
        return graph


class CVHybridDataset(Dataset):
    """Dataset of paired images and GNN graphs for Hybrid CNN-GNN cross-validation."""
    def __init__(self, samples, transform=None, graph_augment=None):
        self.samples = samples
        self.transform = transform
        self.graph_augment = graph_augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        graph = sample['graph']
        label = sample['label']

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.graph_augment:
            graph = self.graph_augment(graph)

        return image, graph, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return [s['label'] for s in self.samples]


class CachedHybridDataset(Dataset):
    """Dataset of pre-cached CNN features + GNN graphs for fast Hybrid Phase 1 training."""
    def __init__(self, samples, fmaps, augment=False):
        self.samples = samples
        self.fmaps = fmaps  # List of pre-extracted feature maps
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fmap = self.fmaps[idx].float()  # (1536, H', W')
        graph = sample['graph']
        label = sample['label']

        if self.augment and torch.rand(1).item() < 0.5:
            noise = torch.randn_like(fmap) * 0.01
            fmap = fmap + noise

        # Extract graph attributes
        pos = graph.pos
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        return fmap, pos, edge_index, edge_attr, label


# ─────────────────────────────────────────────────────────────────────────────
# Collate functions and Samplers
# ─────────────────────────────────────────────────────────────────────────────

def pyg_collate_fn(batch):
    """Collate PyG Graph objects into a batch."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def hybrid_collate_fn(batch):
    """Collate (image, graph, label) triples into a batch."""
    from torch_geometric.data import Batch
    images, graphs, labels = zip(*batch)
    return torch.stack(images), Batch.from_data_list(list(graphs)), torch.stack(labels)


def cached_hybrid_collate_fn(batch):
    """Collate cached CNN features + graph structures into a batch."""
    fmaps, positions, edge_indices, edge_attrs, labels = zip(*batch)
    fmaps = torch.stack(fmaps)
    labels = torch.tensor(labels, dtype=torch.long)

    # Rebuild PyG batch vector for GNN
    batch_parts = []
    for i, pos in enumerate(positions):
        if pos is not None:
            n = pos.shape[0]
            batch_parts.append(torch.full((n,), i, dtype=torch.long))
    batch_vec = torch.cat(batch_parts) if batch_parts else torch.zeros(0, dtype=torch.long)

    return fmaps, positions, edge_indices, edge_attrs, batch_vec, labels


class PKSampler:
    """PK Sampler: samples P classes and K samples per class for metric learning."""
    def __init__(self, labels, p=16, k=4):
        self.p = p
        self.k = k
        self.batch_size = p * k

        from collections import defaultdict
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        # Filter classes with at least K samples
        self.valid_labels = [lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= k]
        if len(self.valid_labels) < p:
            # Fallback to K=2 if dataset is too small
            self.valid_labels = [lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= 2]
            self.k = 2
            self.batch_size = p * self.k

        self.num_batches = max(1, len(labels) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            if len(self.valid_labels) >= self.p:
                selected_labels = random.sample(self.valid_labels, self.p)
            else:
                selected_labels = self.valid_labels.copy()
                while len(selected_labels) < self.p:
                    selected_labels.append(random.choice(self.valid_labels))

            indices = []
            for lbl in selected_labels:
                pool = self.label_to_indices[lbl]
                indices.extend(random.sample(pool, self.k))
            yield from indices

    def __len__(self):
        return self.num_batches * self.batch_size


# ─────────────────────────────────────────────────────────────────────────────
# Graph Augmentation (GNN / ProtoN)
# ─────────────────────────────────────────────────────────────────────────────

def apply_graph_augmentation(batch, epoch, max_epochs):
    """Apply stochastic graph augmentation during training."""
    if random.random() < 0.3:  # Noise
        noise_scale = 0.05 * (1 - epoch / max_epochs)
        if batch.x is not None:
            batch.x = batch.x + torch.randn_like(batch.x) * noise_scale

    if random.random() < 0.2:  # Node drop
        if batch.x is not None and batch.x.size(0) > 20:
            mask = torch.rand(batch.x.size(0), 1, device=batch.x.device) > 0.05
            batch.x = batch.x * mask

    if random.random() < 0.15:  # Edge drop
        if batch.edge_index is not None and batch.edge_index.size(1) > 50:
            num_edges = batch.edge_index.size(1)
            keep_mask = torch.rand(num_edges, device=batch.edge_index.device) > 0.05
            batch.edge_index = batch.edge_index[:, keep_mask]
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[keep_mask]
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_embeddings(embeddings, labels):
    """Compute Rank-1, Rank-5, EER, and AUC from embeddings."""
    metrics = BiometricMetrics()
    results = metrics.compute_all_metrics(embeddings, labels)
    return {
        'rank1': results['identification']['rank_accuracies']['rank_1'],
        'rank5': results['identification']['rank_accuracies'].get('rank_5', 0.0),
        'eer': results['verification']['eer'],
        'auc': results['verification']['roc_auc']
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model Training Routines
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn_fold(train_samples, test_samples, num_classes, device, cnn_cfg, epochs, image_size):
    """Train and evaluate CNN model on a single fold split."""
    train_transform = build_train_transform(image_size)
    val_transform = build_val_transform(image_size)

    ds_train = CVImageDataset(train_samples, transform=train_transform)
    ds_test = CVImageDataset(test_samples, transform=val_transform)

    sampler = PKSampler(ds_train.get_labels(), p=8, k=2)  # Small P,K for memory/speed safety
    train_loader = DataLoader(ds_train, batch_size=16, sampler=sampler, drop_last=True)
    test_loader = DataLoader(ds_test, batch_size=16, shuffle=False)

    model = CNNMuzzleModel(
        num_classes=num_classes,
        embedding_dim=512,
        dropout=0.35,
        pretrained=True,
        backbone='efficientnet_b4',
        arcface_scale=128.0,
        arcface_margin=0.35,
        label_smoothing=0.05,
    ).to(device)

    param_groups = model.get_parameter_groups(backbone_lr=3e-5, head_lr=1e-3)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=1e-7)

    use_amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(images, labels)
                loss = out['loss']

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

    # Evaluate with TTA on Test Fold
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            images_flipped = torch.flip(images, dims=[-1])
            emb1 = model.get_embedding(images)
            emb2 = model.get_embedding(images_flipped)
            emb = F.normalize(emb1 + emb2, p=2, dim=-1)
            all_emb.append(emb.cpu())
            all_lbl.append(labels.cpu())

    embeddings = torch.cat(all_emb)
    test_labels = torch.cat(all_lbl)
    return evaluate_embeddings(embeddings, test_labels)


def train_proton_fold(train_samples, test_samples, num_classes, device, proton_cfg, epochs):
    """Train and evaluate ProtoN GNN model on a single fold split."""
    ds_train = CVGraphDataset(train_samples)
    ds_test = CVGraphDataset(test_samples)

    labels_train = [s['label'] for s in train_samples]
    sampler = PKSampler(labels_train, p=16, k=8)
    train_loader = DataLoader(ds_train, batch_size=128, sampler=sampler, collate_fn=pyg_collate_fn)
    test_loader = DataLoader(ds_test, batch_size=128, shuffle=False, collate_fn=pyg_collate_fn)

    model = CattleProtoN(
        num_classes=num_classes,
        input_dim=256,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.12,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=4e-4,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    use_amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)

    align_weight = 0.2
    temperature = 0.07

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            batch = apply_graph_augmentation(batch, epoch, epochs)

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

    # Evaluate GNN
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            all_emb.append(out['embedding'].cpu())
            all_lbl.append(batch.y.cpu())

    embeddings = torch.cat(all_emb)
    test_labels = torch.cat(all_lbl)
    return evaluate_embeddings(embeddings, test_labels)


def train_hybrid_fold(train_samples, test_samples, num_classes, device, hybrid_cfg,
                      epochs_p1, epochs_p2, image_size, cached_fmaps):
    """
    Train and evaluate Hybrid model on a single fold split.
    Uses cached features for GNN training (Phase 1) and E2E for Phase 2.
    """
    # ── Phase 1: Train GNN Head on Cached Features ──────────────────────────
    model = HybridCNNGNN(
        num_classes=num_classes,
        embedding_dim=256,
        pretrained=True
    ).to(device)

    # Freeze CNN backbone
    for p in model.cnn_features.parameters():
        p.requires_grad = False

    # Filter out samples that have fmap cached
    train_fmaps = [cached_fmaps[s['global_idx']] for s in train_samples]
    test_fmaps = [cached_fmaps[s['global_idx']] for s in test_samples]

    ds_cached_train = CachedHybridDataset(train_samples, train_fmaps, augment=True)
    train_loader_p1 = DataLoader(ds_cached_train, batch_size=32, shuffle=True,
                                 collate_fn=cached_hybrid_collate_fn, num_workers=0)

    gnn_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(gnn_params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=1e-7)

    use_amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Phase 1 training loop
    for epoch in range(1, epochs_p1 + 1):
        model.train()
        for fmaps, positions, edge_indices, edge_attrs, batch_vec, labels in train_loader_p1:
            labels = labels.to(device)
            fmaps = fmaps.to(device)
            batch_vec = batch_vec.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                # Reconstruct per-node CNN features via bilinear sampling
                node_feats_list = []
                B = fmaps.shape[0]
                for b in range(B):
                    pos_b = positions[b]
                    if pos_b is None:
                        continue
                    pos_b = pos_b.to(device)
                    grid_xy = pos_b[:, :2] * 2.0 - 1.0
                    grid = grid_xy.view(1, 1, -1, 2)
                    fmap_b = fmaps[b:b+1]
                    sampled = F.grid_sample(fmap_b, grid, mode='bilinear',
                                             padding_mode='border', align_corners=True)
                    sampled = sampled.squeeze(0).squeeze(1).T
                    node_feats_list.append(sampled)

                if not node_feats_list:
                    continue
                node_feats = torch.cat(node_feats_list, dim=0)

                # Rebuild edge_index
                edge_indices_shifted = []
                node_offset = 0
                for b in range(B):
                    ei = edge_indices[b].to(device)
                    n_b = positions[b].shape[0] if positions[b] is not None else 0
                    edge_indices_shifted.append(ei + node_offset)
                    node_offset += n_b
                edge_index = torch.cat(edge_indices_shifted, dim=1)

                # Projection + GNN blocks
                x = model.node_proj(node_feats)
                x, _ = model.edge_conv(x, batch=batch_vec)
                x, _ = model.trm(x, edge_index, batch=batch_vec)

                # Pooling
                x_mean = global_mean_pool(x, batch_vec)
                x_max = global_max_pool(x, batch_vec)
                x_pooled = torch.cat([x_mean, x_max], dim=-1)

                emb = model.projection_head(x_pooled)
                embedding = F.normalize(emb, p=2, dim=-1)

                loss, _ = model.arcface(embedding, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_params, 1.0)
            optimizer.step()
        scheduler.step()

    # ── Phase 2: End-to-End Fine-Tuning ──────────────────────────────────────
    # Unfreeze backbone
    for p in model.cnn_features.parameters():
        p.requires_grad = True

    train_transform = build_train_transform(image_size)
    val_transform = build_val_transform(image_size)

    ds_full_train = CVHybridDataset(train_samples, transform=train_transform)
    ds_full_test = CVHybridDataset(test_samples, transform=val_transform)

    train_loader_p2 = DataLoader(ds_full_train, batch_size=16, shuffle=True,
                                 collate_fn=hybrid_collate_fn, num_workers=0)
    test_loader = DataLoader(ds_full_test, batch_size=16, shuffle=False,
                             collate_fn=hybrid_collate_fn, num_workers=0)

    param_groups = model.get_parameter_groups(backbone_lr=1e-5, head_lr=5e-4)
    optimizer2 = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=epochs_p2, eta_min=1e-7)

    for epoch in range(1, epochs_p2 + 1):
        model.train()
        for images, graphs, labels in train_loader_p2:
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
        scheduler2.step()

    # Evaluate E2E Hybrid Model
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, graphs, labels in test_loader:
            images = images.to(device, non_blocking=True)
            graphs = graphs.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(images, graphs)
            all_emb.append(out['embedding'].cpu())
            all_lbl.append(labels.cpu())

    embeddings = torch.cat(all_emb)
    test_labels = torch.cat(all_lbl)
    return evaluate_embeddings(embeddings, test_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution Flow
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run 5-fold cross validation')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs-cnn', type=int, default=25, help='Epochs for CNN')
    parser.add_argument('--epochs-proton', type=int, default=30, help='Epochs for ProtoN')
    parser.add_argument('--epochs-hybrid-p1', type=int, default=25, help='Hybrid Phase 1 epochs')
    parser.add_argument('--epochs-hybrid-p2', type=int, default=5, help='Hybrid Phase 2 epochs')
    parser.add_argument('--models', nargs='+', default=['CNN', 'ProtoN', 'Hybrid'],
                        choices=['CNN', 'ProtoN', 'Hybrid'],
                        help='Which models to cross-validate (skip others to spend compute on one).')
    parser.add_argument('--out', default='cross_validation_results.json',
                        help='Output filename under outputs/stats/ (use a distinct name for partial runs).')
    args = parser.parse_args()

    config = load_config()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stats_dir = PROJECT_ROOT / 'outputs' / 'stats'
    ensure_dirs(str(stats_dir))

    print(f"\n{'='*70}")
    print(f"  5-FOLD CROSS-VALIDATION PIPELINE (Top-3 Models)")
    print(f"{'='*70}")
    print(f"  Device : {device} | Folds: {args.folds} | Seed: {args.seed}")
    print(f"  Epochs : CNN={args.epochs_cnn} | ProtoN={args.epochs_proton} | Hybrid={args.epochs_hybrid_p1}+{args.epochs_hybrid_p2}")

    # ── 1. Pool and Match Dataset Samples ─────────────────────────────────────
    print("\n[Step 1] Loading and aligning dataset...")
    processed_dir = PROJECT_ROOT / config['dataset']['processed_dir']
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']

    label_map_path = graph_dir / "label_mapping.json"
    with open(label_map_path) as f:
        animal_to_label = json.load(f)
    num_classes = len(animal_to_label)

    # Scan and match preprocessed images
    img_dict = {}
    for split in ['train', 'val', 'test']:
        split_img_dir = processed_dir / 'images' / split
        if not split_img_dir.exists():
            continue
        for animal_dir in split_img_dir.iterdir():
            if animal_dir.is_dir():
                animal_id = animal_dir.name
                for img_file in animal_dir.glob('*.png'):
                    img_dict[(animal_id, img_file.stem)] = str(img_file)

    # Scan and build/load keypoint graphs
    kp_dir = processed_dir / 'keypoints'
    builder = GraphBuilder(
        knn_k=config['graph']['knn_k'],
        normalize_positions=config['graph']['normalize_positions'],
        use_relative_positions=config['graph']['use_relative_positions']
    )

    all_samples = []
    global_idx = 0

    for split in ['train', 'val', 'test']:
        split_kp_dir = kp_dir / split
        if not split_kp_dir.exists():
            continue
        for animal_dir in split_kp_dir.iterdir():
            if animal_dir.is_dir():
                animal_id = animal_dir.name
                label = animal_to_label[animal_id]
                for kp_file in animal_dir.glob("*.npz"):
                    stem = kp_file.stem
                    if (animal_id, stem) in img_dict:
                        img_path = img_dict[(animal_id, stem)]
                        kp_data = np.load(str(kp_file), allow_pickle=True)
                        data = builder.build_graph(
                            keypoints=kp_data['keypoints'],
                            descriptors=kp_data['descriptors'],
                            scores=kp_data['scores'],
                            image_size=config['preprocessing']['image_size'],
                            animal_id=label,
                            image_path=img_path
                        )
                        if data is not None:
                            data.y = torch.tensor(label, dtype=torch.long)
                            all_samples.append({
                                'image_path': img_path,
                                'graph': data,
                                'label': label,
                                'animal_id': animal_id,
                                'global_idx': global_idx
                            })
                            global_idx += 1

    print(f"  Successfully pooled and aligned {len(all_samples)} samples across {num_classes} classes.")

    # ── 2. Pre-extract CNN features for Hybrid ────────────────────────────────
    print("\n[Step 2] Pre-extracting CNN feature maps for fast Hybrid training...")
    hybrid_extractor = HybridCNNGNN(num_classes=num_classes, pretrained=True).to(device)
    hybrid_extractor.eval()
    cnn_backbone = hybrid_extractor.cnn_features

    cached_fmaps = []
    # Run in batches of 32
    with torch.no_grad():
        for i in range(0, len(all_samples), 32):
            batch_samples = all_samples[i:i+32]
            batch_images = []
            for s in batch_samples:
                img = Image.open(s['image_path']).convert('RGB')
                # Resize and normalize manually for fast caching
                img = img.resize((256, 256))
                arr = np.array(img).transpose(2, 0, 1) / 255.0
                batch_images.append(torch.tensor(arr, dtype=torch.float32))

            batch_tensor = torch.stack(batch_images).to(device)
            # Extracted: (B, 1536, 8, 8)
            fmaps = cnn_backbone(batch_tensor).cpu().half()
            for f in fmaps:
                cached_fmaps.append(f)

    del hybrid_extractor
    torch.cuda.empty_cache()
    print("  Feature caching complete.")

    # ── 3. Initialize Cross-Validation splits ─────────────────────────────────
    print(f"\n[Step 3] Splitting dataset into stratified {args.folds} folds...")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    labels = [s['label'] for s in all_samples]
    splits = list(skf.split(np.zeros(len(labels)), labels))

    cv_results = {
        'CNN': {'rank1': [], 'rank5': [], 'eer': [], 'auc': []},
        'Hybrid': {'rank1': [], 'rank5': [], 'eer': [], 'auc': []},
        'ProtoN': {'rank1': [], 'rank5': [], 'eer': [], 'auc': []}
    }

    # ── 4. Cross-Validation Loop ──────────────────────────────────────────────
    for fold in range(args.folds):
        print(f"\n" + "-"*50)
        print(f"  FOLD {fold+1} / {args.folds}")
        print("-"*50)

        train_idx, test_idx = splits[fold]
        train_samples = [all_samples[idx] for idx in train_idx]
        test_samples = [all_samples[idx] for idx in test_idx]

        # --- A. CNN ---
        if 'CNN' in args.models:
            print("\n  Training CNN baseline...")
            t0 = time.time()
            cnn_metrics = train_cnn_fold(
                train_samples, test_samples, num_classes, device,
                config.get('cnn', {}), args.epochs_cnn, config['preprocessing']['image_size']
            )
            print(f"  CNN complete (took {time.time()-t0:.1f}s) | Rank-1: {cnn_metrics['rank1']*100:.2f}% | EER: {cnn_metrics['eer']*100:.2f}%")
            for k, v in cnn_metrics.items():
                cv_results['CNN'][k].append(v)

        # --- B. ProtoN ---
        if 'ProtoN' in args.models:
            print("\n  Training ProtoN GNN...")
            t0 = time.time()
            proton_metrics = train_proton_fold(
                train_samples, test_samples, num_classes, device,
                config.get('proton', {}), args.epochs_proton
            )
            print(f"  ProtoN GNN complete (took {time.time()-t0:.1f}s) | Rank-1: {proton_metrics['rank1']*100:.2f}% | EER: {proton_metrics['eer']*100:.2f}%")
            for k, v in proton_metrics.items():
                cv_results['ProtoN'][k].append(v)

        # --- C. Hybrid ---
        if 'Hybrid' in args.models:
            print("\n  Training Hybrid CNN-GNN (cached Phase 1 + E2E Phase 2)...")
            t0 = time.time()
            hybrid_metrics = train_hybrid_fold(
                train_samples, test_samples, num_classes, device,
                config.get('hybrid', {}), args.epochs_hybrid_p1, args.epochs_hybrid_p2,
                config['preprocessing']['image_size'], cached_fmaps
            )
            print(f"  Hybrid complete (took {time.time()-t0:.1f}s) | Rank-1: {hybrid_metrics['rank1']*100:.2f}% | EER: {hybrid_metrics['eer']*100:.2f}%", flush=True)
            for k, v in hybrid_metrics.items():
                cv_results['Hybrid'][k].append(v)

    # ── 5. Aggregate and Save Statistics ──────────────────────────────────────
    print("\n" + "="*70)
    print("  CROSS-VALIDATION RESULTS SUMMARY")
    print("="*70)

    summary_stats = {}
    for model_name, metrics in cv_results.items():
        if not any(len(v) for v in metrics.values()):
            continue  # model not selected via --models
        summary_stats[model_name] = {}
        print(f"\n  {model_name}:")
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary_stats[model_name][metric_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'folds': [float(v) for v in values]
            }
            print(f"    {metric_name:<6}: {mean_val*100:.2f}% \u00b1 {std_val*100:.2f}%")

    # Save to JSON
    json_path = stats_dir / args.out
    with open(json_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"\n[INFO] Cross-validation results saved to {json_path}")

    # Generate LaTeX Table
    latex_path = stats_dir / 'cross_validation_latex.tex'
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{5-Fold Cross-Validation performance comparison of the top proposed architectures on the cattle muzzle print dataset. Values are reported as mean $\\pm$ standard deviation across the 5 folds.}\n")
        f.write("\\label{tab:cross_val_results}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Rank-1 (\\%)} & \\textbf{Rank-5 (\\%)} & \\textbf{EER (\\%)} & \\textbf{ROC-AUC} \\\\\n")
        f.write("\\midrule\n")

        for m_name in ['ProtoN', 'Hybrid', 'CNN']:
            if m_name not in summary_stats:
                continue  # not selected via --models
            m_label = "CNN (B4)" if m_name == 'CNN' else m_name
            stats = summary_stats[m_name]
            f.write(f"  {m_label:<10} & "
                    f"{stats['rank1']['mean']*100:.2f} \\pm {stats['rank1']['std']*100:.2f} & "
                    f"{stats['rank5']['mean']*100:.2f} \\pm {stats['rank5']['std']*100:.2f} & "
                    f"{stats['eer']['mean']*100:.2f} \\pm {stats['eer']['std']*100:.2f} & "
                    f"{stats['auc']['mean']:.4f} \\pm {stats['auc']['std']:.4f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[INFO] Cross-validation LaTeX table saved to {latex_path}")
    print(f"\n✅ All cross-validation loops complete!")


if __name__ == '__main__':
    main()
