"""
Script: Run All Ablation Studies (Orchestrator)
================================================
Runs all Phase 2 ablation sweeps sequentially:
  1. Backbone (resnet50, efficientnet_b3, convnext_tiny vs anchor efficientnet_b4)
  2. ArcFace scale/margin ((s=64, m=0.5), (s=96, m=0.45), (s=128, m=0.5) vs anchor (s=128, m=0.35))
  3. Graph Construction (k=4, k=12 vs anchor k=8)
  4. Loss Function (CE, ArcFace, ArcFace+Triplet vs anchor ArcFace+Triplet+LS)
  5. Augmentation (None, Standard vs anchor Mixup+Standard)

To save time while obtaining real results, ablation variants are trained for:
  - CNN: 25 epochs (bs=16, ~6 mins per run)
  - GNN v3: 30 epochs (bs=128, ~2 mins per run)
And anchored against fully-trained checkpoints loaded from Phase 1.

Outputs: outputs/stats/ablation_results.json
"""

import os
import sys
import json
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.training.image_dataset import create_image_loaders, MuzzleImageDataset, PKSamplerForImages
from src.training.augmentation import build_train_transform, build_val_transform
from src.models.arcface import ArcFaceLoss
from src.models.gnn_v3 import CattleGNNv3
from src.evaluation.metrics import BiometricMetrics

# ─────────────────────────────────────────────────────────────────────────────
# Ablation CNN Model definition
# ─────────────────────────────────────────────────────────────────────────────

class AblationCNNModel(nn.Module):
    """Flexible CNN model supporting multiple backbones, loss types, and ArcFace options."""
    def __init__(self, num_classes, embedding_dim=512, dropout=0.35, 
                 backbone='efficientnet_b4', arcface_scale=128.0, 
                 arcface_margin=0.35, label_smoothing=0.05, loss_type='arcface_triplet_ls'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.loss_type = loss_type

        # Backbone selection
        from torchvision.models import (
            efficientnet_b4, EfficientNet_B4_Weights,
            efficientnet_b3, EfficientNet_B3_Weights,
            resnet50, ResNet50_Weights,
            convnext_tiny, ConvNeXt_Tiny_Weights
        )

        if backbone == 'efficientnet_b4':
            net = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            self.features = net.features
            self.avgpool = net.avgpool
            self.backbone_out_dim = 1792
        elif backbone == 'efficientnet_b3':
            net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.features = net.features
            self.avgpool = net.avgpool
            self.backbone_out_dim = 1536
        elif backbone == 'resnet50':
            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.avgpool = net.avgpool
            self.backbone_out_dim = 2048
        elif backbone == 'convnext_tiny':
            net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.features = net.features
            self.avgpool = net.avgpool
            self.backbone_out_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Embedding Head
        self.embedding_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone_out_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(512, embedding_dim),
        )

        # Loss Head Selection
        if loss_type == 'ce':
            self.classifier = nn.Linear(embedding_dim, num_classes)
            self.criterion = nn.CrossEntropyLoss()
        else:
            triplet_w = 0.0 if loss_type == 'arcface' else 0.1
            ls = label_smoothing if 'ls' in loss_type else 0.0
            self.arcface = ArcFaceLoss(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                margin=arcface_margin,
                scale=arcface_scale,
                triplet_weight=triplet_w,
                triplet_margin=0.3,
                label_smoothing=ls
            )

    def extract_features(self, x):
        x = self.features(x)
        if self.backbone_name == 'convnext_tiny':
            x = x.mean(dim=[-2, -1])
        else:
            x = self.avgpool(x)
        return x.flatten(1)

    def get_embedding(self, x):
        feat = self.extract_features(x)
        emb = self.embedding_head(feat)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, x, labels=None):
        emb = self.get_embedding(x)
        result = {'embedding': emb}

        if labels is not None:
            if self.loss_type == 'ce':
                logits = self.classifier(emb)
                loss = self.criterion(logits, labels)
                result['loss'] = loss
            else:
                loss, stats = self.arcface(emb, labels)
                result['loss'] = loss
                result['stats'] = stats

        return result

    def get_parameter_groups(self, backbone_lr, head_lr):
        groups = [
            {'params': self.features.parameters(), 'lr': backbone_lr, 'name': 'backbone'},
            {'params': self.avgpool.parameters(), 'lr': backbone_lr, 'name': 'avgpool'},
            {'params': self.embedding_head.parameters(), 'lr': head_lr, 'name': 'head'},
        ]
        if self.loss_type == 'ce':
            groups.append({'params': self.classifier.parameters(), 'lr': head_lr, 'name': 'loss'})
        else:
            groups.append({'params': self.arcface.parameters(), 'lr': head_lr, 'name': 'loss'})
        return groups

# ─────────────────────────────────────────────────────────────────────────────
# Mixup Utility
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = float(torch.distributions.Beta(alpha, alpha).sample())
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y

# ─────────────────────────────────────────────────────────────────────────────
# Training helper for CNN
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn_ablation(config, backbone, loss_type, arcface_scale, arcface_margin, 
                       use_mixup, use_aug, device, epochs=25):
    print(f"\n  [TRAIN CNN] Backbone: {backbone} | Loss: {loss_type} | s={arcface_scale}, m={arcface_margin} | Mixup: {use_mixup} | Aug: {use_aug}")
    
    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    image_size = config.get('preprocessing', {}).get('image_size', 256)
    
    # Custom transforms for Augmentation ablation
    if use_aug:
        train_transform = build_train_transform(image_size)
    else:
        # No aug: resize + normalize only
        train_transform = build_val_transform(image_size)
        
    val_transform = build_val_transform(image_size)
    
    # Loaders
    split_json_train = os.path.join(preprocessed_dir, 'train_split.json')
    split_json_test = os.path.join(preprocessed_dir, 'test_split.json')
    
    ds_train = MuzzleImageDataset(split_json_train, transform=train_transform)
    ds_test = MuzzleImageDataset(split_json_test, transform=val_transform)
    
    num_classes = len(set(ds_train.get_labels()))
    
    batch_size = 16
    sampler = PKSamplerForImages(ds_train.get_labels(), p=4, k=4)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, sampler=sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    model = AblationCNNModel(
        num_classes=num_classes,
        embedding_dim=512,
        backbone=backbone,
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
        label_smoothing=0.05,
        loss_type=loss_type
    ).to(device)
    
    param_groups = model.get_parameter_groups(backbone_lr=3e-5, head_lr=1e-3)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=5e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-7)
    
    use_amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_mixup:
                images, labels = mixup_data(images, labels, alpha=0.2, device=device)
                
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
                
            total_loss += loss.item()
            num_batches += 1
            
        scheduler.step()
        
    # Evaluate Rank-1 on Test set at the end of training
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            # TTA flip evaluation
            images_flipped = torch.flip(images, dims=[-1])
            emb1 = model.get_embedding(images)
            emb2 = model.get_embedding(images_flipped)
            emb = F.normalize(emb1 + emb2, p=2, dim=-1)
            all_emb.append(emb.cpu())
            all_lbl.append(labels)
            
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    sim = torch.mm(emb, emb.t())
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)
    r1 = (lbl[nn_idx] == lbl).float().mean().item()
    print(f"  [EVAL] Final Ablation Test Rank-1: {r1*100:.2f}%")
    return r1

# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction & GNN v3 training helper
# ─────────────────────────────────────────────────────────────────────────────

def build_graphs_for_k(knn_k, config):
    from src.features.graph_builder import GraphBuilder
    
    builder = GraphBuilder(
        knn_k=knn_k,
        normalize_positions=config['graph']['normalize_positions'],
        use_relative_positions=config['graph']['use_relative_positions']
    )
    
    processed_dir = PROJECT_ROOT / config['dataset']['processed_dir']
    kp_dir = processed_dir / "keypoints"
    
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']
    label_map_path = graph_dir / "label_mapping.json"
    with open(label_map_path) as f:
        animal_to_label = json.load(f)
        
    splits = {}
    for split in ['train', 'val', 'test']:
        split_graphs = []
        split_kp_dir = kp_dir / split
        if not split_kp_dir.exists():
            continue
        for animal_dir in split_kp_dir.iterdir():
            if not animal_dir.is_dir():
                continue
            animal_id = animal_dir.name
            label = animal_to_label[animal_id]
            for kp_file in animal_dir.glob("*.npz"):
                kp_data = np.load(str(kp_file), allow_pickle=True)
                data = builder.build_graph(
                    keypoints=kp_data['keypoints'],
                    descriptors=kp_data['descriptors'],
                    scores=kp_data['scores'],
                    image_size=config['preprocessing']['image_size'],
                    animal_id=label,
                    image_path=str(kp_data.get('image_path', ''))
                )
                if data is not None:
                    data.y = torch.tensor(label, dtype=torch.long)
                    split_graphs.append(data)
        splits[split] = split_graphs
    return splits

def get_loaders_for_graphs(graphs_dict, config):
    from src.training.dataset import TransformListDataset, PKSampler
    from torch_geometric.loader import DataLoader
    
    loaders = {}
    batch_size = 64
    for split in ['train', 'val', 'test']:
        g_list = graphs_dict[split]
        dataset = TransformListDataset(g_list, transform=None)
        if split == 'train':
            labels = [g.y.item() for g in g_list]
            pk_config = config['training'].get('triplet', {})
            k_per_class = pk_config.get('samples_per_class', 4)
            p_classes = max(2, batch_size // k_per_class)
            sampler = PKSampler(labels, p=p_classes, k=k_per_class)
            loaders['train'] = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loaders

def train_gnn_ablation(config, knn_k, device, epochs=30):
    print(f"\n  [TRAIN GNN] Building graphs for k={knn_k}...")
    graphs_dict = build_graphs_for_k(knn_k, config)
    loaders = get_loaders_for_graphs(graphs_dict, config)
    
    num_classes = len(graphs_dict['train'][0].y.unique()) if hasattr(graphs_dict['train'][0].y, 'unique') else 260
    # Safeguard num_classes lookup
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']
    label_map_path = graph_dir / "label_mapping.json"
    with open(label_map_path) as f:
        num_classes = len(json.load(f))
        
    v3_cfg = config.get('gnn_v3', {})
    model = CattleGNNv3(
        input_dim=256,
        hidden_dim=v3_cfg.get('hidden_dim', 192),
        num_heads=v3_cfg.get('num_heads', 4),
        num_layers=v3_cfg.get('num_layers', 4),
        edge_enc_dim=v3_cfg.get('edge_enc_dim', 96),
        fusion_dim=v3_cfg.get('fusion_dim', 768),
        projection_hidden=v3_cfg.get('projection_hidden', 512),
        dropout=0.10,
    ).to(device)
    model.set_num_classes(num_classes)
    
    model.arcface = ArcFaceLoss(
        embedding_dim=model.embedding_dim,
        num_classes=num_classes,
        margin=0.35,
        scale=48.0,
        triplet_weight=0.15,
        triplet_margin=0.3,
        label_smoothing=0.05,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=5e-5)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=4e-4,
        epochs=epochs,
        steps_per_epoch=len(loaders['train']),
        pct_start=0.1,
    )
    
    use_amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loaders['train']:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
                loss = out.get('loss', None)
                if loss is None:
                    loss, _ = model.arcface(out['embedding'], batch.y)
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
            
    # Evaluation on Test set
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loaders['test']:
            batch = batch.to(device)
            out = model(batch)
            all_emb.append(out['embedding'].cpu())
            all_lbl.append(batch.y.cpu())
            
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    sim = torch.mm(emb, emb.t())
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)
    r1 = (lbl[nn_idx] == lbl).float().mean().item()
    print(f"  [EVAL] GNN (k={knn_k}) Test Rank-1: {r1*100:.2f}%")
    return r1

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    
    stats_dir = PROJECT_ROOT / 'outputs' / 'stats'
    ensure_dirs(str(stats_dir))
    out_file = stats_dir / 'ablation_results.json'
    
    # ── Load Anchor Baselines (from Phase 1 results) ──────────────────────────
    cnn_anchor_r1 = 0.9544
    gnn_anchor_r1 = 0.9149
    
    cnn_results_file = stats_dir / 'cnn_results.json'
    if cnn_results_file.exists():
        with open(cnn_results_file) as f:
            res = json.load(f)
            cnn_anchor_r1 = float(res.get('test_rank1', cnn_anchor_r1))
            
    gnn_results_file = stats_dir / 'gnn_v3_optimized_results.json'
    if gnn_results_file.exists():
        with open(gnn_results_file) as f:
            res = json.load(f)
            gnn_anchor_r1 = float(res.get('test_rank1', gnn_anchor_r1))
            
    print(f"Loaded anchor results: CNN Rank-1 = {cnn_anchor_r1*100:.2f}% | GNN Rank-1 = {gnn_anchor_r1*100:.2f}%")
    
    # Load existing results if they exist for resuming
    if out_file.exists():
        try:
            with open(out_file) as f:
                ablation_results = json.load(f)
            print(f"Loaded existing ablation results from {out_file}. Resuming...")
        except Exception as e:
            print(f"Could not load existing ablation results: {e}. Starting fresh.")
            ablation_results = {}
    else:
        ablation_results = {}

    def get_existing_result(study_name, label):
        if study_name in ablation_results:
            for item in ablation_results[study_name]:
                if item["label"] == label:
                    return item["rank1"]
        return None

    def save_current_results():
        save_stats(ablation_results, str(out_file))

    # ── 1. Backbone Ablation ──────────────────────────────────────────────────
    print("\n" + "="*50 + "\n1. BACKBONE ABLATION STUDY\n" + "="*50)
    if "Backbone" not in ablation_results:
        ablation_results["Backbone"] = [
            {"label": "EfficientNet-B4", "rank1": cnn_anchor_r1, "baseline": True}
        ]
        save_current_results()
        
    for bb in ['resnet50', 'efficientnet_b3', 'convnext_tiny']:
        label_map = {
            'resnet50': 'ResNet-50',
            'efficientnet_b3': 'EfficientNet-B3',
            'convnext_tiny': 'ConvNeXt-Tiny'
        }
        label = label_map[bb]
        cached_r1 = get_existing_result("Backbone", label)
        if cached_r1 is not None:
            print(f"  [RESUME] Skipping Backbone: {label} (already computed: {cached_r1*100:.2f}%)")
            # Ensure it is in the list
            if not any(d["label"] == label for d in ablation_results["Backbone"]):
                ablation_results["Backbone"].append({"label": label, "rank1": cached_r1})
            continue
            
        r1 = train_cnn_ablation(
            config, backbone=bb, loss_type='arcface_triplet_ls', 
            arcface_scale=128.0, arcface_margin=0.35, use_mixup=True, 
            use_aug=True, device=device, epochs=25
        )
        ablation_results["Backbone"].append({"label": label, "rank1": r1})
        save_current_results()

    # ── 2. ArcFace Ablation ───────────────────────────────────────────────────
    print("\n" + "="*50 + "\n2. ARCFACE MARGIN/SCALE STUDY\n" + "="*50)
    if "ArcFace" not in ablation_results:
        ablation_results["ArcFace"] = [
            {"label": "s=128, m=0.35", "rank1": cnn_anchor_r1, "baseline": True}
        ]
        save_current_results()
        
    for s, m in [(64, 0.50), (96, 0.45), (128, 0.50)]:
        label = f"s={s}, m={m:.2f}"
        cached_r1 = get_existing_result("ArcFace", label)
        if cached_r1 is not None:
            print(f"  [RESUME] Skipping ArcFace: {label} (already computed: {cached_r1*100:.2f}%)")
            if not any(d["label"] == label for d in ablation_results["ArcFace"]):
                ablation_results["ArcFace"].append({"label": label, "rank1": cached_r1})
            continue
            
        r1 = train_cnn_ablation(
            config, backbone='efficientnet_b4', loss_type='arcface_triplet_ls', 
            arcface_scale=float(s), arcface_margin=float(m), use_mixup=True, 
            use_aug=True, device=device, epochs=25
        )
        ablation_results["ArcFace"].append({"label": label, "rank1": r1})
        save_current_results()

    # ── 3. Graph Construction Ablation ────────────────────────────────────────
    print("\n" + "="*50 + "\n3. GRAPH CONSTRUCTION STUDY\n" + "="*50)
    if "Graph Construction" not in ablation_results:
        ablation_results["Graph Construction"] = [
            {"label": "DISK k=8", "rank1": gnn_anchor_r1, "baseline": True}
        ]
        save_current_results()
        
    for k in [4, 12]:
        label = f"DISK k={k}"
        cached_r1 = get_existing_result("Graph Construction", label)
        if cached_r1 is not None:
            print(f"  [RESUME] Skipping Graph Construction: {label} (already computed: {cached_r1*100:.2f}%)")
            if not any(d["label"] == label for d in ablation_results["Graph Construction"]):
                ablation_results["Graph Construction"].append({"label": label, "rank1": cached_r1})
            continue
            
        r1 = train_gnn_ablation(config, knn_k=k, device=device, epochs=30)
        ablation_results["Graph Construction"].append({"label": label, "rank1": r1})
        save_current_results()

    # ── 4. Loss Function Ablation ─────────────────────────────────────────────
    print("\n" + "="*50 + "\n4. LOSS FUNCTION STUDY\n" + "="*50)
    if "Loss Function" not in ablation_results:
        ablation_results["Loss Function"] = [
            {"label": "ArcFace+Triplet+LS", "rank1": cnn_anchor_r1, "baseline": True}
        ]
        save_current_results()
        
    loss_types = [
        ('ce', 'Cross-Entropy'),
        ('arcface', 'ArcFace Only'),
        ('arcface_triplet', 'ArcFace+Triplet')
    ]
    for lt, name in loss_types:
        cached_r1 = get_existing_result("Loss Function", name)
        if cached_r1 is not None:
            print(f"  [RESUME] Skipping Loss Function: {name} (already computed: {cached_r1*100:.2f}%)")
            if not any(d["label"] == name for d in ablation_results["Loss Function"]):
                ablation_results["Loss Function"].append({"label": name, "rank1": cached_r1})
            continue
            
        r1 = train_cnn_ablation(
            config, backbone='efficientnet_b4', loss_type=lt, 
            arcface_scale=128.0, arcface_margin=0.35, use_mixup=True, 
            use_aug=True, device=device, epochs=25
        )
        ablation_results["Loss Function"].append({"label": name, "rank1": r1})
        save_current_results()

    # ── 5. Augmentation Ablation ──────────────────────────────────────────────
    print("\n" + "="*50 + "\n5. AUGMENTATION STUDY\n" + "="*50)
    if "Augmentation" not in ablation_results:
        ablation_results["Augmentation"] = [
            {"label": "Mixup+Standard", "rank1": cnn_anchor_r1, "baseline": True}
        ]
        save_current_results()
        
    # None
    label_none = "None"
    cached_r1 = get_existing_result("Augmentation", label_none)
    if cached_r1 is not None:
        print(f"  [RESUME] Skipping Augmentation: {label_none} (already computed: {cached_r1*100:.2f}%)")
        if not any(d["label"] == label_none for d in ablation_results["Augmentation"]):
            ablation_results["Augmentation"].append({"label": label_none, "rank1": cached_r1})
    else:
        r1_none = train_cnn_ablation(
            config, backbone='efficientnet_b4', loss_type='arcface_triplet_ls',
            arcface_scale=128.0, arcface_margin=0.35, use_mixup=False,
            use_aug=False, device=device, epochs=25
        )
        ablation_results["Augmentation"].append({"label": label_none, "rank1": r1_none})
        save_current_results()
        
    # Standard
    label_std = "Standard"
    cached_r1 = get_existing_result("Augmentation", label_std)
    if cached_r1 is not None:
        print(f"  [RESUME] Skipping Augmentation: {label_std} (already computed: {cached_r1*100:.2f}%)")
        if not any(d["label"] == label_std for d in ablation_results["Augmentation"]):
            ablation_results["Augmentation"].append({"label": label_std, "rank1": cached_r1})
    else:
        r1_std = train_cnn_ablation(
            config, backbone='efficientnet_b4', loss_type='arcface_triplet_ls',
            arcface_scale=128.0, arcface_margin=0.35, use_mixup=False,
            use_aug=True, device=device, epochs=25
        )
        ablation_results["Augmentation"].append({"label": label_std, "rank1": r1_std})
        save_current_results()

    print("\n" + "="*50)
    print(f"✅ All ablation studies complete! Results saved to:\n   {out_file}")
    print("="*50)

if __name__ == '__main__':
    main()
