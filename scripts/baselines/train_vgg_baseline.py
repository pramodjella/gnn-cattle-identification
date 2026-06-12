"""
Script: Train VGG-16 Baseline (Bello et al. 2020)
===================================================
Re-implementation of the baseline from Bello et al. (2020):
  - Backbone: VGG-16 (ImageNet pre-trained)
  - Head: Fully connected classifier trained with Cross-Entropy Loss
  - At test time, features from the penultimate layer (fc2, 4096-dim projected to 512-dim)
    are extracted and matched using Cosine Similarity.
  - Saves results to outputs/stats/vgg_baseline_results.json
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
from torchvision.models import vgg16, VGG16_Weights

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.training.image_dataset import create_image_loaders
from src.evaluation.metrics import BiometricMetrics

class VGGBiometricModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, pretrained=True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vgg16(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Penultimate embedding head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Classification projection (used only for Cross-Entropy training)
        self.output_layer = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.classifier(x)
        logits = self.output_layer(emb)
        return logits, F.normalize(emb, p=2, dim=-1)

    def get_embedding(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.classifier(x)
        return F.normalize(emb, p=2, dim=-1)

def main():
    config = load_config()
    set_seed(config['project']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt_dir = PROJECT_ROOT / 'outputs' / 'vgg_baseline'
    ensure_dirs(str(ckpt_dir), str(PROJECT_ROOT / 'outputs/stats'))

    print(f"\n{'='*70}")
    print("  TRAINING VGG-16 BASELINE (Bello et al. 2020)")
    print(f"{'='*70}")
    print(f"  Device: {device}")

    # Load data
    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    loaders = create_image_loaders(preprocessed_dir, config)
    
    with open(os.path.join(preprocessed_dir, 'train_split.json')) as f:
        train_data = json.load(f)
    num_classes = len(set(item.get('animal_id', item.get('label', '')) for item in train_data))
    print(f"  Classes: {num_classes} | Train samples: {len(loaders['train'].dataset)}")

    model = VGGBiometricModel(num_classes=num_classes, embedding_dim=512, pretrained=True).to(device)
    
    # Differential LR: backbone slow, classifier fast
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        {'params': model.output_layer.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=1e-7)
    
    use_amp = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = GradScaler('cuda', enabled=use_scaler)

    epochs = 15
    best_r1 = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in loaders['train']:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(images)
                loss = criterion(logits, labels)
                
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
        avg_loss = total_loss / max(num_batches, 1)
        
        # Validation evaluation
        model.eval()
        all_emb, all_lbl = [], []
        with torch.no_grad():
            for images, labels in loaders['val']:
                images = images.to(device)
                emb = model.get_embedding(images)
                all_emb.append(emb.cpu())
                all_lbl.append(labels.cpu())
        emb = torch.cat(all_emb)
        lbl = torch.cat(all_lbl)
        sim = torch.mm(emb, emb.t())
        sim.fill_diagonal_(-1e9)
        val_r1 = (lbl[sim.argmax(dim=1)] == lbl).float().mean().item()
        
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Val R1: {val_r1*100:.2f}% | {time.time()-t0:.1f}s")
        
        if val_r1 > best_r1:
            best_r1 = val_r1
            torch.save(model.state_dict(), ckpt_dir / 'best_model.pt')

    # Load best model for testing
    model.load_state_dict(torch.load(ckpt_dir / 'best_model.pt'))
    model.eval()
    
    metrics = BiometricMetrics()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, labels in loaders['test']:
            images = images.to(device)
            emb = model.get_embedding(images)
            all_emb.append(emb.cpu())
            all_lbl.append(labels.cpu())
            
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)
    
    save_stats({
        'model': 'VGG-16 Baseline (Bello et al. 2020)',
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies'].get('rank_5', 0.0),
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
    }, str(PROJECT_ROOT / 'outputs/stats/vgg_baseline_results.json'))
    
    print("\n✅ VGG-16 Baseline results saved to outputs/stats/vgg_baseline_results.json")

if __name__ == '__main__':
    main()
