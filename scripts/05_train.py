"""
Script 05: Train CattleGNN Model
===================================
Trains the CattleGNN model using triplet loss with online hard mining.
Saves checkpoints, training history, and statistics.

Input:  data/graphs/ (pre-built PyG graphs)
Output: outputs/checkpoints/ (model weights)
Stats:  outputs/stats/training_stats.json
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, setup_logging, set_seed, Timer, count_parameters
from src.models.gnn_model import CattleGNN
from src.models.losses import TripletLossWithMining, CombinedLoss
from src.training.dataset import create_data_loaders
from src.training.trainer import Trainer


def main():
    print("=" * 70)
    print("PHASE 5-6: Model Training")
    print("=" * 70)
    
    # Load config
    config = load_config()
    set_seed(config['project']['seed'])
    logger = setup_logging(config['outputs']['log_dir'], "05_train")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # ── RTX 5070 / Blackwell GPU optimisations ────────────────────────
    if device.type == 'cuda':
        # Allow TF32 on Ampere+ (Blackwell supports it natively, ~10% speedup)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # benchmark=True: cuDNN auto-tunes kernel for repeated input sizes
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] GPU: {props.name}")
        print(f"[INFO] VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"[INFO] CUDA Capability: sm_{props.major}{props.minor}")
    
    # Paths
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    stats_dir = str(PROJECT_ROOT / config['outputs']['stats_dir'])
    checkpoint_dir = str(PROJECT_ROOT / config['training']['checkpoint_dir'])
    ensure_dirs(stats_dir, checkpoint_dir)
    
    # Update checkpoint dir in config
    config['training']['checkpoint_dir'] = checkpoint_dir
    
    # Create data loaders
    print("\n--- Creating Data Loaders ---")
    loaders = create_data_loaders(graph_dir, config, augment_train=True)
    
    if 'train' not in loaders:
        print("[ERROR] No training data found. Run scripts 01-04 first.")
        return
    
    # Load label mapping to get number of classes
    label_map_path = os.path.join(graph_dir, "label_mapping.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_mapping = json.load(f)
        num_classes = len(label_mapping)
    else:
        # Infer from data
        all_labels = set()
        for batch in loaders['train']:
            all_labels.update(batch.y.tolist())
        num_classes = len(all_labels)
    
    print(f"[INFO] Number of classes: {num_classes}")
    
    # Initialize model
    print("\n--- Initializing Model ---")
    model = CattleGNN(config=config)
    model.set_num_classes(num_classes)
    model_summary = model.summary()
    
    print(f"[INFO] Total parameters: {count_parameters(model):,}")
    
    # Initialize loss function
    triplet_config = config['training'].get('triplet', {})
    loss_fn = CombinedLoss(
        margin=triplet_config.get('margin', 0.5),
        mining_type=triplet_config.get('mining_type', 'hard'),
        ce_weight=0.5,
    )
    
    # Initialize trainer
    trainer = Trainer(model, loss_fn, config, device=device)
    
    # Train
    print("\n--- Starting Training ---")
    with Timer("Training") as timer:
        history = trainer.train(
            train_loader=loaders['train'],
            val_loader=loaders.get('val'),
        )
    
    # Save training statistics
    training_stats = {
        'model_summary': model_summary,
        'num_classes': num_classes,
        'device': str(device),
        'training_time_seconds': timer.elapsed,
        'best_val_accuracy': trainer.best_val_acc,
        'best_val_loss': trainer.best_val_loss,
        'total_epochs': trainer.epoch,
        'final_history': {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    
    stats_path = os.path.join(stats_dir, "training_stats.json")
    save_stats(training_stats, stats_path)
    
    print(f"\n[INFO] Training statistics saved to {stats_path}")
    print(f"\n[SUCCESS] [OK] Phase 5-6 complete!")
    print(f"  Best model: {checkpoint_dir}/best_model.pt")
    print(f"  Training stats: {stats_path}")
    
    return training_stats


if __name__ == "__main__":
    main()
