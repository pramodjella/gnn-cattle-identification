"""
Script: Train GNN++ (All Improvements Applied)
================================================
Uses enhanced graphs (train/val/test_graphs_v2.pt) with:
  - MobileNetV3 CNN patch features (576-d) + SIFT (256-d) + pos (2-d) = 834-d nodes
  - 4-layer Residual EdgeConv (k=16 dynamic)
  - 3-stream pooling (mean + max + attention)
  - ArcFace (margin=0.35, scale=48, triplet_weight=0.25)
  - Warmup + CosineAnnealing LR schedule
  - Full 150 epochs

Output: outputs/gnn_plus_v2/best_model.pt
         outputs/stats/gnn_plus_v2_results.json
"""

import os
import sys
import json
import time
import math
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.models.gnn_plus_v2 import CattleGNNPlusPlus
from src.training.dataset import PKSampler
from src.evaluation.metrics import BiometricMetrics


# ── LR Warmup + Cosine Schedule ──────────────────────────────────────────────

def get_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=1e-7):
    """Linear warmup then cosine annealing."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def validate_rank1(model, loader, device):
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            emb = model.get_embedding(batch)
            all_emb.append(emb.cpu())
            all_lbl.append(batch.y.cpu())
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    sim = torch.mm(emb, emb.t())
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)
    return (lbl[nn_idx] == lbl).float().mean().item()


def load_graphs_v2(graph_dir, split):
    """Load v2 graphs (with CNN features) or fall back to v1."""
    v2_path = graph_dir / f'{split}_graphs_v2.pt'
    v1_path = graph_dir / f'{split}_graphs.pt'
    if v2_path.exists():
        print(f"  [{split}] Loading enhanced graphs: {v2_path.name}")
        return torch.load(str(v2_path), weights_only=False), True
    else:
        print(f"  [{split}] WARN: v2 not found, using original: {v1_path.name}")
        return torch.load(str(v1_path), weights_only=False), False


def main():
    config = load_config()
    set_seed(config['project']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pp_cfg = config.get('gnn_plus_v2', {})
    epochs       = pp_cfg.get('epochs', 150)
    batch_size   = pp_cfg.get('batch_size', 64)
    base_lr      = pp_cfg.get('learning_rate', 3e-4)
    weight_decay = pp_cfg.get('weight_decay', 1e-4)
    warmup_eps   = pp_cfg.get('warmup_epochs', 8)
    use_amp      = pp_cfg.get('use_amp', True)
    patience     = pp_cfg.get('patience', 35)
    min_delta    = pp_cfg.get('min_delta', 0.001)
    ckpt_dir     = str(PROJECT_ROOT / pp_cfg.get('checkpoint_dir', 'outputs/gnn_plus_v2'))

    ensure_dirs(ckpt_dir, str(PROJECT_ROOT / 'outputs/stats'))
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']

    print(f"\n{'='*65}")
    print("  GNN++ TRAINING  (All Improvements Applied)")
    print(f"{'='*65}")
    print(f"  Device   : {device} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs   : {epochs} | Batch: {batch_size} | LR: {base_lr}")
    print(f"  Warmup   : {warmup_eps} epochs linear | then cosine decay")
    print(f"  ArcFace  : margin=0.35, scale=48, triplet_weight=0.25")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_graphs, using_v2 = load_graphs_v2(graph_dir, 'train')
    val_graphs, _          = load_graphs_v2(graph_dir, 'val')
    test_graphs, _         = load_graphs_v2(graph_dir, 'test')

    input_dim = train_graphs[0].x.shape[1] if train_graphs else 834
    print(f"  Node feature dim: {input_dim}-d ({'CNN+SIFT+pos' if using_v2 else 'SIFT only - run extract_patch_features.py first'})")

    labels = [g.y.item() for g in train_graphs]
    num_classes = len(set(labels))

    # PK sampler: P=16 classes × K=4 samples
    p_cls = min(16, num_classes)
    k_smp = max(2, batch_size // p_cls)
    sampler = PKSampler(labels, p=p_cls, k=k_smp)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, sampler=sampler,
                              drop_last=True, num_workers=0,
                              pin_memory=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)


    print(f"  Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")
    print(f"  Classes: {num_classes}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CattleGNNPlusPlus(config=config, input_dim=input_dim)
    model.set_num_classes(num_classes)
    model = model.to(device)
    model.summary()

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"  AMP: {use_amp} ({amp_dtype})")

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_r1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_r1': [], 'lr': [], 'epoch_time': []}

    for epoch in range(1, epochs + 1):
        # Set learning rate (warmup + cosine)
        lr = get_lr(epoch - 1, warmup_eps, epochs, base_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                result = model(batch, labels=batch.y)
                loss = result['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        val_r1   = validate_rank1(model, val_loader, device)
        t_ep     = time.time() - t0

        vram = f" | VRAM:{torch.cuda.memory_allocated(0)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
        phase = "warmup" if epoch <= warmup_eps else "cosine"
        print(f"Epoch {epoch:3d}/{epochs} [{phase}] | Loss:{avg_loss:.4f} | "
              f"R1:{val_r1:.4f} | LR:{lr:.2e} | {t_ep:.1f}s{vram}", flush=True)

        history['train_loss'].append(avg_loss)
        history['val_r1'].append(val_r1)
        history['lr'].append(lr)
        history['epoch_time'].append(t_ep)

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
                'input_dim': input_dim,
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  >> New best! R1:{best_r1:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early stopping] Epoch {epoch}")
                break

    # ── Final Evaluation ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"GNN++ Training Complete! Best R1: {best_r1:.4f} @ epoch {best_epoch}")
    print(f"{'='*65}")

    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    metrics = BiometricMetrics()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            all_emb.append(model.get_embedding(batch).cpu())
            all_lbl.append(batch.y.cpu())

    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    save_stats({
        'model': 'GNN++ (MobileNetV3 patches + 4-layer ResEdgeConv + 3-stream pool + ArcFace)',
        'best_epoch': best_epoch,
        'best_val_r1': best_r1,
        'input_dim': input_dim,
        'using_cnn_features': using_v2,
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies'].get('rank_5', 0),
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
        'history': history,
    }, str(PROJECT_ROOT / 'outputs/stats/gnn_plus_v2_results.json'))

    print(f"\nGNN++ results saved to outputs/stats/gnn_plus_v2_results.json")
    print(f"Test Rank-1: {results['identification']['rank_accuracies']['rank_1']*100:.1f}%")


if __name__ == '__main__':
    main()
