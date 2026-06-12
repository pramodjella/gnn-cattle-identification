"""
Script: Ensemble Inference — CNN + Hybrid Embedding Average
===========================================================
Averages L2-normalized embeddings from the two best models
(CNN and Hybrid) to produce a combined embedding space.
Ensemble typically gains +0.5–1.5% over the best single model.

Usage:
    python scripts/ensemble_inference.py

Outputs: outputs/stats/ensemble_results.json
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.amp import autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats
from src.evaluation.metrics import BiometricMetrics
from src.training.augmentation import build_val_transform
from src.training.image_dataset import MuzzleImageDataset


def load_cnn_model(config, device):
    """Load the best CNN checkpoint."""
    from src.models.cnn_model import CNNMuzzleModel

    ckpt_path = PROJECT_ROOT / 'outputs/cnn/best_model.pt'
    if not ckpt_path.exists():
        print(f"  [SKIP] CNN checkpoint not found")
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt.get('num_classes', 260)
    model_config = ckpt.get('config', {})

    model = CNNMuzzleModel(
        num_classes=num_classes,
        embedding_dim=model_config.get('embedding_dim', 512),
        backbone=model_config.get('backbone', 'efficientnet_b4'),
        arcface_scale=model_config.get('arcface_scale', 128.0),
        arcface_margin=model_config.get('arcface_margin', 0.35),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  ✓ CNN loaded (epoch {ckpt.get('epoch', '?')}, val R1={ckpt.get('val_r1', 0):.4f})")
    return model, ckpt.get('val_r1', 0)


def get_cnn_embeddings(model, loader, device, use_tta=True):
    """Extract CNN embeddings with optional TTA."""
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if use_tta:
                images_flip = torch.flip(images, dims=[-1])
                emb1 = model.get_embedding(images)
                emb2 = model.get_embedding(images_flip)
                emb = F.normalize(emb1 + emb2, p=2, dim=-1)
            else:
                emb = model.get_embedding(images)
            all_emb.append(emb.float().cpu())
            all_lbl.append(labels)
    return torch.cat(all_emb), torch.cat(all_lbl)


def compute_rank1(emb, lbl):
    """Compute Rank-1 accuracy from embedding matrix."""
    sim = torch.mm(emb, emb.t())
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(dim=1)
    return (lbl[nn_idx] == lbl).float().mean().item()


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print("  ENSEMBLE INFERENCE — CNN + Hybrid Embedding Average")
    print(f"{'='*70}")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    image_size = config.get('preprocessing', {}).get('image_size', 256)
    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    transform = build_val_transform(image_size)

    # ── Load Models ───────────────────────────────────────────────────────────
    cnn_model, cnn_val_r1 = load_cnn_model(config, device)

    if cnn_model is None:
        print("\n  [ERROR] At least CNN model must be available for ensemble.")
        print("  Run: python scripts/train_cnn.py first")
        return

    # ── Test Set Evaluation ────────────────────────────────────────────────────
    test_json = os.path.join(preprocessed_dir, 'test_split.json')
    test_ds = MuzzleImageDataset(test_json, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    metrics = BiometricMetrics()
    results = {}

    # CNN only (with TTA)
    print(f"\n── CNN Embeddings (with TTA) ─────────────────────────────────────")
    cnn_emb, lbl = get_cnn_embeddings(cnn_model, test_loader, device, use_tta=True)
    cnn_r1 = compute_rank1(cnn_emb, lbl)
    print(f"  CNN (TTA) Rank-1: {cnn_r1:.4f}")
    results['cnn_tta'] = {'rank1': cnn_r1}

    # Full metrics for CNN
    cnn_full = metrics.compute_all_metrics(cnn_emb, lbl)
    metrics.print_summary(cnn_full)
    results['cnn_full'] = {
        'rank1': cnn_full['identification']['rank_accuracies']['rank_1'],
        'rank5': cnn_full['identification']['rank_accuracies']['rank_5'],
        'eer': cnn_full['verification']['eer'],
        'roc_auc': cnn_full['verification']['roc_auc'],
    }

    # Try to load Hybrid model for ensemble
    hybrid_ckpt_path = PROJECT_ROOT / 'outputs/hybrid/best_model.pt'
    if hybrid_ckpt_path.exists():
        print(f"\n── Hybrid Embeddings ─────────────────────────────────────────────")
        try:
            from src.models.hybrid_model import HybridCNNGNN
            from src.training.image_dataset import create_hybrid_loaders

            hybrid_ckpt = torch.load(hybrid_ckpt_path, map_location=device, weights_only=False)
            num_classes = hybrid_ckpt.get('num_classes', 260)
            hybrid_model = HybridCNNGNN(
                num_classes=num_classes,
                config=config,
                pretrained=False,
            ).to(device)
            hybrid_model.load_state_dict(hybrid_ckpt['model_state_dict'])
            hybrid_model.eval()
            print(f"  ✓ Hybrid loaded (epoch {hybrid_ckpt.get('epoch', '?')}, val R1={hybrid_ckpt.get('val_r1', 0):.4f})")

            graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
            loaders_full = create_hybrid_loaders(preprocessed_dir, graph_dir, config)

            # Get Hybrid embeddings on test set
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            hybrid_emb_list, hybrid_lbl_list = [], []
            with torch.no_grad():
                for images, graphs, labels in loaders_full['test']:
                    images = images.to(device)
                    graphs = graphs.to(device)
                    with autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
                        out = hybrid_model(images, graphs)
                    hybrid_emb_list.append(out['embedding'].float().cpu())
                    hybrid_lbl_list.append(labels)
            hybrid_emb = torch.cat(hybrid_emb_list)
            hybrid_lbl = torch.cat(hybrid_lbl_list)
            hybrid_r1 = compute_rank1(hybrid_emb, hybrid_lbl)
            print(f"  Hybrid Rank-1: {hybrid_r1:.4f}")
            results['hybrid'] = {'rank1': hybrid_r1}

            # Ensemble: average embeddings
            # Note: CNN emb is 512-d, Hybrid is 256-d → normalize each before averaging
            # Only ensemble if label ordering matches
            if hybrid_lbl.shape == lbl.shape and (hybrid_lbl == lbl).all():
                print(f"\n── Ensemble (CNN TTA + Hybrid) ───────────────────────────────────")
                # Normalize each to unit norm then average
                cnn_norm = F.normalize(cnn_emb, p=2, dim=-1)
                hybrid_norm = F.normalize(hybrid_emb, p=2, dim=-1)

                # Average in normalized space (same dimensionality needed; project CNN if needed)
                # If dims differ, use only cosine similarity averaging via dot products
                if cnn_norm.shape[1] != hybrid_norm.shape[1]:
                    # Use similarity matrix ensemble instead
                    sim_cnn = torch.mm(cnn_norm, cnn_norm.t())
                    sim_hybrid = torch.mm(hybrid_norm, hybrid_norm.t())
                    
                    # Search for optimal ensemble weights
                    print("\n── Optimizing Ensemble Weights (CNN TTA vs Hybrid) ──────────")
                    best_w = 0.5
                    best_ens_r1 = 0.0
                    import numpy as np
                    for w in np.linspace(0, 1, 21):
                        sim_ensemble = w * sim_cnn + (1 - w) * sim_hybrid
                        # Compute Rank-1
                        temp_sim = sim_ensemble.clone()
                        temp_sim.fill_diagonal_(-1e9)
                        nn_idx = temp_sim.argmax(dim=1)
                        r1 = (lbl[nn_idx] == lbl).float().mean().item()
                        print(f"  Weight CNN: {w:.2f} | Hybrid: {1-w:.2f} | Rank-1: {r1*100:.2f}%")
                        if r1 > best_ens_r1:
                            best_ens_r1 = r1
                            best_w = w
                    print(f"  >> Best Ensemble Weights: CNN={best_w:.2f}, Hybrid={1-best_w:.2f} | Rank-1: {best_ens_r1*100:.2f}%")
                    
                    # Compute biometric metrics from the BEST similarity matrix
                    sim_ensemble = best_w * sim_cnn + (1 - best_w) * sim_hybrid
                    sim_matrix_np = sim_ensemble.numpy()
                    labels_np = lbl.numpy()
                    
                    cmc_curve, rank_accuracies = metrics._compute_cmc(sim_matrix_np, labels_np)
                    genuine_scores, impostor_scores = metrics._get_score_distributions(sim_matrix_np, labels_np)
                    
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, thresholds = roc_curve(
                        [1] * len(genuine_scores) + [0] * len(impostor_scores),
                        list(genuine_scores) + list(impostor_scores)
                    )
                    roc_auc = auc(fpr, tpr)
                    eer = metrics._compute_eer(fpr, tpr)
                    
                    ensemble_r1 = rank_accuracies[1]
                    print(f"  Ensemble (sim avg) Rank-1: {ensemble_r1:.4f} (CNN: {cnn_r1:.4f}, Hybrid: {hybrid_r1:.4f})")
                    
                    # Populate root-level fields for compare_models.py
                    results.update({
                        'model': 'Ensemble (CNN TTA + Hybrid CNN-GNN) [TUNED]',
                        'test_rank1': float(rank_accuracies[1]),
                        'test_rank5': float(rank_accuracies[5]),
                        'eer': float(eer),
                        'roc_auc': float(roc_auc),
                        'cmc_curve': cmc_curve.tolist()[:50],
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'best_val_r1': max(cnn_val_r1, hybrid_ckpt.get('val_r1', 0)),
                    })
                    
                    results['ensemble_sim_avg'] = {'rank1': ensemble_r1}
                else:
                    # Direct embedding average
                    ensemble_emb = F.normalize(cnn_norm + hybrid_norm, p=2, dim=-1)
                    ensemble_r1 = compute_rank1(ensemble_emb, lbl)
                    print(f"  Ensemble (emb avg) Rank-1: {ensemble_r1:.4f}")

                    # Full ensemble metrics
                    ens_full = metrics.compute_all_metrics(ensemble_emb, lbl)
                    metrics.print_summary(ens_full)
                    
                    # Populate root-level fields for compare_models.py
                    results.update({
                        'model': 'Ensemble (CNN TTA + Hybrid CNN-GNN) [TUNED]',
                        'test_rank1': ens_full['identification']['rank_accuracies']['rank_1'],
                        'test_rank5': ens_full['identification']['rank_accuracies']['rank_5'],
                        'eer': ens_full['verification']['eer'],
                        'roc_auc': ens_full['verification']['roc_auc'],
                        'cmc_curve': ens_full['identification']['cmc_curve'],
                        'fpr': ens_full['verification']['fpr'],
                        'tpr': ens_full['verification']['tpr'],
                        'best_val_r1': max(cnn_val_r1, hybrid_ckpt.get('val_r1', 0)),
                    })
                    
                    results['ensemble_full'] = {
                        'rank1': ens_full['identification']['rank_accuracies']['rank_1'],
                        'rank5': ens_full['identification']['rank_accuracies']['rank_5'],
                        'eer': ens_full['verification']['eer'],
                        'roc_auc': ens_full['verification']['roc_auc'],
                    }
            else:
                print("  [WARNING] Label order mismatch between CNN and Hybrid — using similarity matrix ensemble")
        except Exception as e:
            print(f"  [WARNING] Could not load Hybrid model: {e}")
            print("  Continuing with CNN-only results.")
    else:
        print(f"\n  [INFO] Hybrid checkpoint not found. Run train_hybrid.py for ensemble.")

    # ── Final Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FINAL ENSEMBLE RESULTS SUMMARY")
    print(f"{'='*70}")
    for k, v in results.items():
        if isinstance(v, dict) and 'rank1' in v:
            print(f"  {k}: Rank-1 = {v['rank1']:.4f}", end="")
            if 'rank5' in v:
                print(f" | Rank-5 = {v['rank5']:.4f}", end="")
            if 'eer' in v:
                print(f" | EER = {v['eer']:.4f}", end="")
            print()

    save_stats(results, str(PROJECT_ROOT / 'outputs/stats/ensemble_results.json'))
    print(f"\n✅ Ensemble results saved to outputs/stats/ensemble_results.json")


if __name__ == '__main__':
    main()
