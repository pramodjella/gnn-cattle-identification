"""
Script: Ensemble Inference — CNN + Hybrid (validation-selected blend)
====================================================================
Blends the CNN and Hybrid CNN-GNN by averaging their per-split cosine
similarity matrices, then reporting biometric metrics.

METHODOLOGY FIX (important for publication)
-------------------------------------------
The blend weight ``w`` is selected on the VALIDATION split and then applied
unchanged to the TEST split. Selecting ``w`` directly on test (as an earlier
version did) fits a hyperparameter to the test set and inflates the reported
number. For transparency we also print the *test-oracle* weight (the best ``w``
had we cheated and tuned on test) so the selection gap is visible.

Usage:
    python scripts/ensemble_inference.py

Outputs: outputs/stats/ensemble_results.json
"""

import os
import sys
import numpy as np
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


# ─────────────────────────────────────────────────────────────────────────────
# Model loading & embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_cnn_model(config, device):
    from src.models.cnn_model import CNNMuzzleModel
    ckpt_path = PROJECT_ROOT / 'outputs/cnn/best_model.pt'
    if not ckpt_path.exists():
        print("  [SKIP] CNN checkpoint not found")
        return None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ckpt.get('config', {})
    model = CNNMuzzleModel(
        num_classes=ckpt.get('num_classes', 260),
        embedding_dim=mc.get('embedding_dim', 512),
        backbone=mc.get('backbone', 'efficientnet_b4'),
        arcface_scale=mc.get('arcface_scale', 128.0),
        arcface_margin=mc.get('arcface_margin', 0.35),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  [OK] CNN loaded (epoch {ckpt.get('epoch', '?')}, "
          f"val R1={ckpt.get('val_r1', 0):.4f})")
    return model, ckpt.get('val_r1', 0)


def get_cnn_embeddings(model, loader, device, use_tta=True):
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if use_tta:
                emb = F.normalize(
                    model.get_embedding(images) +
                    model.get_embedding(torch.flip(images, dims=[-1])),
                    p=2, dim=-1)
            else:
                emb = model.get_embedding(images)
            all_emb.append(emb.float().cpu())
            all_lbl.append(labels)
    return torch.cat(all_emb), torch.cat(all_lbl)


def get_hybrid_embeddings(model, loader, device):
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, graphs, labels in loader:
            images, graphs = images.to(device), graphs.to(device)
            with autocast(device_type='cuda', dtype=amp_dtype,
                          enabled=(device.type == 'cuda')):
                out = model(images, graphs)
            all_emb.append(out['embedding'].float().cpu())
            all_lbl.append(labels)
    return torch.cat(all_emb), torch.cat(all_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Blend helpers
# ─────────────────────────────────────────────────────────────────────────────

def sim_matrix(emb):
    e = F.normalize(emb, p=2, dim=-1)
    return torch.mm(e, e.t())


def rank1_from_sim(sim, lbl):
    s = sim.clone()
    s.fill_diagonal_(-1e9)
    return (lbl[s.argmax(dim=1)] == lbl).float().mean().item()


def sweep_weight(sim_a, sim_b, lbl, num=21):
    """Return (best_w, best_r1, table) maximising Rank-1 over w in [0,1]."""
    best_w, best_r1, table = 0.5, -1.0, []
    for w in np.linspace(0, 1, num):
        r1 = rank1_from_sim(w * sim_a + (1 - w) * sim_b, lbl)
        table.append({'w_cnn': float(w), 'rank1': r1})
        if r1 > best_r1:
            best_r1, best_w = r1, float(w)
    return best_w, best_r1, table


def full_metrics_from_sim(metrics, sim, lbl):
    from sklearn.metrics import roc_curve, auc
    sim_np, lbl_np = sim.numpy(), lbl.numpy()
    cmc, ranks = metrics._compute_cmc(sim_np, lbl_np)
    gen, imp = metrics._get_score_distributions(sim_np, lbl_np)
    fpr, tpr, _ = roc_curve([1] * len(gen) + [0] * len(imp), list(gen) + list(imp))
    return {
        'rank1': float(ranks[1]), 'rank5': float(ranks[5]),
        'eer': float(metrics._compute_eer(fpr, tpr)), 'roc_auc': float(auc(fpr, tpr)),
        'cmc_curve': cmc.tolist()[:50], 'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 70)
    print("  ENSEMBLE INFERENCE — CNN + Hybrid (validation-selected blend)")
    print("=" * 70)

    image_size = config.get('preprocessing', {}).get('image_size', 256)
    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    transform = build_val_transform(image_size)
    metrics = BiometricMetrics()
    results = {}

    cnn_model, cnn_val_r1 = load_cnn_model(config, device)
    if cnn_model is None:
        print("\n  [ERROR] CNN model required. Run scripts/train_cnn.py first.")
        return

    # CNN embeddings on val + test
    cnn_emb, cnn_lbl = {}, {}
    for split in ['val', 'test']:
        js = os.path.join(preprocessed_dir, f'{split}_split.json')
        if not os.path.exists(js):
            print(f"  [WARN] missing {split}_split.json")
            continue
        ds = MuzzleImageDataset(js, transform=transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        cnn_emb[split], cnn_lbl[split] = get_cnn_embeddings(cnn_model, loader, device, use_tta=True)

    cnn_test_r1 = rank1_from_sim(sim_matrix(cnn_emb['test']), cnn_lbl['test'])
    print(f"\n  CNN (TTA) test Rank-1: {cnn_test_r1*100:.2f}%")
    results['cnn_tta'] = {'rank1': cnn_test_r1}
    cnn_full = metrics.compute_all_metrics(cnn_emb['test'], cnn_lbl['test'])
    results['cnn_full'] = {
        'rank1': cnn_full['identification']['rank_accuracies']['rank_1'],
        'rank5': cnn_full['identification']['rank_accuracies']['rank_5'],
        'eer': cnn_full['verification']['eer'],
        'roc_auc': cnn_full['verification']['roc_auc'],
    }

    # ── Hybrid + ensemble (if available) ──────────────────────────────────────
    hybrid_ckpt_path = PROJECT_ROOT / 'outputs/hybrid/best_model.pt'
    if not hybrid_ckpt_path.exists():
        print("\n  [INFO] Hybrid checkpoint not found; reporting CNN only.")
        save_stats(results, str(PROJECT_ROOT / 'outputs/stats/ensemble_results.json'))
        return

    try:
        from src.models.hybrid_model import HybridCNNGNN
        from src.training.image_dataset import create_hybrid_loaders

        hy_ckpt = torch.load(hybrid_ckpt_path, map_location=device, weights_only=False)
        hybrid = HybridCNNGNN(num_classes=hy_ckpt.get('num_classes', 260),
                              config=config, pretrained=False).to(device)
        hybrid.load_state_dict(hy_ckpt['model_state_dict'])
        hybrid.eval()
        hy_val_r1 = hy_ckpt.get('val_r1', 0)
        print(f"  [OK] Hybrid loaded (epoch {hy_ckpt.get('epoch', '?')}, val R1={hy_val_r1:.4f})")

        loaders = create_hybrid_loaders(preprocessed_dir,
                                        str(PROJECT_ROOT / config['dataset']['graph_dir']),
                                        config)
        hy_emb, hy_lbl = {}, {}
        for split in ['val', 'test']:
            if split in loaders:
                hy_emb[split], hy_lbl[split] = get_hybrid_embeddings(hybrid, loaders[split], device)

        # Alignment check: CNN and Hybrid must enumerate the same probes per split.
        for split in ['val', 'test']:
            if split not in hy_emb or hy_lbl[split].shape != cnn_lbl[split].shape \
               or not (hy_lbl[split] == cnn_lbl[split]).all():
                print(f"  [WARN] label order mismatch on {split}; cannot ensemble.")
                save_stats(results, str(PROJECT_ROOT / 'outputs/stats/ensemble_results.json'))
                return

        # Per-split similarity matrices.
        sim = {s: {'cnn': sim_matrix(cnn_emb[s]), 'hy': sim_matrix(hy_emb[s])}
               for s in ['val', 'test']}

        hy_test_r1 = rank1_from_sim(sim['test']['hy'], cnn_lbl['test'])
        results['hybrid'] = {'rank1': hy_test_r1}
        print(f"  Hybrid test Rank-1: {hy_test_r1*100:.2f}%")

        # 1) Select w on VALIDATION.
        val_w, val_r1, val_table = sweep_weight(sim['val']['cnn'], sim['val']['hy'], cnn_lbl['val'])
        print(f"\n  Validation-selected weight: CNN={val_w:.2f}, Hybrid={1-val_w:.2f} "
              f"(val Rank-1={val_r1*100:.2f}%)")

        # 2) Apply that w to TEST (the honest number).
        test_sim = val_w * sim['test']['cnn'] + (1 - val_w) * sim['test']['hy']
        ens = full_metrics_from_sim(metrics, test_sim, cnn_lbl['test'])
        print(f"  >> Ensemble TEST Rank-1 @ val-selected w: {ens['rank1']*100:.2f}%  "
              f"(EER={ens['eer']*100:.2f}%, AUC={ens['roc_auc']:.4f})")

        # 3) Test-oracle w (transparency only — NOT the reported result).
        oracle_w, oracle_r1, _ = sweep_weight(sim['test']['cnn'], sim['test']['hy'], cnn_lbl['test'])
        print(f"  (Test-oracle w={oracle_w:.2f} would give {oracle_r1*100:.2f}% — "
              f"selection gap {abs(oracle_r1-ens['rank1'])*100:.2f} pts)")

        results.update({
            'model': 'Ensemble (CNN TTA + Hybrid CNN-GNN) [val-selected]',
            'selection': {
                'val_selected_w_cnn': val_w,
                'val_rank1': val_r1,
                'test_oracle_w_cnn': oracle_w,
                'test_oracle_rank1': oracle_r1,
                'selection_gap': abs(oracle_r1 - ens['rank1']),
                'val_sweep': val_table,
            },
            'test_rank1': ens['rank1'],
            'test_rank5': ens['rank5'],
            'eer': ens['eer'],
            'roc_auc': ens['roc_auc'],
            'cmc_curve': ens['cmc_curve'],
            'fpr': ens['fpr'],
            'tpr': ens['tpr'],
            'best_val_r1': max(cnn_val_r1, hy_val_r1),
        })
        results['ensemble_val_selected'] = {
            'rank1': ens['rank1'], 'rank5': ens['rank5'],
            'eer': ens['eer'], 'roc_auc': ens['roc_auc'],
        }
    except Exception as e:
        import traceback
        print(f"  [WARN] Hybrid ensemble failed: {e}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("  FINAL ENSEMBLE RESULTS")
    print("=" * 70)
    for k, v in results.items():
        if isinstance(v, dict) and 'rank1' in v:
            line = f"  {k}: Rank-1={v['rank1']*100:.2f}%"
            if 'eer' in v:
                line += f" | EER={v['eer']*100:.2f}%"
            print(line)

    save_stats(results, str(PROJECT_ROOT / 'outputs/stats/ensemble_results.json'))
    print("\n  Saved -> outputs/stats/ensemble_results.json")


if __name__ == '__main__':
    main()
