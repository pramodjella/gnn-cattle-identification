"""
Cross-Dataset Transfer Evaluation
=================================
Evaluates a model trained on the primary dataset on a *different* muzzle
dataset, WITHOUT any fine-tuning, to measure generalisation. This is the
headline experiment for a top-tier submission's generalisation claim.

Reports, on the external dataset:
  * Closed-set Rank-1 / Rank-5 / EER / ROC-AUC (self-similarity protocol).
  * Open-set AUC and DIR@FAR (enrol half the identities, reject the rest).

The external dataset must be organised as one folder per animal
(<root>/<animal_id>/<image>). No identity overlap with training is assumed.

Usage:
    python scripts/evaluate_cross_dataset.py --data-root path/to/other_dataset
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats
from src.training.augmentation import build_val_transform
from src.training.external_dataset import ExternalMuzzleImageDataset
from src.evaluation.metrics import BiometricMetrics
from src.evaluation.openset import evaluate_openset


def load_cnn(config, device):
    from src.models.cnn_model import CNNMuzzleModel
    ckpt = torch.load(PROJECT_ROOT / 'outputs/cnn/best_model.pt',
                      map_location=device, weights_only=False)
    mc = ckpt.get('config', {})
    model = CNNMuzzleModel(
        num_classes=ckpt.get('num_classes', 260),
        embedding_dim=mc.get('embedding_dim', 512),
        backbone=mc.get('backbone', 'efficientnet_b4'),
        arcface_scale=mc.get('arcface_scale', 128.0),
        arcface_margin=mc.get('arcface_margin', 0.35),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.eval()


def adapt_bn(model, ds, device, passes=1):
    """AdaBN: recompute BatchNorm running stats on the (unlabelled) target
    domain, countering feature-distribution shift. No labels, no backprop.

    Reference: Li et al. (2017), "Revisiting Batch Normalization for Practical
    Domain Adaptation".
    """
    import torch.nn as nn
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None   # cumulative moving average over the target pass
        m.train()
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    with torch.no_grad():
        for _ in range(passes):
            for images, _ in loader:
                model.get_embedding(images.to(device))
    model.eval()
    print(f"  AdaBN: adapted {len(bns)} BatchNorm layers on target ({passes} pass)")
    return model


def snorm_matrix(sim):
    """Adaptive symmetric score normalisation (S-norm).

    Recalibrates each pair score by both endpoints' cohort (row) statistics, so
    a single operating threshold transfers across probes/domains. Unsupervised
    (cohort = all other samples). Reference: Cumani et al.; Matejka et al.
    """
    S = sim.copy().astype(np.float64)
    np.fill_diagonal(S, np.nan)                       # exclude self from cohort
    mu = np.nanmean(S, axis=1, keepdims=True)
    sd = np.nanstd(S, axis=1, keepdims=True) + 1e-8
    Z = 0.5 * ((S - mu) / sd + (S - mu.T) / sd.T)     # symmetric
    np.fill_diagonal(Z, -np.inf)
    return Z


def metrics_from_sim(sim_np, lbl_np, metrics):
    """Closed-set + verification metrics directly from a similarity matrix."""
    from sklearn.metrics import roc_curve, auc
    cmc, ranks = metrics._compute_cmc(sim_np, lbl_np)
    gen, imp = metrics._get_score_distributions(sim_np, lbl_np)
    fpr, tpr, _ = roc_curve([1] * len(gen) + [0] * len(imp), list(gen) + list(imp))
    return {'rank1': float(ranks[1]), 'rank5': float(ranks.get(5, 0)),
            'eer': float(metrics._compute_eer(fpr, tpr)), 'roc_auc': float(auc(fpr, tpr))}


def embed(model, ds, device, tta=True):
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    embs, lbls = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if tta:
                e = F.normalize(model.get_embedding(images) +
                                model.get_embedding(torch.flip(images, dims=[-1])),
                                p=2, dim=-1)
            else:
                e = model.get_embedding(images)
            embs.append(e.float().cpu()); lbls.append(labels)
    return torch.cat(embs), torch.cat(lbls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True,
                    help='External dataset root (folder per animal).')
    ap.add_argument('--known-frac', type=float, default=0.5)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--preprocess', choices=['clahe', 'none'], default='clahe',
                    help="Match training preprocessing. 'clahe' applies the same "
                         "CLAHE the CNN was trained on (avoids domain-shift collapse).")
    ap.add_argument('--min-images', type=int, default=1,
                    help="Drop identities with fewer images. Use 2 for a fair "
                         "closed-set number (a single-image identity has no "
                         "genuine gallery mate and is a forced miss).")
    ap.add_argument('--name', default=None,
                    help="Optional tag; results saved to cross_dataset_<name>.json.")
    ap.add_argument('--adabn', action='store_true',
                    help="Test-time BatchNorm adaptation on the target domain "
                         "(no labels/retraining); counters feature-distribution shift.")
    ap.add_argument('--score-norm', choices=['none', 'snorm'], default='none',
                    help="Recalibrate scores (S-norm) so the verification "
                         "threshold transfers across domains; fixes EER gap.")
    args = ap.parse_args()

    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = build_val_transform(config.get('preprocessing', {}).get('image_size', 256))

    print("\n" + "=" * 70)
    print("  CROSS-DATASET TRANSFER EVALUATION")
    print("=" * 70)

    model = load_cnn(config, device)
    clahe_cfg = config.get('preprocessing', {}).get('clahe', {}) if args.preprocess == 'clahe' else None
    ds = ExternalMuzzleImageDataset(args.data_root, transform=transform, clahe=clahe_cfg,
                                    min_images_per_animal=args.min_images)
    print(f"  Preprocessing: {args.preprocess}")
    if args.adabn:
        adapt_bn(model, ds, device)
    emb, lbl = embed(model, ds, device)

    # Closed-set self-similarity metrics.
    metrics = BiometricMetrics()
    if args.score_norm == 'snorm':
        sim = (F.normalize(emb, p=2, dim=-1) @ F.normalize(emb, p=2, dim=-1).t()).numpy()
        m = metrics_from_sim(snorm_matrix(sim), lbl.numpy(), metrics)
        cs = {'rank_1_accuracy': m['rank1'], 'rank_5_accuracy': m['rank5'],
              'eer': m['eer'], 'roc_auc': m['roc_auc']}
        print(f"  Score norm: S-norm")
    else:
        cs = metrics.compute_all_metrics(emb, lbl)['summary']
    print(f"\n  Closed-set (self-similarity):")
    print(f"    Rank-1 {cs['rank_1_accuracy']*100:.2f}%  Rank-5 {cs['rank_5_accuracy']*100:.2f}%  "
          f"EER {cs['eer']*100:.2f}%  AUC {cs['roc_auc']:.4f}")

    # Open-set: enrol half the identities, reject the rest (gallery == probe set
    # split by identity; single-image identities simply act as probes).
    all_ids = sorted(set(int(l) for l in lbl))
    per_seed = []
    for seed in args.seeds:
        ids = list(all_ids); np.random.RandomState(seed).shuffle(ids)
        known = sorted(ids[:int(round(args.known_frac * len(ids)))])
        r = evaluate_openset(emb, lbl, emb, lbl, known)
        per_seed.append(r)
    def agg(k):
        v = np.array([r[k] for r in per_seed], float)
        return {'mean': float(v.mean()), 'std': float(v.std())}
    os_auc = agg('openset_auc'); dir1 = agg('DIR@FAR=0.01')
    print(f"  Open-set (mean over {len(args.seeds)} seeds):")
    print(f"    AUC {os_auc['mean']:.4f}+/-{os_auc['std']:.4f}  "
          f"DIR@FAR=1% {dir1['mean']*100:.2f}+/-{dir1['std']*100:.2f}%")
    print("=" * 70)

    out = {
        'data_root': args.data_root,
        'num_images': len(ds), 'num_identities': len(all_ids),
        'closed_set': {
            'rank1': cs['rank_1_accuracy'], 'rank5': cs['rank_5_accuracy'],
            'eer': cs['eer'], 'roc_auc': cs['roc_auc'],
        },
        'open_set': {'openset_auc': os_auc, 'DIR@FAR=0.01': dir1,
                     'DIR@FAR=0.05': agg('DIR@FAR=0.05')},
        'note': 'Zero-shot transfer: no fine-tuning on the external dataset.',
    }
    out['min_images'] = args.min_images
    fname = f"cross_dataset_{args.name}.json" if args.name else "cross_dataset_results.json"
    save_stats(out, str(PROJECT_ROOT / 'outputs/stats' / fname))
    print(f"  Saved -> outputs/stats/{fname}")


if __name__ == '__main__':
    main()
