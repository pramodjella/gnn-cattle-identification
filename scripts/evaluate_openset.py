"""
Open-Set Evaluation Script
==========================
Evaluates a trained embedding model in the open-set biometric protocol:
identify enrolled ("known") animals while rejecting probes from animals that
were never enrolled ("unknown").

Identities are partitioned known/unknown by a fixed seed. The gallery is built
from the TRAIN split (enrolment templates) for known identities only; probes
come from the TEST split. Unknown probes must be rejected.

Usage:
    python scripts/evaluate_openset.py --model cnn --known-frac 0.6
    python scripts/evaluate_openset.py --model cnn --known-frac 0.5 --seed 1

Outputs: outputs/stats/openset_results.json
"""

import os
import sys
import json
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
from src.training.image_dataset import MuzzleImageDataset
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


def cnn_embeddings(model, split_json, transform, device, tta=True):
    ds = MuzzleImageDataset(split_json, transform=transform)
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
    ap.add_argument('--model', default='cnn', choices=['cnn'])
    ap.add_argument('--known-frac', type=float, default=0.6,
                    help='Fraction of identities enrolled as known.')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                    help='Partition seeds; results are aggregated mean +/- std.')
    args = ap.parse_args()

    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = config.get('preprocessing', {}).get('image_size', 256)
    transform = build_val_transform(image_size)
    preprocessed = str(PROJECT_ROOT / config['dataset']['processed_dir'])

    print("\n" + "=" * 70)
    print("  OPEN-SET EVALUATION (multi-seed)")
    print("=" * 70)

    # Embeddings computed once; partitions vary by seed.
    model = load_cnn(config, device)
    train_emb, train_lbl = cnn_embeddings(
        model, os.path.join(preprocessed, 'train_split.json'), transform, device)
    test_emb, test_lbl = cnn_embeddings(
        model, os.path.join(preprocessed, 'test_split.json'), transform, device)
    all_ids = sorted(set(int(l) for l in train_lbl))

    metric_keys = ['rank1_on_known_probes', 'openset_auc',
                   'DIR@FAR=0.01', 'DIR@FAR=0.05', 'DIR@FAR=0.1']
    per_seed = []
    for seed in args.seeds:
        ids = list(all_ids)
        np.random.RandomState(seed).shuffle(ids)
        n_known = int(round(args.known_frac * len(ids)))
        known_ids = sorted(ids[:n_known])
        res = evaluate_openset(train_emb, train_lbl, test_emb, test_lbl, known_ids)
        res['seed'] = seed
        per_seed.append(res)
        print(f"  seed {seed}: known-R1={res['rank1_on_known_probes']*100:.2f}%  "
              f"AUC={res['openset_auc']:.4f}  "
              f"DIR@1%={res['DIR@FAR=0.01']*100:.2f}%  "
              f"DIR@5%={res['DIR@FAR=0.05']*100:.2f}%")

    agg = {}
    for k in metric_keys:
        vals = np.array([r[k] for r in per_seed], dtype=float)
        agg[k] = {'mean': float(vals.mean()), 'std': float(vals.std())}

    print("\n  Aggregated over %d seeds (known_frac=%.2f):" % (len(args.seeds), args.known_frac))
    print(f"    Rank-1 (known):   {agg['rank1_on_known_probes']['mean']*100:.2f} "
          f"+/- {agg['rank1_on_known_probes']['std']*100:.2f} %")
    print(f"    Open-set AUC:     {agg['openset_auc']['mean']:.4f} "
          f"+/- {agg['openset_auc']['std']:.4f}")
    for far in ('DIR@FAR=0.01', 'DIR@FAR=0.05', 'DIR@FAR=0.1'):
        print(f"    {far}:  {agg[far]['mean']*100:.2f} +/- {agg[far]['std']*100:.2f} %")
    print("=" * 70)

    out = {
        'model': args.model,
        'known_frac': args.known_frac,
        'seeds': args.seeds,
        'aggregated': agg,
        'per_seed': per_seed,
        'note': ('Model trained closed-set on all identities; known/unknown '
                 'partition is defined by enrolment. For a strict unseen-'
                 'identity claim, retrain with unknown ids held out.'),
    }
    save_stats(out, str(PROJECT_ROOT / 'outputs/stats/openset_results.json'))
    print("  Saved -> outputs/stats/openset_results.json")


if __name__ == '__main__':
    main()
