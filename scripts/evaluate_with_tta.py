"""
Script: Evaluate All Models with Test-Time Augmentation (TTA)
==============================================================
Re-evaluates all saved model checkpoints with 2-view TTA
(original + horizontal flip) to get free accuracy boost.

Usage:
    python scripts/evaluate_with_tta.py

Outputs results to outputs/stats/<model>_tta_results.json
"""

import os
import sys
import json
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


def extract_embeddings_tta(model, loader, device, use_amp=True, model_type='cnn'):
    """
    Extract embeddings with 2-view TTA: original + horizontal flip.
    Averages L2-normalized embeddings from both views.
    """
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    model.eval()
    all_emb, all_lbl = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            images_flip = torch.flip(images, dims=[-1])  # horizontal flip

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                if model_type == 'cnn':
                    emb1 = model.get_embedding(images)
                    emb2 = model.get_embedding(images_flip)
                else:
                    emb1 = model(images)['embedding'] if hasattr(model(images), '__getitem__') else model.get_embedding(images)
                    emb2 = model(images_flip)['embedding'] if hasattr(model(images_flip), '__getitem__') else model.get_embedding(images_flip)

            # Average embeddings after normalizing each
            emb = F.normalize(emb1.float() + emb2.float(), p=2, dim=-1)
            all_emb.append(emb.cpu())
            all_lbl.append(labels)

    return torch.cat(all_emb), torch.cat(all_lbl)


def evaluate_cnn_tta(config, device):
    """Evaluate CNN model with TTA."""
    from src.models.cnn_model import CNNMuzzleModel

    ckpt_path = PROJECT_ROOT / 'outputs/cnn/best_model.pt'
    if not ckpt_path.exists():
        print(f"  [SKIP] CNN checkpoint not found at {ckpt_path}")
        return None

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
    print(f"  CNN: Loaded from epoch {ckpt.get('epoch', '?')} | val R1={ckpt.get('val_r1', 0):.4f}")

    image_size = config.get('preprocessing', {}).get('image_size', 256)
    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    transform = build_val_transform(image_size)

    results = {}
    for split in ['val', 'test']:
        split_json = os.path.join(preprocessed_dir, f'{split}_split.json')
        if not os.path.exists(split_json):
            continue
        ds = MuzzleImageDataset(split_json, transform=transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        # Without TTA
        model.eval()
        all_emb, all_lbl = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                emb = model.get_embedding(images)
                all_emb.append(emb.cpu())
                all_lbl.append(labels)
        emb_no_tta = torch.cat(all_emb)
        lbl = torch.cat(all_lbl)
        sim = torch.mm(emb_no_tta, emb_no_tta.t())
        sim.fill_diagonal_(-1e9)
        r1_no_tta = (lbl[sim.argmax(dim=1)] == lbl).float().mean().item()

        # With TTA
        emb_tta, lbl_tta = extract_embeddings_tta(model, loader, device)
        sim_tta = torch.mm(emb_tta, emb_tta.t())
        sim_tta.fill_diagonal_(-1e9)
        r1_tta = (lbl_tta[sim_tta.argmax(dim=1)] == lbl_tta).float().mean().item()

        results[split] = {'rank1_no_tta': r1_no_tta, 'rank1_tta': r1_tta, 'tta_gain': r1_tta - r1_no_tta}
        print(f"  CNN [{split}] No TTA: {r1_no_tta:.4f} | TTA: {r1_tta:.4f} | Gain: {r1_tta - r1_no_tta:+.4f}")

    # Full metrics on test with TTA
    if 'test' in results:
        test_ds = MuzzleImageDataset(
            os.path.join(preprocessed_dir, 'test_split.json'), transform=transform
        )
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        emb_test, lbl_test = extract_embeddings_tta(model, test_loader, device)
        metrics = BiometricMetrics()
        full_results = metrics.compute_all_metrics(emb_test, lbl_test)
        metrics.print_summary(full_results)
        # We also populate root-level fields for compare_models.py:
        results.update({
            'test_rank1': full_results['identification']['rank_accuracies']['rank_1'],
            'test_rank5': full_results['identification']['rank_accuracies']['rank_5'],
            'eer': full_results['verification']['eer'],
            'roc_auc': full_results['verification']['roc_auc'],
            'cmc_curve': full_results['identification']['cmc_curve'],
            'fpr': full_results['verification']['fpr'],
            'tpr': full_results['verification']['tpr'],
            'best_val_r1': ckpt.get('val_r1', 0),
        })

        results['test_full_metrics'] = {
            'rank1': full_results['identification']['rank_accuracies']['rank_1'],
            'rank5': full_results['identification']['rank_accuracies']['rank_5'],
            'eer': full_results['verification']['eer'],
            'roc_auc': full_results['verification']['roc_auc'],
        }

    return results


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print("  TTA EVALUATION — All Models")
    print(f"{'='*70}")
    print(f"  Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    all_results = {}

    # ── CNN ────────────────────────────────────────────────────────────────────
    print(f"\n── CNN (EfficientNet-B4 + ArcFace) ────────────────────────────────")
    cnn_results = evaluate_cnn_tta(config, device)
    if cnn_results:
        all_results['cnn'] = cnn_results
        save_stats(cnn_results, str(PROJECT_ROOT / 'outputs/stats/cnn_tta_results.json'))

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  TTA EVALUATION SUMMARY")
    print(f"{'='*70}")
    for model_name, res in all_results.items():
        if 'test' in res:
            test_res = res['test']
            full = res.get('test_full_metrics', {})
            print(f"  {model_name.upper()}:")
            print(f"    No TTA:  Rank-1 = {test_res['rank1_no_tta']:.4f}")
            print(f"    With TTA: Rank-1 = {test_res['rank1_tta']:.4f} ({test_res['tta_gain']:+.4f})")
            if full:
                print(f"    Full TTA Metrics: R1={full['rank1']:.4f} R5={full['rank5']:.4f} EER={full['eer']:.4f}")

    save_stats(all_results, str(PROJECT_ROOT / 'outputs/stats/tta_evaluation_summary.json'))
    print(f"\n✅ TTA results saved to outputs/stats/tta_evaluation_summary.json")


if __name__ == '__main__':
    main()
