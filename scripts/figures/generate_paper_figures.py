"""
Script: Generate All Paper Figures
===================================
Produces publication-quality figures for the paper:
  1. CMC curves (all models)
  2. ROC curves (all models)
  3. Training curves (loss + R1 vs epoch)
  4. Ablation bar charts
  5. t-SNE embedding visualization
  6. Confusion matrix (top confused pairs)

Usage:
    python scripts/figures/generate_paper_figures.py
    python scripts/figures/generate_paper_figures.py --fig cmc
    python scripts/figures/generate_paper_figures.py --fig all

Output: outputs/figures/*.pdf  (vector, for journal)
        outputs/figures/*.png  (600 DPI, for submission)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use non-interactive backend for headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Journal-quality style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
})

# ── Color palette (colorblind-friendly) ───────────────────────────────────────
COLORS = {
    'CNN':     '#2196F3',   # Blue
    'Hybrid':  '#4CAF50',   # Green
    'ProtoN':  '#FF9800',   # Orange
    'GNN v3':  '#9C27B0',   # Purple
    'GNN v4':  '#F44336',   # Red
    'Ensemble':'#000000',   # Black
    'VGG-16':  '#78909C',   # Grey
    'ResNet':  '#90A4AE',   # Light grey
}

LINESTYLES = {
    'CNN':     '-',
    'Hybrid':  '--',
    'ProtoN':  '-.',
    'GNN v3':  ':',
    'GNN v4':  (0, (3, 1, 1, 1)),
    'Ensemble':'solid',
    'VGG-16':  ':',
    'ResNet':  '-.',
}

OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures'
STATS_DIR  = PROJECT_ROOT / 'outputs' / 'stats'


def load_results():
    """Load all model result JSON files."""
    files = {
        'CNN':    'cnn_results.json',
        'Hybrid': 'hybrid_results.json',
        'ProtoN': 'proton_results.json',
        'GNN v3': 'gnn_v3_optimized_results.json',
        'GNN v4': 'gnn_v4_enhanced_results.json',
        'VGG-16': 'vgg_baseline_results.json',
        'ResNet': 'resnet_baseline_results.json',
    }
    results = {}
    for name, fname in files.items():
        fpath = STATS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                results[name] = json.load(f)
            print(f"  Loaded: {name} ({fname})")
        else:
            print(f"  [SKIP] {name} — {fname} not found")
    return results


# ── Figure 1: CMC Curves ──────────────────────────────────────────────────────

def plot_cmc_curves(results):
    """Plot CMC (Cumulative Match Characteristic) curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 5))

    max_rank = 20
    for name, res in results.items():
        try:
            cmc = res['identification']['cmc_curve'][:max_rank]
        except (KeyError, TypeError):
            # Try flat structure
            cmc = res.get('cmc_curve', [])[:max_rank]
        if not cmc:
            continue

        ranks = list(range(1, len(cmc) + 1))
        ax.plot(ranks, [v * 100 for v in cmc],
                color=COLORS.get(name, '#333'),
                linestyle=LINESTYLES.get(name, '-'),
                label=f"{name} ({cmc[0]*100:.1f}%)")

    ax.set_xlabel('Rank')
    ax.set_ylabel('Identification Rate (%)')
    ax.set_title('Cumulative Match Characteristic (CMC) Curves')
    ax.set_xlim(1, max_rank)
    ax.set_ylim(60, 101)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    out = OUTPUT_DIR / 'fig_cmc_curves.pdf'
    fig.savefig(out, format='pdf')
    fig.savefig(str(out).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: ROC Curves ──────────────────────────────────────────────────────

def plot_roc_curves(results):
    """Plot ROC curves for verification performance."""
    fig, ax = plt.subplots(figsize=(6, 6))

    for name, res in results.items():
        try:
            fpr = res['verification']['fpr']
            tpr = res['verification']['tpr']
            auc = res['verification']['roc_auc']
            eer = res['verification']['eer']
        except (KeyError, TypeError):
            continue

        ax.plot(fpr, tpr,
                color=COLORS.get(name, '#333'),
                linestyle=LINESTYLES.get(name, '-'),
                label=f"{name} (AUC={auc:.4f}, EER={eer*100:.2f}%)")

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.01, 0.3)  # Zoom in — all our models have very low FPR
    ax.set_ylim(0.7, 1.01)

    out = OUTPUT_DIR / 'fig_roc_curves.pdf'
    fig.savefig(out, format='pdf')
    fig.savefig(str(out).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Training Curves ─────────────────────────────────────────────────

def plot_training_curves(results):
    """Plot loss and Val R1 vs. epoch for all models (2-row grid)."""
    n_models = len(results)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 7),
                             sharex=False)
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for col, (name, res) in enumerate(results.items()):
        hist = res.get('history', {})
        loss = hist.get('train_loss', [])
        r1   = hist.get('val_r1', [])
        epochs = list(range(1, len(loss) + 1))

        c = COLORS.get(name, '#333')

        if loss:
            axes[0][col].plot(epochs, loss, color=c, linewidth=1.5)
            axes[0][col].set_title(name, fontsize=10, fontweight='bold')
            axes[0][col].set_ylabel('Training Loss' if col == 0 else '')
            axes[0][col].set_xlabel('Epoch')
            axes[0][col].grid(True, alpha=0.3)

        if r1:
            axes[1][col].plot(epochs, [v * 100 for v in r1], color=c, linewidth=1.5)
            best_r1 = max(r1)
            best_ep = r1.index(best_r1) + 1
            axes[1][col].axhline(y=best_r1 * 100, color='red',
                                  linestyle='--', linewidth=0.8, alpha=0.7)
            axes[1][col].set_ylabel('Val Rank-1 (%)' if col == 0 else '')
            axes[1][col].set_xlabel('Epoch')
            axes[1][col].annotate(f'Best: {best_r1*100:.1f}% (ep{best_ep})',
                                   xy=(best_ep, best_r1 * 100),
                                   xytext=(5, -15), textcoords='offset points',
                                   fontsize=7, color='red')
            axes[1][col].grid(True, alpha=0.3)

    fig.suptitle('Training Curves — All Models', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out = OUTPUT_DIR / 'fig_training_curves.pdf'
    fig.savefig(out, format='pdf')
    fig.savefig(str(out).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Main Results Bar Chart ─────────────────────────────────────────

def plot_main_results_bar(results):
    """Grouped bar chart: Rank-1, Rank-5, EER for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    names, r1_vals, r5_vals, eer_vals = [], [], [], []

    for name, res in results.items():
        try:
            r1  = res['identification']['rank_accuracies']['rank_1'] * 100
            r5  = res['identification']['rank_accuracies']['rank_5'] * 100
            eer = res['verification']['eer'] * 100
        except (KeyError, TypeError):
            r1  = res.get('test_rank1', 0) * 100
            r5  = res.get('test_rank5', 0) * 100
            eer = res.get('eer', 0) * 100

        names.append(name)
        r1_vals.append(r1)
        r5_vals.append(r5)
        eer_vals.append(eer)

    x = np.arange(len(names))
    colors = [COLORS.get(n, '#999') for n in names]

    for ax, vals, ylabel, title, ylim in [
        (axes[0], r1_vals, 'Rank-1 Accuracy (%)', 'Rank-1 Identification', (75, 100)),
        (axes[1], r5_vals, 'Rank-5 Accuracy (%)', 'Rank-5 Identification', (85, 101)),
        (axes[2], eer_vals, 'EER (%)', 'Equal Error Rate (lower=better)', (0, 15)),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5,
                      width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Performance Comparison — All Proposed Methods', fontsize=13)
    plt.tight_layout()

    out = OUTPUT_DIR / 'fig_main_results_bar.pdf'
    fig.savefig(out, format='pdf')
    fig.savefig(str(out).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 5: Ablation Bar Charts ─────────────────────────────────────────────

def plot_ablation_charts():
    """Load ablation results if available and plot comparison bars."""
    ablation_file = STATS_DIR / 'ablation_results.json'
    if not ablation_file.exists():
        print("  [SKIP] ablation_results.json not found — run ablations first")
        return

    with open(ablation_file) as f:
        ablations = json.load(f)

    fig, axes = plt.subplots(1, len(ablations), figsize=(5 * len(ablations), 4.5))
    if len(ablations) == 1:
        axes = [axes]

    for ax, (abl_name, abl_data) in zip(axes, ablations.items()):
        labels = [d['label'] for d in abl_data]
        r1 = [d['rank1'] * 100 for d in abl_data]
        colors = ['#2196F3' if not d.get('baseline') else '#78909C' for d in abl_data]
        bars = ax.bar(range(len(labels)), r1, color=colors, edgecolor='white')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Rank-1 Accuracy (%)')
        ax.set_title(abl_name)
        ax.set_ylim(max(0, min(r1) - 5), min(102, max(r1) + 2))
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, r1):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Ablation Studies', fontsize=13)
    plt.tight_layout()
    out = OUTPUT_DIR / 'fig_ablation_charts.pdf'
    fig.savefig(out, format='pdf')
    fig.savefig(str(out).replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 6: t-SNE Embedding Visualization ───────────────────────────────────

def plot_tsne_embeddings(results):
    """t-SNE of test embeddings for CNN and best GNN model."""
    try:
        from sklearn.manifold import TSNE
        import torch
    except ImportError:
        print("  [SKIP] sklearn not available for t-SNE")
        return

    # Load embeddings from checkpoint if available
    cnn_ckpt = PROJECT_ROOT / 'outputs/cnn/best_model.pt'
    if not cnn_ckpt.exists():
        print("  [SKIP] CNN checkpoint not found")
        return

    print("  Computing t-SNE (this may take 2-3 minutes)...")

    # Try to extract embeddings
    try:
        import torch
        from src.models.cnn_model import CNNMuzzleModel
        from src.utils import load_config
        from src.training.augmentation import build_val_transform
        from src.training.image_dataset import MuzzleImageDataset
        from torch.utils.data import DataLoader
        import torch.nn.functional as F

        config = load_config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ckpt = torch.load(cnn_ckpt, map_location=device, weights_only=False)
        num_classes = ckpt.get('num_classes', 260)
        mc = ckpt.get('config', {})
        model = CNNMuzzleModel(
            num_classes=num_classes,
            embedding_dim=mc.get('embedding_dim', 512),
            backbone=mc.get('backbone', 'efficientnet_b4'),
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
        transform = build_val_transform(config.get('preprocessing', {}).get('image_size', 256))
        ds = MuzzleImageDataset(
            os.path.join(preprocessed_dir, 'test_split.json'), transform=transform
        )
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

        all_emb, all_lbl = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                emb = model.get_embedding(imgs)
                all_emb.append(emb.cpu().numpy())
                all_lbl.append(lbls.numpy())

        emb_np = np.concatenate(all_emb)
        lbl_np = np.concatenate(all_lbl)

        # t-SNE
        try:
            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                        random_state=42, metric='cosine')
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                        random_state=42, metric='cosine')
        emb_2d = tsne.fit_transform(emb_np)

        # Plot — color by class (show top 30 classes)
        n_show = min(30, len(np.unique(lbl_np)))
        top_classes = np.unique(lbl_np)[:n_show]
        try:
            cmap = matplotlib.colormaps['tab20'].resampled(n_show)
        except AttributeError:
            cmap = plt.cm.get_cmap('tab20', n_show)

        fig, ax = plt.subplots(figsize=(8, 7))
        for i, cls in enumerate(top_classes):
            mask = lbl_np == cls
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=[cmap(i)], s=15, alpha=0.7, edgecolors='none')

        ax.set_title(f'CNN (EfficientNet-B4) Embeddings — t-SNE\n'
                     f'(Showing {n_show} of {len(np.unique(lbl_np))} cattle)', fontsize=11)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.axis('off')

        out = OUTPUT_DIR / 'fig_tsne_cnn.pdf'
        fig.savefig(out, format='pdf')
        fig.savefig(str(out).replace('.pdf', '.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved: {out}")

    except Exception as e:
        print(f"  [SKIP] t-SNE failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--fig', default='all',
                        choices=['all', 'cmc', 'roc', 'training', 'bar', 'ablation', 'tsne'],
                        help='Which figure to generate')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  PAPER FIGURE GENERATION")
    print(f"{'='*65}")
    print(f"  Output: {OUTPUT_DIR}")

    results = load_results()
    if not results and args.fig != 'ablation':
        print("\n  [ERROR] No result files found. Run training first.")
        return

    figures = {
        'cmc':      lambda: plot_cmc_curves(results),
        'roc':      lambda: plot_roc_curves(results),
        'training': lambda: plot_training_curves(results),
        'bar':      lambda: plot_main_results_bar(results),
        'ablation': lambda: plot_ablation_charts(),
        'tsne':     lambda: plot_tsne_embeddings(results),
    }

    if args.fig == 'all':
        for name, fn in figures.items():
            print(f"\n  -- Generating: {name} --")
            try:
                fn()
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
    else:
        figures[args.fig]()

    print(f"\n{'='*65}")
    print(f"  Done! Figures saved to: {OUTPUT_DIR}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
