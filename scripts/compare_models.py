"""
Script: Compare Models and Generate Paper Results
==================================================
Loads the best checkpoints for all models (GNN+, GNN++, CNN, Hybrid),
performs a unified evaluation on the test set, and generates:
  1. Comparison Table (Rank-1, Rank-5, EER, AUC)
  2. CMC Curves (Cumulative Match Characteristic)
  3. ROC Curves (Receiver Operating Characteristic)
  4. SIFT vs Kornia DISK comparison panel (if stats available)
  5. Training convergence curves

Produces the final "main result" figures for the research paper.
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, ensure_dirs


# ── Publication Styling ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

sns.set_palette("colorblind")

# Model display name remapping
MODEL_NAMES = {
    'GNN_PLUS': 'GNN+ (Kornia DISK)',
    'GNN_PLUS_V2': 'GNN++ (CNN Patches)',
    'GNN_V3_OPTIMIZED': 'GNN v3 (GATv2 + VN)',
    'GNN_V4_ENHANCED': 'GNN v4 (GATv2 + VN - Enhanced)',
    'CNN': 'CNN (EfficientNet-B4)',
    'CNN_TTA': 'CNN (with TTA)',
    'HYBRID': 'Hybrid CNN-GNN',
    'PROTON': 'ProtoN (Prototype Node)',
    'ENSEMBLE': 'Ensemble (CNN TTA + Hybrid)',
    'VISGIN': 'VisGIN (Visibility GNN)',
    'KEYPOINTMATCHER': 'Keypoint Matcher (Sinkhorn)',
    'VGG_BASELINE': 'VGG-16 Baseline (Bello et al. 2020)',
    'RESNET_BASELINE': 'ResNet-50 Baseline (Qin et al. 2021)',
}

MODEL_ORDER = ['ENSEMBLE', 'CNN_TTA', 'CNN', 'HYBRID', 'PROTON', 'VGG_BASELINE', 'RESNET_BASELINE', 'KEYPOINTMATCHER', 'VISGIN', 'GNN_V4_ENHANCED', 'GNN_V3_OPTIMIZED', 'GNN_PLUS_V2', 'GNN_PLUS']



def load_model_results(stats_dir):
    """Load results from JSON files in stats directory."""
    results = {}
    for f in os.listdir(stats_dir):
        if f.endswith('_results.json'):
            name = f.replace('_results.json', '').upper()
            with open(os.path.join(stats_dir, f)) as j:
                data = json.load(j)
            results[name] = data
    return results


def get_display_name(raw_name):
    return MODEL_NAMES.get(raw_name, raw_name)


def sorted_models(all_results):
    """Sort models by priority order, falling back to Rank-1 sort for unknowns."""
    ordered = []
    for name in MODEL_ORDER:
        if name in all_results:
            ordered.append(name)
    for name in sorted(all_results.keys(), key=lambda x: all_results[x].get('test_rank1', 0), reverse=True):
        if name not in ordered:
            ordered.append(name)
    return ordered


def plot_cmc_curves(all_results, save_path):
    """Plot high-DPI CMC curves using actual Rank 1-10 data."""
    plt.figure(figsize=(10, 7))

    markers = ['o', 's', '^', 'D', 'v', '*']
    colors  = ['#E74C3C', '#2ECC71', '#F39C12', '#3498DB', '#9B59B6', '#1ABC9C', '#34495E', '#16A085', '#D35400']

    for i, name in enumerate(sorted_models(all_results)):
        res = all_results[name]
        cmc = res.get('cmc_curve')
        if cmc is None:
            print(f"  [WARN] {name} has no cmc_curve in JSON, skipping plot.")
            continue

        ranks = np.arange(1, min(len(cmc), 10) + 1)
        cmc_slice = np.array(cmc)[:len(ranks)] * 100
        display = get_display_name(name)

        plt.plot(ranks, cmc_slice,
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 markersize=8, linewidth=2.5,
                 label=f"{display} (R1: {cmc_slice[0]:.1f}%)")

    plt.xlabel('Rank')
    plt.ylabel('Identification Accuracy (%)')
    plt.title('Cumulative Match Characteristic (CMC) Curve')
    plt.xticks(np.arange(1, 11))
    plt.yticks(np.arange(0, 101, 10))
    plt.ylim([0, 105])
    plt.xlim([0.8, 10.2])
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(all_results, save_path):
    """Plot high-DPI ROC curves using actual FPR and TPR data."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#E74C3C', '#2ECC71', '#F39C12', '#3498DB', '#9B59B6', '#1ABC9C', '#34495E', '#16A085', '#D35400']

    for i, name in enumerate(sorted_models(all_results)):
        res = all_results[name]
        fpr = res.get('fpr')
        tpr = res.get('tpr')
        auc = res.get('roc_auc', 0)

        if fpr is None or tpr is None:
            print(f"  [WARN] {name} has no fpr/tpr in JSON, skipping plot.")
            continue

        display = get_display_name(name)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=2.5, label=f"{display} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Random Chance')

    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Zoom in plot inset for top left corner
    axins = ax.inset_axes([0.4, 0.2, 0.4, 0.4])
    for i, name in enumerate(sorted_models(all_results)):
        res = all_results[name]
        fpr = res.get('fpr')
        tpr = res.get('tpr')
        if fpr is not None and tpr is not None:
            axins.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2)

    axins.set_xlim(-0.01, 0.15)
    axins.set_ylim(0.85, 1.01)
    axins.grid(True, linestyle='--', alpha=0.5)
    axins.set_title('Zoom: Low FPR Region', fontsize=9)
    ax.indicate_inset_zoom(axins, edgecolor='black')

    plt.savefig(save_path)
    plt.close()


def plot_training_curves(all_results, save_path):
    """Plot training convergence curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Convergence Comparison', fontsize=16, fontweight='bold')

    colors = ['#E74C3C', '#2ECC71', '#F39C12', '#3498DB', '#9B59B6', '#1ABC9C', '#34495E', '#16A085', '#D35400']

    for i, name in enumerate(sorted_models(all_results)):
        res = all_results[name]
        history = res.get('history', {})
        display = get_display_name(name)
        color = colors[i % len(colors)]

        train_loss = history.get('train_loss', [])
        val_r1 = history.get('val_r1', [])

        if train_loss:
            epochs = np.arange(1, len(train_loss) + 1)
            axes[0].plot(epochs, train_loss, label=display, color=color, linewidth=2)

        if val_r1:
            epochs_r1 = np.arange(1, len(val_r1) + 1)
            axes[1].plot(epochs_r1, [v * 100 for v in val_r1],
                         label=display, color=color, linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Convergence')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.5)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Rank-1 Accuracy (%)')
    axes[1].set_title('Validation Rank-1 Accuracy')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_comparison_table(all_results, save_path):
    """Generate a Markdown table of results."""
    table = "| Model | Rank-1 (%) | Rank-5 (%) | EER (%) | ROC AUC | Best Val R1 |\n"
    table += "|-------|------------|------------|---------|---------|-------------|\n"

    for name in sorted_models(all_results):
        res = all_results[name]
        display = get_display_name(name)
        r1 = res.get('test_rank1', 0) * 100
        r5 = res.get('test_rank5', 0) * 100
        eer = res.get('eer', 0) * 100
        auc = res.get('roc_auc', 0)
        val_r1 = res.get('best_val_r1', 0) * 100
        table += f"| {display} | {r1:.1f} | {r5:.1f} | {eer:.2f} | {auc:.4f} | {val_r1:.1f} |\n"

    with open(save_path, 'w') as f:
        f.write("# Model Comparison Results\n\n")
        f.write("Comparative study: CNN, GNN+, GNN++, and Hybrid architectures.\n")
        f.write("Feature extraction: Kornia DISK neural keypoint detector.\n\n")
        f.write(table)
        f.write("\n## Notes\n\n")
        f.write("- GNN+ uses Kornia DISK keypoints (replaces classical SIFT)\n")
        f.write("- GNN++ adds MobileNetV3 CNN patch features (576-d) to GNN nodes\n")
        f.write("- Hybrid uses EfficientNet-B3 feature map sampling at keypoint locations\n")
        f.write("- All models use ArcFace loss with cosine annealing LR schedule\n")
    return table


def plot_bar_comparison(all_results, save_path):
    """Publication-ready bar chart comparison."""
    names_ordered = sorted_models(all_results)
    display_names = [get_display_name(n) for n in names_ordered]

    rank1 = [all_results[n].get('test_rank1', 0) * 100 for n in names_ordered]
    rank5 = [all_results[n].get('test_rank5', 0) * 100 for n in names_ordered]
    eer   = [all_results[n].get('eer', 0) * 100 for n in names_ordered]
    auc   = [all_results[n].get('roc_auc', 0) * 100 for n in names_ordered]

    x = np.arange(len(names_ordered))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - 1.5*width, rank1, width, label='Rank-1 (%)', color='#3498DB', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, rank5, width, label='Rank-5 (%)', color='#2ECC71', alpha=0.85, edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, [100-e for e in eer], width, label='1-EER (%)', color='#E74C3C', alpha=0.85, edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, auc,   width, label='ROC AUC×100', color='#9B59B6', alpha=0.85, edgecolor='black')

    def add_labels(bars, fmt='{:.1f}'):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    fmt.format(h), ha='center', va='bottom', fontsize=8, fontweight='bold')

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)

    ax.set_xlabel('Model Architecture', fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Biometric Identification Benchmark\n(Kornia DISK Neural Features)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=15, ha='right', fontsize=11)
    ax.set_ylim([0, 110])
    ax.legend(loc='upper left', fontsize=11, frameon=True)
    ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    config = load_config()
    stats_dir   = str(PROJECT_ROOT / 'outputs/stats')
    results_dir = str(PROJECT_ROOT / 'outputs/results')
    figure_dir  = str(PROJECT_ROOT / 'outputs/figures')
    ensure_dirs(results_dir, figure_dir)

    print(f"\n{'='*60}")
    print("  PAPER RESULTS GENERATION")
    print(f"{'='*60}")

    # 1. Load results
    all_results = load_model_results(stats_dir)
    if not all_results:
        print("  [ERROR] No result files found in outputs/stats/. ")
        print("  Please run training scripts first.")
        return

    print(f"  Loaded results for: {', '.join(all_results.keys())}")

    # Check for missing fields and warn
    for name, res in all_results.items():
        missing = []
        for field in ['cmc_curve', 'fpr', 'tpr', 'test_rank1', 'roc_auc']:
            if res.get(field) is None:
                missing.append(field)
        if missing:
            print(f"  [WARN] {name} is missing fields: {missing}")

    # 2. Generate comparison table
    table = generate_comparison_table(all_results,
                                      os.path.join(results_dir, 'publication_report.md'))
    print(f"\n{table}")

    # 3. Bar chart (quick visual)
    bar_path = os.path.join(figure_dir, 'bar_comparison.png')
    plot_bar_comparison(all_results, bar_path)
    print(f"  [SAVED] Bar chart -> {bar_path}")

    # 4. Plot CMC Curves
    cmc_path = os.path.join(figure_dir, 'cmc_comparison.png')
    plot_cmc_curves(all_results, cmc_path)
    print(f"  [SAVED] CMC plot -> {cmc_path}")

    # 5. Plot ROC Curves
    roc_path = os.path.join(figure_dir, 'roc_comparison.png')
    plot_roc_curves(all_results, roc_path)
    print(f"  [SAVED] ROC plot -> {roc_path}")

    # 6. Training convergence curves
    conv_path = os.path.join(figure_dir, 'training_convergence.png')
    plot_training_curves(all_results, conv_path)
    print(f"  [SAVED] Convergence curves -> {conv_path}")

    # 7. Note about SIFT vs Kornia
    sift_kornia_stats = os.path.join(stats_dir, 'sift_vs_kornia_stats.json')
    if os.path.exists(sift_kornia_stats):
        with open(sift_kornia_stats) as f:
            sk = json.load(f)
        print(f"\n  SIFT vs Kornia DISK Summary:")
        comp = sk.get('comparison', {})
        for method, m_data in comp.items():
            print(f"    {method}: {m_data.get('avg_keypoints', 0):.1f} kp avg, "
                  f"{m_data.get('avg_coverage', 0)*100:.1f}% coverage")
    else:
        print(f"\n  [INFO] Run compare_sift_vs_kornia.py to see SIFT vs Kornia stats.")

    print(f"\n{'='*60}")
    print("  Comparison summary generated successfully.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
