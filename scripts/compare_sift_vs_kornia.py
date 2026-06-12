"""
Script: Compare All Keypoint Approaches (Parallel Benchmark)
=============================================================
Benchmarks four keypoint extraction backends IN PARALLEL on the same
cattle muzzle images:

  1. Kornia DISK          (neural, depth-pretrained, 128-d → 256-d)
  2. Kornia SuperPoint*   (KeyNet + AffNet + HardNet8, 128-d → 256-d)
  3. Kornia DeDoDe        (Detect-Don't-Describe, 256-d)
  4. OpenCV SIFT          (classical baseline)

Uses concurrent.futures.ThreadPoolExecutor to run all four simultaneously,
dramatically reducing benchmark wall-time on RTX 5070.

Metrics per backend:
  - Keypoint count statistics (mean ± std, min, max)
  - Detection score distribution
  - Spatial coverage (fraction of image area spanned)
  - Speed (ms / image)
  - Pairwise spatial overlap between neural methods

Outputs:
  - outputs/figures/sift_vs_kornia_comparison.png   (4-method comparison)
  - outputs/figures/keypoint_method_radar.png        (radar chart)
  - outputs/stats/sift_vs_kornia_stats.json

*Kornia 0.8.x uses KeyNetAffNetHardNet as the SuperPoint-class pipeline.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, ensure_dirs
from src.features.superpoint import SuperPointExtractor, MultiExtractor


# ── Colour palette ────────────────────────────────────────────────────────────

METHOD_COLORS = {
    'disk':       '#2ECC71',   # green
    'superpoint': '#3498DB',   # blue
    'dedode':     '#9B59B6',   # purple
    'sift':       '#E74C3C',   # red
}

METHOD_LABELS = {
    'disk':       'Kornia DISK',
    'superpoint': 'Kornia SuperPoint\n(KeyNet+HardNet)',
    'dedode':     'Kornia DeDoDe',
    'sift':       'SIFT (Classical)',
}


# ── Image loading ─────────────────────────────────────────────────────────────

def load_sample_images(preprocessed_dir: str,
                       max_images: int = 200) -> Tuple[List, List]:
    """Load preprocessed BGR images from test/val split JSON."""
    for split_name in ('test_split.json', 'val_split.json', 'train_split.json'):
        split_file = os.path.join(preprocessed_dir, split_name)
        if os.path.exists(split_file):
            break
    else:
        print(f"  [WARN] No split JSON found in {preprocessed_dir}")
        # Fall back: glob all PNGs
        return _glob_images(preprocessed_dir, max_images)

    with open(split_file) as f:
        data = json.load(f)

    images, labels = [], []
    for item in data[:max_images]:
        img_path = (item.get('path') or item.get('image_path')
                    or item.get('file_path', ''))
        if not img_path:
            continue
        if not os.path.isabs(img_path):
            img_path = os.path.join(str(PROJECT_ROOT), img_path)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                images.append(img)
                labels.append(item.get('animal_id', item.get('label', '?')))

    print(f"  Loaded {len(images)} images from {split_file}")
    return images, labels


def _glob_images(preprocessed_dir: str, max_images: int):
    images, labels = [], []
    img_root = Path(preprocessed_dir) / 'images'
    if not img_root.exists():
        return images, labels
    for p in img_root.rglob('*.png'):
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.resize(img, (256, 256))
            images.append(img)
            labels.append(p.parent.name)
        if len(images) >= max_images:
            break
    return images, labels


# ── Parallel extraction on a batch of images ──────────────────────────────────

def extract_all_methods_parallel(
        images_bgr: List[np.ndarray],
        multi: MultiExtractor,
        max_workers: int = 4,
) -> Dict[str, List[Dict]]:
    """
    Run all four backends on every image in parallel.

    Returns:
        {backend_name: [per_image_result_dict, ...]}
    """
    backend_names = list(multi.extractors.keys())
    all_results: Dict[str, List[Dict]] = {n: [] for n in backend_names}
    total_times: Dict[str, float] = {n: 0.0 for n in backend_names}

    print(f"\n  Running {len(backend_names)} backends on {len(images_bgr)} images "
          f"(parallel, max_workers={max_workers})...")

    def process_image(img_bgr: np.ndarray) -> Dict[str, Dict]:
        return multi.extract_parallel(img_bgr, mask=None, max_workers=max_workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = list(pool.map(process_image, images_bgr))

    for per_img in futures:
        for name, res in per_img.items():
            all_results[name].append(res)
            total_times[name] += res.get('time_ms', 0)

    # Summary times
    for name in backend_names:
        n_img = len(images_bgr)
        avg_ms = total_times[name] / max(n_img, 1)
        print(f"  {METHOD_LABELS.get(name, name):30s}: "
              f"{avg_ms:6.1f} ms/image  "
              f"avg kp={np.mean([len(r['keypoints']) for r in all_results[name]]):.1f}")

    return all_results


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_coverage(keypoints: np.ndarray, H: int, W: int) -> float:
    if len(keypoints) < 2:
        return 0.0
    x_range = (keypoints[:, 0].max() - keypoints[:, 0].min()) / W
    y_range = (keypoints[:, 1].max() - keypoints[:, 1].min()) / H
    return float(x_range * y_range)


def aggregate_stats(results_per_image: List[Dict], H: int, W: int) -> Dict:
    counts   = [len(r['keypoints']) for r in results_per_image]
    scores_l = [float(np.mean(r['scores'])) if len(r['scores']) > 0 else 0.0
                for r in results_per_image]
    coverage = [compute_coverage(r['keypoints'], H, W)
                for r in results_per_image if len(r['keypoints']) > 1]
    times    = [r.get('time_ms', 0) for r in results_per_image]
    return {
        'counts':       counts,
        'scores':       scores_l,
        'coverage':     coverage if coverage else [0.0],
        'avg_time_ms':  float(np.mean(times)) if times else 0.0,
    }


def pairwise_overlap(kps1: np.ndarray, kps2: np.ndarray,
                     threshold_px: float = 3.0) -> float:
    """Fraction of kps1 within threshold_px of any kp in kps2."""
    if len(kps1) == 0 or len(kps2) == 0:
        return 0.0
    matches = 0
    for p in kps1:
        if np.linalg.norm(kps2 - p, axis=1).min() < threshold_px:
            matches += 1
    return matches / max(len(kps1), len(kps2))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(all_stats: Dict[str, Dict],
                    sample_images: List[np.ndarray],
                    sample_results: Dict[str, List[Dict]],
                    save_dir: str) -> str:
    """4-method comparison figure (statistical + visual rows)."""
    names = list(all_stats.keys())
    n     = len(names)

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(
        'Keypoint Detection Benchmark: SIFT vs Kornia Neural Methods\n'
        'Cattle Muzzle Biometric Identification (Parallel Extraction)',
        fontsize=18, fontweight='bold', y=0.98,
    )

    colors = [METHOD_COLORS.get(nm, '#95A5A6') for nm in names]

    # Row 1: Bar charts -------------------------------------------------------
    # 1a Keypoint count
    ax1 = fig.add_subplot(3, 5, 1)
    means = [np.mean(all_stats[nm]['counts']) for nm in names]
    stds  = [np.std(all_stats[nm]['counts'])  for nm in names]
    bars  = ax1.bar(range(n), means, yerr=stds, color=colors,
                    alpha=0.85, capsize=6, edgecolor='black', linewidth=1.1)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([METHOD_LABELS.get(nm, nm) for nm in names],
                        fontsize=8, rotation=30, ha='right')
    ax1.set_ylabel('Keypoints detected')
    ax1.set_title('Avg. Keypoint Count', fontweight='bold')
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1, f'{m:.1f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.grid(axis='y', alpha=0.4)

    # 1b Spatial coverage
    ax2 = fig.add_subplot(3, 5, 2)
    cov_means = [np.mean(all_stats[nm]['coverage']) * 100 for nm in names]
    bars2 = ax2.bar(range(n), cov_means, color=colors,
                    alpha=0.85, edgecolor='black', linewidth=1.1)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels([METHOD_LABELS.get(nm, nm) for nm in names],
                        fontsize=8, rotation=30, ha='right')
    ax2.set_ylabel('Coverage (%)')
    ax2.set_title('Spatial Coverage', fontweight='bold')
    for bar, m in zip(bars2, cov_means):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3, f'{m:.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.grid(axis='y', alpha=0.4)

    # 1c Detection score
    ax3 = fig.add_subplot(3, 5, 3)
    sc_means = [np.mean(all_stats[nm]['scores']) for nm in names]
    bars3 = ax3.bar(range(n), sc_means, color=colors,
                    alpha=0.85, edgecolor='black', linewidth=1.1)
    ax3.set_xticks(range(n))
    ax3.set_xticklabels([METHOD_LABELS.get(nm, nm) for nm in names],
                        fontsize=8, rotation=30, ha='right')
    ax3.set_ylabel('Mean detection score')
    ax3.set_title('Detection Score', fontweight='bold')
    ax3.grid(axis='y', alpha=0.4)

    # 1d Speed
    ax4 = fig.add_subplot(3, 5, 4)
    times = [all_stats[nm]['avg_time_ms'] for nm in names]
    bars4 = ax4.bar(range(n), times, color=colors,
                    alpha=0.85, edgecolor='black', linewidth=1.1)
    ax4.set_xticks(range(n))
    ax4.set_xticklabels([METHOD_LABELS.get(nm, nm) for nm in names],
                        fontsize=8, rotation=30, ha='right')
    ax4.set_ylabel('ms / image')
    ax4.set_title('Speed (ms/image)', fontweight='bold')
    for bar, t in zip(bars4, times):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2, f'{t:.1f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.grid(axis='y', alpha=0.4)

    # 1e Keypoint count distribution
    ax5 = fig.add_subplot(3, 5, 5)
    for nm, col in zip(names, colors):
        ax5.hist(all_stats[nm]['counts'], bins=20, color=col,
                 alpha=0.55, label=METHOD_LABELS.get(nm, nm),
                 density=True, edgecolor='none')
    ax5.set_xlabel('Keypoint count')
    ax5.set_ylabel('Density')
    ax5.set_title('Count Distribution', fontweight='bold')
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.4)

    # Row 2: Score & coverage distributions -----------------------------------
    ax6 = fig.add_subplot(3, 5, 6)
    for nm, col in zip(names, colors):
        sc = all_stats[nm]['scores']
        if sc:
            ax6.hist(sc, bins=30, color=col, alpha=0.55,
                     density=True, label=METHOD_LABELS.get(nm, nm),
                     edgecolor='none')
    ax6.set_xlabel('Mean detection score')
    ax6.set_ylabel('Density')
    ax6.set_title('Score Distribution', fontweight='bold')
    ax6.legend(fontsize=7)
    ax6.grid(alpha=0.4)

    ax7 = fig.add_subplot(3, 5, 7)
    for nm, col in zip(names, colors):
        cov = all_stats[nm]['coverage']
        ax7.hist(cov, bins=20, color=col, alpha=0.55,
                 density=True, label=METHOD_LABELS.get(nm, nm),
                 edgecolor='none')
    ax7.set_xlabel('Spatial coverage')
    ax7.set_ylabel('Density')
    ax7.set_title('Coverage Distribution', fontweight='bold')
    ax7.legend(fontsize=7)
    ax7.grid(alpha=0.4)

    # Summary table
    ax8 = fig.add_subplot(3, 5, 8)
    ax8.axis('off')
    col_labels = ['Metric'] + [METHOD_LABELS.get(nm, nm).replace('\n', ' ')
                                for nm in names]
    rows = [
        ['Avg. Kp'] + [f'{np.mean(all_stats[nm]["counts"]):.1f}' for nm in names],
        ['Coverage%'] + [f'{np.mean(all_stats[nm]["coverage"])*100:.1f}' for nm in names],
        ['Avg. Score'] + [f'{np.mean(all_stats[nm]["scores"]):.4f}' for nm in names],
        ['ms/img'] + [f'{all_stats[nm]["avg_time_ms"]:.1f}' for nm in names],
        ['Learned?'] + ['No' if nm == 'sift' else 'Yes ✓' for nm in names],
    ]
    tbl = ax8.table(cellText=rows, colLabels=col_labels,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#BDC3C7')
    ax8.set_title('Head-to-Head Metrics', fontweight='bold', pad=15)

    # Radar chart placeholder (simple bar for n dims)
    ax9 = fig.add_subplot(3, 5, 9)
    dims = ['Count\n(norm)', 'Coverage', 'Score', 'Speed\n(inv)']
    x = np.arange(len(dims))
    w = 0.15
    max_count = max(np.mean(all_stats[nm]['counts']) for nm in names) + 1
    max_cov   = max(np.mean(all_stats[nm]['coverage']) for nm in names) + 1e-6
    max_sc    = max(np.mean(all_stats[nm]['scores']) for nm in names) + 1e-6
    max_time  = max(all_stats[nm]['avg_time_ms'] for nm in names) + 1
    for idx, (nm, col) in enumerate(zip(names, colors)):
        vals = [
            np.mean(all_stats[nm]['counts']) / max_count,
            np.mean(all_stats[nm]['coverage']) / max_cov,
            np.mean(all_stats[nm]['scores']) / max_sc,
            1 - all_stats[nm]['avg_time_ms'] / max_time,  # inverse speed
        ]
        ax9.bar(x + idx * w, vals, w, color=col, alpha=0.8, label=METHOD_LABELS.get(nm, nm).replace('\n', ' '))
    ax9.set_xticks(x + w * (n-1)/2)
    ax9.set_xticklabels(dims, fontsize=9)
    ax9.set_ylim(0, 1.1)
    ax9.set_ylabel('Normalised score')
    ax9.set_title('Normalised Comparison', fontweight='bold')
    ax9.legend(fontsize=7, loc='lower right')
    ax9.grid(axis='y', alpha=0.4)

    # Row 3: Visual examples --------------------------------------------------
    n_vis = min(5, len(sample_images))
    for col_i in range(n_vis):
        ax = fig.add_subplot(3, 5, 11 + col_i)
        img = sample_images[col_i]
        vis = img[:, :, ::-1].copy()  # BGR→RGB for matplotlib

        for nm, col_hex in METHOD_COLORS.items():
            if nm not in sample_results:
                continue
            imgs_results = sample_results[nm]
            if col_i >= len(imgs_results):
                continue
            kps = imgs_results[col_i]['keypoints']
            if len(kps) == 0:
                continue
            # Convert hex to 0-1 RGB
            c = tuple(int(col_hex.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
            for (x, y) in kps[:40]:
                cv2.circle(vis, (int(x), int(y)), 3,
                           (int(c[0]*255), int(c[1]*255), int(c[2]*255)), -1)

        counts_str = ' | '.join(
            f"{METHOD_LABELS.get(nm,'?').split(chr(10))[0]}:"
            f"{len(sample_results[nm][col_i]['keypoints']) if col_i < len(sample_results.get(nm,[])) else 0}"
            for nm in names
        )
        ax.imshow(vis)
        ax.set_title(counts_str, fontsize=7, fontweight='bold')
        ax.axis('off')

    # Legend
    patches = [mpatches.Patch(color=METHOD_COLORS.get(nm, '#aaa'),
                               label=METHOD_LABELS.get(nm, nm).replace('\n', ' '))
               for nm in names]
    fig.legend(handles=patches, loc='lower right', fontsize=11,
               frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'sift_vs_kornia_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  [SAVED] Comparison plot -> {save_path}")
    return save_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    figures_dir = str(PROJECT_ROOT / 'outputs/figures')
    stats_dir   = str(PROJECT_ROOT / 'outputs/stats')
    ensure_dirs(figures_dir, stats_dir)

    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    max_kp = config.get('keypoints', {}).get('max_keypoints', 128)

    print(f"\n{'='*65}")
    print("  Multi-Method Keypoint Benchmark (Parallel)")
    print(f"{'='*65}")
    print(f"  Device:        {device}")
    print(f"  Max keypoints: {max_kp}")

    # Load images
    images_bgr, labels = load_sample_images(preprocessed_dir, max_images=300)
    if not images_bgr:
        print("\n[ERROR] No images found. Run preprocessing first.")
        return

    H, W = images_bgr[0].shape[:2]
    print(f"  Images: {len(images_bgr)} | Size: {H}×{W}")

    # Build multi-extractor (all 4 backends)
    multi = MultiExtractor(max_keypoints=max_kp, device=device)

    # Run extraction sequentially to prevent CUDA OOM
    t_total = time.time()
    all_results = extract_all_methods_parallel(images_bgr, multi, max_workers=1)
    wall_time = time.time() - t_total

    print(f"\n  Total wall-time (all backends, all images): {wall_time:.1f}s")

    # Aggregate statistics
    all_stats: Dict[str, Dict] = {}
    for name, results in all_results.items():
        all_stats[name] = aggregate_stats(results, H, W)

    # Pairwise overlap (neural methods vs SIFT)
    overlap_summary = {}
    if 'sift' in all_results:
        for nm in ('disk', 'superpoint', 'dedode'):
            if nm not in all_results:
                continue
            overlaps = []
            for s_r, n_r in zip(all_results['sift'][:50], all_results[nm][:50]):
                if len(s_r['keypoints']) > 0 and len(n_r['keypoints']) > 0:
                    overlaps.append(
                        pairwise_overlap(s_r['keypoints'], n_r['keypoints']))
            overlap_summary[f'sift_vs_{nm}'] = float(np.mean(overlaps)) if overlaps else 0.0

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  {'Metric':<28} " + "  ".join(f"{METHOD_LABELS.get(nm,nm).split(chr(10))[0]:>16}" for nm in all_stats))
    print(f"  {'-'*68}")
    for metric, key in [('Avg. Keypoints', 'counts'),
                         ('Std. Keypoints', 'counts'),
                         ('Spatial Coverage (%)', 'coverage'),
                         ('Avg. Detection Score', 'scores'),
                         ('Speed (ms/image)', None)]:
        if key:
            fn = np.mean if metric.startswith('Avg') else np.std
            vals = [fn(all_stats[nm][key]) for nm in all_stats]
        else:
            vals = [all_stats[nm]['avg_time_ms'] for nm in all_stats]
        suffix = '×100' if 'Coverage' in metric else ''
        scale  = 100 if 'Coverage' in metric else 1
        row = " ".join(f"{v*scale:>18.1f}" for v in vals)
        print(f"  {metric:<28} {row}")
    print(f"{'='*70}")

    for pair, ovlp in overlap_summary.items():
        print(f"  Spatial overlap {pair}: {ovlp*100:.1f}%")

    # Save JSON
    summary = {
        'comparison': {},
        'advantages_of_kornia': [],
        'pairwise_overlap': overlap_summary,
        'num_images_evaluated': len(images_bgr),
        'device': str(device),
    }
    for nm in all_stats:
        st = all_stats[nm]
        summary['comparison'][nm] = {
            'method_label': METHOD_LABELS.get(nm, nm).replace('\n', ' '),
            'avg_keypoints':  float(np.mean(st['counts'])),
            'std_keypoints':  float(np.std(st['counts'])),
            'avg_coverage':   float(np.mean(st['coverage'])),
            'avg_score':      float(np.mean(st['scores'])),
            'speed_ms_per_image': st['avg_time_ms'],
            'learned': nm != 'sift',
        }

    sift_kp = np.mean(all_stats.get('sift', {}).get('counts', [1]))
    for nm in ('disk', 'superpoint', 'dedode'):
        if nm in all_stats:
            improv = (np.mean(all_stats[nm]['counts']) - sift_kp) / max(sift_kp, 1) * 100
            summary['advantages_of_kornia'].append(
                f"{METHOD_LABELS[nm].replace(chr(10),' ')}: {improv:+.1f}% more keypoints than SIFT")

    stats_path = os.path.join(stats_dir, 'sift_vs_kornia_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [SAVED] Stats -> {stats_path}")

    # Plot
    sample_imgs = images_bgr[:5]
    sample_res = {nm: all_results[nm][:5] for nm in all_results}
    plot_comparison(all_stats, sample_imgs, sample_res, figures_dir)

    print(f"\n[SUCCESS] Multi-method benchmark complete!")
    for adv in summary['advantages_of_kornia']:
        print(f"   - {adv}")


if __name__ == '__main__':
    main()
