"""
Script 07: Generate Comprehensive Paper Statistics
=====================================================
Aggregates all statistics from every pipeline stage into a single
comprehensive report for paper writing.

Reads from: outputs/stats/*.json
Output: outputs/stats/paper_statistics.json
        outputs/stats/paper_tables.txt (LaTeX-ready tables)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs


def load_json_safe(filepath):
    """Load JSON file safely."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def generate_latex_tables(stats):
    """Generate LaTeX-formatted tables for the paper."""
    tables = []
    
    # Table 1: Dataset Statistics
    ds = stats.get('dataset', {})
    if ds:
        table = r"""
% Table 1: Dataset Statistics
\begin{table}[h]
\centering
\caption{Dataset Statistics}
\label{tab:dataset}
\begin{tabular}{lr}
\toprule
\textbf{Property} & \textbf{Value} \\
\midrule
Dataset & Beef Cattle Muzzle Database \\
Total Images & """ + str(ds.get('total_images', 'N/A')) + r""" \\
Total Animals & """ + str(ds.get('total_animals', 'N/A')) + r""" \\
Avg Images/Animal & """ + str(ds.get('avg_per_animal', 'N/A')) + r""" \\
Train Set & """ + str(ds.get('train_size', 'N/A')) + r""" \\
Validation Set & """ + str(ds.get('val_size', 'N/A')) + r""" \\
Test Set & """ + str(ds.get('test_size', 'N/A')) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
        tables.append(table)
    
    # Table 2: Preprocessing Statistics
    prep = stats.get('preprocessing', {})
    if prep:
        table = r"""
% Table 2: Preprocessing Statistics
\begin{table}[h]
\centering
\caption{Preprocessing Pipeline Statistics}
\label{tab:preprocessing}
\begin{tabular}{lrr}
\toprule
\textbf{Metric} & \textbf{Before} & \textbf{After} \\
\midrule
RMS Contrast & """ + str(prep.get('contrast_before', 'N/A')) + r""" & """ + str(prep.get('contrast_after', 'N/A')) + r""" \\
Shannon Entropy & """ + str(prep.get('entropy_before', 'N/A')) + r""" & """ + str(prep.get('entropy_after', 'N/A')) + r""" \\
Mask Coverage & -- & """ + str(prep.get('mask_coverage', 'N/A')) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
        tables.append(table)
    
    # Table 3: Model Architecture
    model = stats.get('model', {})
    if model:
        table = r"""
% Table 3: Model Architecture
\begin{table}[h]
\centering
\caption{CattleGNN Architecture Details}
\label{tab:architecture}
\begin{tabular}{lr}
\toprule
\textbf{Component} & \textbf{Details} \\
\midrule
Input Features & SuperPoint 256-d descriptors \\
EdgeConv Layers & 3 (dims: 256, 256, 512) \\
Dynamic KNN & k = 12 \\
TRM Heads & 4 \\
TRM Layers & 2 \\
Global Pooling & Mean + Max (512-d) \\
Embedding Dim & 256 \\
Total Parameters & """ + str(model.get('total_params', 'N/A')) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
        tables.append(table)
    
    # Table 4: Identification Results
    results = stats.get('results', {})
    if results:
        table = r"""
% Table 4: Identification Results
\begin{table}[h]
\centering
\caption{Biometric Identification Performance}
\label{tab:results}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Rank-1 Accuracy & """ + f"{results.get('rank_1', 0)*100:.2f}" + r"""\% \\
Rank-5 Accuracy & """ + f"{results.get('rank_5', 0)*100:.2f}" + r"""\% \\
Rank-10 Accuracy & """ + f"{results.get('rank_10', 0)*100:.2f}" + r"""\% \\
EER & """ + f"{results.get('eer', 0)*100:.2f}" + r"""\% \\
ROC AUC & """ + f"{results.get('roc_auc', 0):.4f}" + r""" \\
TAR @ FAR=1\% & """ + f"{results.get('tar_far_001', 0)*100:.2f}" + r"""\% \\
TAR @ FAR=0.1\% & """ + f"{results.get('tar_far_0001', 0)*100:.2f}" + r"""\% \\
d' (d-prime) & """ + f"{results.get('d_prime', 0):.4f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
        tables.append(table)
    
    return '\n'.join(tables)


def main():
    print("=" * 70)
    print("PAPER STATISTICS COMPILATION")
    print("=" * 70)
    
    config = load_config()
    stats_dir = str(PROJECT_ROOT / config['outputs']['stats_dir'])
    results_dir = str(PROJECT_ROOT / config['outputs']['results_dir'])
    ensure_dirs(stats_dir)
    
    # Load all statistics files
    dataset_stats = load_json_safe(os.path.join(stats_dir, "dataset_stats.json"))
    preprocessing_stats = load_json_safe(os.path.join(stats_dir, "preprocessing_stats.json"))
    keypoint_stats = load_json_safe(os.path.join(stats_dir, "keypoint_stats.json"))
    graph_stats = load_json_safe(os.path.join(stats_dir, "graph_stats.json"))
    training_stats = load_json_safe(os.path.join(stats_dir, "training_stats.json"))
    evaluation_stats = load_json_safe(os.path.join(stats_dir, "evaluation_stats.json"))
    eval_results = load_json_safe(os.path.join(results_dir, "evaluation_results.json"))
    
    # Compile comprehensive paper statistics
    paper_stats = {
        'generated_at': datetime.now().isoformat(),
        
        'dataset': {
            'name': 'Beef Cattle Muzzle Database',
            'source': 'Zenodo (DOI: 10.5281/zenodo.6324361)',
            'total_images': dataset_stats.get('dataset_info', {}).get('total_images', 'N/A'),
            'total_animals': dataset_stats.get('dataset_info', {}).get('total_animals', 'N/A'),
            'avg_per_animal': dataset_stats.get('paper_ready', {}).get('avg_samples_per_class', 'N/A'),
            'train_size': dataset_stats.get('paper_ready', {}).get('train_size', 'N/A'),
            'val_size': dataset_stats.get('paper_ready', {}).get('val_size', 'N/A'),
            'test_size': dataset_stats.get('paper_ready', {}).get('test_size', 'N/A'),
            'split_ratio': '70/15/15',
        },
        
        'preprocessing': {
            'image_size': '256 × 256',
            'clahe_clip_limit': preprocessing_stats.get('clahe_enhancement', {}).get('clahe_params', {}).get('clip_limit', 3.0),
            'contrast_before': preprocessing_stats.get('clahe_enhancement', {}).get('contrast_rms', {}).get('before_mean', 'N/A'),
            'contrast_after': preprocessing_stats.get('clahe_enhancement', {}).get('contrast_rms', {}).get('after_mean', 'N/A'),
            'contrast_improvement': preprocessing_stats.get('clahe_enhancement', {}).get('contrast_rms', {}).get('improvement_pct', 'N/A'),
            'entropy_before': preprocessing_stats.get('clahe_enhancement', {}).get('entropy', {}).get('before_mean', 'N/A'),
            'entropy_after': preprocessing_stats.get('clahe_enhancement', {}).get('entropy', {}).get('after_mean', 'N/A'),
            'mask_coverage': (
                preprocessing_stats.get('segmentation', {}).get('mask_coverage', {}).get('mean', 'N/A')
                if isinstance(preprocessing_stats.get('segmentation', {}).get('mask_coverage'), dict)
                else 'N/A'
            ),
        },
        
        'keypoints': {
            'method': keypoint_stats.get('method', 'SuperPoint'),
            'descriptor_dim': 256,
            'mean_keypoints': keypoint_stats.get('keypoint_counts', {}).get('mean', 'N/A'),
            'std_keypoints': keypoint_stats.get('keypoint_counts', {}).get('std', 'N/A'),
            'total_keypoints': keypoint_stats.get('keypoint_counts', {}).get('total', 'N/A'),
            'spatial_coverage': keypoint_stats.get('spatial_coverage', {}).get('mean', 'N/A'),
        },
        
        'graph': {
            'knn_k': graph_stats.get('knn_k', 12),
            'mean_nodes': graph_stats.get('nodes_per_graph', {}).get('mean', 'N/A'),
            'mean_edges': graph_stats.get('edges_per_graph', {}).get('mean', 'N/A'),
            'avg_degree': graph_stats.get('avg_degree', {}).get('mean', 'N/A'),
            'graph_density': graph_stats.get('graph_density', {}).get('mean', 'N/A'),
        },
        
        'model': {
            'architecture': 'CattleGNN (EdgeConv + TRM)',
            'edge_conv_layers': 3,
            'edge_conv_dims': '256 -> 256 -> 512',
            'trm_heads': 4,
            'trm_layers': 2,
            'embedding_dim': 256,
            'total_params': training_stats.get('model_summary', {}).get('total_parameters', 'N/A'),
            'trainable_params': training_stats.get('model_summary', {}).get('trainable_parameters', 'N/A'),
        },
        
        'training': {
            'loss': 'Triplet + Cross-Entropy',
            'triplet_margin': 0.5,
            'mining': 'Online Hard Negative',
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'scheduler': 'Cosine Annealing',
            'epochs_trained': training_stats.get('total_epochs', 'N/A'),
            'training_time': training_stats.get('training_time_seconds', 'N/A'),
            'best_val_accuracy': training_stats.get('best_val_accuracy', 'N/A'),
        },
        
        'results': {
            'rank_1': eval_results.get('summary', {}).get('rank_1_accuracy', 0),
            'rank_5': eval_results.get('summary', {}).get('rank_5_accuracy', 0),
            'rank_10': eval_results.get('summary', {}).get('rank_10_accuracy', 0),
            'eer': eval_results.get('summary', {}).get('eer', 0),
            'roc_auc': eval_results.get('summary', {}).get('roc_auc', 0),
            'tar_far_001': eval_results.get('summary', {}).get('tar_at_far_0.01', 0),
            'tar_far_0001': eval_results.get('summary', {}).get('tar_at_far_0.001', 0),
            'd_prime': eval_results.get('score_statistics', {}).get('d_prime', 0),
        },
    }
    
    # Save comprehensive stats
    paper_stats_path = os.path.join(stats_dir, "paper_statistics.json")
    save_stats(paper_stats, paper_stats_path)
    
    # Generate LaTeX tables
    latex_tables = generate_latex_tables(paper_stats)
    latex_path = os.path.join(stats_dir, "paper_tables.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_tables)
    
    # Print comprehensive summary
    print(f"\n{'=' * 70}")
    print("COMPREHENSIVE PAPER STATISTICS")
    print(f"{'=' * 70}")
    
    for section, data in paper_stats.items():
        if section == 'generated_at':
            continue
        print(f"\n  [{section.upper()}]")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            print(f"    {data}")
    
    print(f"\n{'=' * 70}")
    print(f"  Paper statistics: {paper_stats_path}")
    print(f"  LaTeX tables:     {latex_path}")
    print(f"{'=' * 70}")
    
    print(f"\n[SUCCESS] [OK] All paper statistics compiled!")
    return paper_stats


if __name__ == "__main__":
    main()
