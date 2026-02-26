"""
Script 06: Evaluate Model
============================
Full evaluation of the trained CattleGNN model:
- Rank-1/5/10 identification accuracy
- TAR at FAR 0.1%, 1%, 10%
- EER and ROC AUC
- CMC curves
- Score distributions
- t-SNE embeddings
- All paper-quality figures

Input:  outputs/checkpoints/best_model.pt
Output: outputs/results/, outputs/figures/, outputs/stats/
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, setup_logging, set_seed, Timer
from src.models.gnn_model import CattleGNN
from src.evaluation.metrics import BiometricMetrics
from src.evaluation.visualization import ResultVisualizer
from src.training.dataset import create_data_loaders


def extract_embeddings(model, data_loader, device):
    """Extract embeddings from all data in a loader."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)
            all_embeddings.append(output['embedding'].cpu())
            all_labels.append(batch.y.cpu())
    
    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    
    return embeddings, labels


def main():
    print("=" * 70)
    print("PHASE 7-8: Evaluation & Paper Statistics")
    print("=" * 70)
    
    config = load_config()
    set_seed(config['project']['seed'])
    logger = setup_logging(config['outputs']['log_dir'], "06_evaluate")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Paths
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    checkpoint_dir = str(PROJECT_ROOT / config['training']['checkpoint_dir'])
    stats_dir = str(PROJECT_ROOT / config['outputs']['stats_dir'])
    figure_dir = str(PROJECT_ROOT / config['outputs']['figure_dir'])
    results_dir = str(PROJECT_ROOT / config['outputs']['results_dir'])
    ensure_dirs(stats_dir, figure_dir, results_dir)
    
    # Load label mapping
    label_map_path = os.path.join(graph_dir, "label_mapping.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_mapping = json.load(f)
        num_classes = len(label_mapping)
    else:
        num_classes = 268  # Default expected
    
    # Initialize model
    print("\n--- Loading Model ---")
    model = CattleGNN(config=config)
    model.set_num_classes(num_classes)
    
    # Load best checkpoint
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded best model from epoch {checkpoint['epoch']}")
        print(f"[INFO] Best val accuracy: {checkpoint['best_val_acc']:.4f}")
    else:
        # Try final model
        final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
        if os.path.exists(final_model_path):
            checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Loaded final model from epoch {checkpoint['epoch']}")
        else:
            print("[WARNING] No checkpoint found. Using untrained model.")
            checkpoint = {}
    
    model = model.to(device)
    
    # Create data loaders
    print("\n--- Loading Test Data ---")
    loaders = create_data_loaders(graph_dir, config, augment_train=False)
    
    # Extract embeddings for each split
    print("\n--- Extracting Embeddings ---")
    split_embeddings = {}
    
    with Timer("Embedding extraction") as timer:
        for split_name, loader in loaders.items():
            emb, lab = extract_embeddings(model, loader, device)
            split_embeddings[split_name] = {'embeddings': emb, 'labels': lab}
            print(f"  {split_name}: {len(emb)} embeddings extracted")
    
    # Evaluate on test set
    print("\n--- Computing Metrics ---")
    eval_config = config.get('evaluation', {})
    metrics = BiometricMetrics(
        far_points=eval_config.get('far_points', [0.001, 0.01, 0.1]),
        rank_k=eval_config.get('rank_k', [1, 5, 10]),
    )
    
    # Primary evaluation on test set
    test_data = split_embeddings.get('test', split_embeddings.get('val', split_embeddings.get('train')))
    
    with Timer("Metrics computation") as timer:
        eval_results = metrics.compute_all_metrics(
            test_data['embeddings'], test_data['labels']
        )
    
    # Print summary
    metrics.print_summary(eval_results)
    
    # Save full results
    results_path = os.path.join(results_dir, "evaluation_results.json")
    
    # Make serializable
    serializable_results = json.loads(
        json.dumps(eval_results, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    )
    save_stats(serializable_results, results_path)
    
    # Also evaluate on validation set if available
    if 'val' in split_embeddings and 'test' in split_embeddings:
        val_results = metrics.compute_all_metrics(
            split_embeddings['val']['embeddings'],
            split_embeddings['val']['labels']
        )
        val_results_path = os.path.join(results_dir, "validation_results.json")
        save_stats(
            json.loads(json.dumps(val_results, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))),
            val_results_path
        )
    
    # Generate paper figures
    print("\n--- Generating Paper Figures ---")
    visualizer = ResultVisualizer(output_dir=figure_dir)
    
    # Load training history if available
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    history = None
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    
    figures = visualizer.generate_all_paper_figures(
        eval_results,
        history=history,
        embeddings=test_data['embeddings'],
        labels=test_data['labels'],
    )
    
    # Save evaluation statistics summary
    eval_stats = {
        'evaluation_time_seconds': timer.elapsed,
        'test_set_size': len(test_data['embeddings']),
        'num_test_classes': len(torch.unique(test_data['labels'])),
        'metrics_summary': eval_results['summary'],
        'score_statistics': eval_results['score_statistics'],
        'figures_generated': list(figures.keys()),
    }
    
    stats_path = os.path.join(stats_dir, "evaluation_stats.json")
    save_stats(eval_stats, stats_path)
    
    print(f"\n{'=' * 70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Results:  {results_path}")
    print(f"  Figures:  {figure_dir}")
    print(f"  Stats:    {stats_path}")
    print(f"{'=' * 70}")
    
    return eval_results


if __name__ == "__main__":
    main()
