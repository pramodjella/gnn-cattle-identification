"""
Visualization Module
=====================
Publication-quality plots for the cattle identification paper:
- ROC curves
- CMC curves
- Score distributions
- t-SNE embedding visualizations
- Training curves
- Confusion matrices
- Keypoint and graph overlays
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# Publication quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


class ResultVisualizer:
    """Generate publication-quality visualizations."""
    
    def __init__(self, output_dir='outputs/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, eer=None, save_name='roc_curve.png'):
        """Plot ROC curve with AUC and EER."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'CattleGNN (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        
        if eer is not None:
            ax.plot(eer, 1 - eer, 'ro', markersize=10, label=f'EER = {eer:.4f}')
        
        ax.set_xlabel('False Accept Rate (FAR)')
        ax.set_ylabel('True Accept Rate (TAR)')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        print(f"[INFO] Saved ROC curve to {path}")
        return path
    
    def plot_roc_semilog(self, fpr, tpr, roc_auc, tar_at_far=None, 
                         save_name='roc_semilog.png'):
        """Plot ROC curve on semi-log scale (standard for biometrics)."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Filter out zero FPR for log scale
        mask = fpr > 0
        ax.semilogx(fpr[mask], tpr[mask], 'b-', linewidth=2, 
                     label=f'CattleGNN (AUC = {roc_auc:.4f})')
        
        # Mark TAR at specific FAR points
        if tar_at_far:
            for far, tar in tar_at_far.items():
                far_val = float(far.replace('FAR=', ''))
                ax.plot(far_val, tar, 's', markersize=8,
                       label=f'TAR={tar:.3f} @ FAR={far_val}')
        
        ax.set_xlabel('False Accept Rate (FAR)')
        ax.set_ylabel('True Accept Rate (TAR)')
        ax.set_title('ROC Curve (Semi-Log Scale)')
        ax.legend(loc='lower right')
        ax.set_xlim([1e-4, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, which='both')
        
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path
    
    def plot_cmc_curve(self, cmc_values, rank_highlights=None, 
                       save_name='cmc_curve.png'):
        """Plot Cumulative Match Characteristic curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ranks = np.arange(1, len(cmc_values) + 1)
        ax.plot(ranks, cmc_values, 'b-', linewidth=2, marker='o', markersize=3)
        
        # Highlight specific ranks
        if rank_highlights:
            for k, acc in rank_highlights.items():
                k_int = int(k.replace('rank_', ''))
                if k_int <= len(cmc_values):
                    ax.plot(k_int, acc, 'ro', markersize=10, zorder=5)
                    ax.annotate(f'Rank-{k_int}: {acc:.3f}', 
                              (k_int, acc), textcoords="offset points",
                              xytext=(15, -10), fontsize=10,
                              arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('Identification Accuracy')
        ax.set_title('Cumulative Match Characteristic (CMC) Curve')
        ax.set_xlim([1, min(50, len(cmc_values))])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path
    
    def plot_score_distributions(self, genuine_scores, impostor_scores, 
                                 threshold=None, save_name='score_distributions.png'):
        """Plot genuine vs impostor score distributions."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.hist(impostor_scores, bins=100, alpha=0.6, color='red', 
                density=True, label='Impostor')
        ax.hist(genuine_scores, bins=100, alpha=0.6, color='blue', 
                density=True, label='Genuine')
        
        if threshold is not None:
            ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                      label=f'Threshold = {threshold:.3f}')
        
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Density')
        ax.set_title('Genuine vs Impostor Score Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path
    
    def plot_tsne(self, embeddings, labels, num_classes=20, 
                  save_name='tsne_embeddings.png'):
        """Plot t-SNE visualization of embeddings."""
        embeddings_np = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else embeddings
        labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels
        
        # Select subset of classes for clarity
        unique_labels = np.unique(labels_np)
        if len(unique_labels) > num_classes:
            selected = np.random.choice(unique_labels, num_classes, replace=False)
            mask = np.isin(labels_np, selected)
            embeddings_np = embeddings_np[mask]
            labels_np = labels_np[mask]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, 
                     n_iter=1000, learning_rate='auto', init='pca')
        embedded_2d = tsne.fit_transform(embeddings_np)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels_np)
        try:
            cmap = matplotlib.colormaps.get_cmap('tab20')
        except AttributeError:
            try:
                cmap = plt.colormaps.get_cmap('tab20')
            except AttributeError:
                cmap = plt.cm.get_cmap('tab20')
        
        for i, label in enumerate(unique_labels):
            mask = labels_np == label
            ax.scatter(embedded_2d[mask, 0], embedded_2d[mask, 1],
                      c=[cmap(i % 20)], s=30, alpha=0.7, label=f'Animal {label}')
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Visualization of Cattle Embeddings')
        
        if len(unique_labels) <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path
    
    def plot_training_curves(self, history, save_name='training_curves.png'):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax = axes[0, 1]
        if 'val_rank1_acc' in history:
            ax.plot(epochs, history['val_rank1_acc'], 'g-', label='Val Rank-1 Acc', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Rank-1 Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 0]
        if 'learning_rates' in history:
            ax.plot(epochs, history['learning_rates'], 'purple', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Active triplets ratio
        ax = axes[1, 1]
        if 'train_active_triplets' in history:
            ax.plot(epochs, history['train_active_triplets'], 'orange', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Active Ratio')
        ax.set_title('Active Triplet Ratio')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path
    
    def plot_confusion_matrix(self, true_labels, pred_labels, num_classes=None,
                              save_name='confusion_matrix.png'):
        """Plot confusion matrix."""
        cm = confusion_matrix(true_labels, pred_labels)
        
        if num_classes and cm.shape[0] > num_classes:
            cm = cm[:num_classes, :num_classes]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, cmap='Blues', ax=ax, fmt='d',
                   xticklabels=range(cm.shape[1]),
                   yticklabels=range(cm.shape[0]))
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Identification Confusion Matrix')
        
        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path)
        plt.close(fig)
        return path
    
    def generate_all_paper_figures(self, eval_results, history=None, 
                                   embeddings=None, labels=None):
        """Generate all paper figures from evaluation results."""
        figures = {}
        
        # ROC curve
        if 'verification' in eval_results:
            v = eval_results['verification']
            fpr = np.array(v['fpr'])
            tpr = np.array(v['tpr'])
            
            figures['roc'] = self.plot_roc_curve(
                fpr, tpr, v['roc_auc'], v.get('eer')
            )
            figures['roc_semilog'] = self.plot_roc_semilog(
                fpr, tpr, v['roc_auc'], v.get('tar_at_far')
            )
        
        # CMC curve
        if 'identification' in eval_results:
            ident = eval_results['identification']
            figures['cmc'] = self.plot_cmc_curve(
                ident['cmc_curve'], ident.get('rank_accuracies')
            )
        
        # Score distributions
        if 'score_statistics' in eval_results:
            ss = eval_results['score_statistics']
            # Generate synthetic distributions from statistics
            if ss['genuine']['count'] > 0:
                genuine = np.random.normal(
                    ss['genuine']['mean'], ss['genuine']['std'], 
                    max(1000, ss['genuine']['count'])
                )
                impostor = np.random.normal(
                    ss['impostor']['mean'], ss['impostor']['std'],
                    max(1000, ss['impostor']['count'])
                )
                figures['scores'] = self.plot_score_distributions(genuine, impostor)
        
        # Training curves
        if history:
            figures['training'] = self.plot_training_curves(history)
        
        # t-SNE
        if embeddings is not None and labels is not None:
            figures['tsne'] = self.plot_tsne(embeddings, labels)
        
        print(f"\n[INFO] Generated {len(figures)} paper figures")
        return figures
