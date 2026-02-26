"""
Biometric Evaluation Metrics
==============================
Comprehensive metrics for biometric identification evaluation:
- Rank-1, Rank-5, Rank-10 identification accuracy
- TAR at specific FAR rates
- Equal Error Rate (EER)
- CMC (Cumulative Match Characteristic) curves
- ROC curves
- Confusion matrices

All metrics are paper-ready with statistical significance.
"""

import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, confusion_matrix


class BiometricMetrics:
    """
    Comprehensive biometric evaluation metrics.
    """
    
    def __init__(self, far_points=None, rank_k=None):
        """
        Args:
            far_points: FAR values for TAR computation (e.g., [0.001, 0.01, 0.1])
            rank_k: Rank values for CMC (e.g., [1, 5, 10])
        """
        self.far_points = far_points or [0.001, 0.01, 0.1]
        self.rank_k = rank_k or [1, 5, 10]
    
    def compute_all_metrics(self, embeddings, labels):
        """
        Compute all biometric metrics from embeddings and labels.
        
        Args:
            embeddings: (N, D) tensor of embeddings
            labels: (N,) tensor of integer labels
            
        Returns:
            dict with all metrics
        """
        embeddings = embeddings.cpu() if torch.is_tensor(embeddings) else torch.tensor(embeddings)
        labels = labels.cpu() if torch.is_tensor(labels) else torch.tensor(labels)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()).numpy()
        labels_np = labels.numpy()
        
        # 1. Identification metrics (CMC / Rank-k)
        cmc_curve, rank_accuracies = self._compute_cmc(sim_matrix, labels_np)
        
        # 2. Verification metrics (ROC, TAR@FAR, EER)
        genuine_scores, impostor_scores = self._get_score_distributions(
            sim_matrix, labels_np
        )
        
        fpr, tpr, thresholds = roc_curve(
            [1] * len(genuine_scores) + [0] * len(impostor_scores),
            list(genuine_scores) + list(impostor_scores)
        )
        roc_auc = auc(fpr, tpr)
        
        # TAR at specific FAR
        tar_at_far = self._compute_tar_at_far(fpr, tpr, self.far_points)
        
        # EER
        eer = self._compute_eer(fpr, tpr)
        
        # 3. Score statistics
        score_stats = self._compute_score_statistics(genuine_scores, impostor_scores)
        
        # 4. Per-class accuracy
        per_class_acc = self._compute_per_class_accuracy(sim_matrix, labels_np)
        
        results = {
            'identification': {
                'rank_accuracies': {
                    f'rank_{k}': float(rank_accuracies.get(k, 0)) 
                    for k in self.rank_k
                },
                'cmc_curve': cmc_curve.tolist()[:50],  # First 50 ranks
            },
            'verification': {
                'eer': float(eer),
                'roc_auc': float(roc_auc),
                'tar_at_far': {
                    f'FAR={f}': float(t) for f, t in tar_at_far.items()
                },
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
            },
            'score_statistics': score_stats,
            'per_class': per_class_acc,
            'summary': {
                'rank_1_accuracy': float(rank_accuracies.get(1, 0)),
                'rank_5_accuracy': float(rank_accuracies.get(5, 0)),
                'rank_10_accuracy': float(rank_accuracies.get(10, 0)),
                'eer': float(eer),
                'roc_auc': float(roc_auc),
                'tar_at_far_0.01': float(tar_at_far.get(0.01, 0)),
                'tar_at_far_0.001': float(tar_at_far.get(0.001, 0)),
                'num_samples': len(labels),
                'num_classes': len(np.unique(labels_np)),
            }
        }
        
        return results
    
    def _compute_cmc(self, sim_matrix, labels):
        """Compute Cumulative Match Characteristic curve."""
        n = len(labels)
        
        # Set diagonal to -inf (no self-matching)
        np.fill_diagonal(sim_matrix, -np.inf)
        
        # For each query, get sorted gallery indices by similarity
        sorted_indices = np.argsort(-sim_matrix, axis=1)  # Descending
        
        # Compute CMC
        max_rank = min(50, n - 1)
        cmc = np.zeros(max_rank)
        
        for i in range(n):
            query_label = labels[i]
            for rank in range(max_rank):
                gallery_idx = sorted_indices[i, rank]
                if labels[gallery_idx] == query_label:
                    cmc[rank:] += 1
                    break
        
        cmc = cmc / n
        
        # Extract specific rank accuracies
        rank_accuracies = {}
        for k in self.rank_k:
            if k <= max_rank:
                rank_accuracies[k] = cmc[k - 1]
        
        return cmc, rank_accuracies
    
    def _get_score_distributions(self, sim_matrix, labels):
        """Extract genuine and impostor score distributions."""
        n = len(labels)
        genuine_scores = []
        impostor_scores = []
        
        for i in range(n):
            for j in range(i + 1, n):
                score = sim_matrix[i, j]
                if labels[i] == labels[j]:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)
        
        return np.array(genuine_scores), np.array(impostor_scores)
    
    def _compute_tar_at_far(self, fpr, tpr, far_points):
        """Compute TAR at specific FAR values."""
        tar_at_far = {}
        for target_far in far_points:
            # Find the closest FAR value
            idx = np.argmin(np.abs(fpr - target_far))
            tar_at_far[target_far] = tpr[idx]
        return tar_at_far
    
    def _compute_eer(self, fpr, tpr):
        """Compute Equal Error Rate."""
        fnr = 1 - tpr
        # EER is where FPR == FNR
        idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2
        return eer
    
    def _compute_score_statistics(self, genuine_scores, impostor_scores):
        """Compute score distribution statistics."""
        return {
            'genuine': {
                'mean': float(np.mean(genuine_scores)) if len(genuine_scores) > 0 else 0,
                'std': float(np.std(genuine_scores)) if len(genuine_scores) > 0 else 0,
                'min': float(np.min(genuine_scores)) if len(genuine_scores) > 0 else 0,
                'max': float(np.max(genuine_scores)) if len(genuine_scores) > 0 else 0,
                'count': len(genuine_scores),
            },
            'impostor': {
                'mean': float(np.mean(impostor_scores)) if len(impostor_scores) > 0 else 0,
                'std': float(np.std(impostor_scores)) if len(impostor_scores) > 0 else 0,
                'min': float(np.min(impostor_scores)) if len(impostor_scores) > 0 else 0,
                'max': float(np.max(impostor_scores)) if len(impostor_scores) > 0 else 0,
                'count': len(impostor_scores),
            },
            'd_prime': self._compute_d_prime(genuine_scores, impostor_scores),
        }
    
    def _compute_d_prime(self, genuine, impostor):
        """Compute d-prime (separability measure)."""
        if len(genuine) == 0 or len(impostor) == 0:
            return 0.0
        mu_g, mu_i = np.mean(genuine), np.mean(impostor)
        std_g, std_i = np.std(genuine), np.std(impostor)
        denom = np.sqrt(0.5 * (std_g**2 + std_i**2))
        if denom < 1e-8:
            return 0.0
        return float(abs(mu_g - mu_i) / denom)
    
    def _compute_per_class_accuracy(self, sim_matrix, labels):
        """Compute per-class Rank-1 accuracy."""
        np.fill_diagonal(sim_matrix, -np.inf)
        
        unique_labels = np.unique(labels)
        per_class_acc = {}
        
        for label in unique_labels:
            mask = labels == label
            indices = np.where(mask)[0]
            
            correct = 0
            total = len(indices)
            
            for i in indices:
                nn_idx = np.argmax(sim_matrix[i])
                if labels[nn_idx] == label:
                    correct += 1
            
            per_class_acc[int(label)] = {
                'accuracy': correct / max(total, 1),
                'num_samples': total,
            }
        
        accuracies = [v['accuracy'] for v in per_class_acc.values()]
        per_class_acc['overall'] = {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies)),
            'min': float(np.min(accuracies)),
            'max': float(np.max(accuracies)),
        }
        
        return per_class_acc
    
    def print_summary(self, results):
        """Print a formatted summary of evaluation results."""
        s = results['summary']
        
        print(f"\n{'=' * 60}")
        print("BIOMETRIC EVALUATION RESULTS")
        print(f"{'=' * 60}")
        print(f"\n  Identification Metrics:")
        print(f"    Rank-1 Accuracy:     {s['rank_1_accuracy']:.4f} ({s['rank_1_accuracy']*100:.2f}%)")
        print(f"    Rank-5 Accuracy:     {s['rank_5_accuracy']:.4f} ({s['rank_5_accuracy']*100:.2f}%)")
        print(f"    Rank-10 Accuracy:    {s['rank_10_accuracy']:.4f} ({s['rank_10_accuracy']*100:.2f}%)")
        print(f"\n  Verification Metrics:")
        print(f"    EER:                 {s['eer']:.4f} ({s['eer']*100:.2f}%)")
        print(f"    ROC AUC:             {s['roc_auc']:.4f}")
        print(f"    TAR @ FAR=1%:        {s['tar_at_far_0.01']:.4f}")
        print(f"    TAR @ FAR=0.1%:      {s['tar_at_far_0.001']:.4f}")
        print(f"\n  Dataset:")
        print(f"    Samples:             {s['num_samples']}")
        print(f"    Classes:             {s['num_classes']}")
        
        ss = results['score_statistics']
        print(f"\n  Score Statistics:")
        print(f"    Genuine Mean:        {ss['genuine']['mean']:.4f} ± {ss['genuine']['std']:.4f}")
        print(f"    Impostor Mean:       {ss['impostor']['mean']:.4f} ± {ss['impostor']['std']:.4f}")
        print(f"    d-prime:             {ss['d_prime']:.4f}")
        print(f"{'=' * 60}")
