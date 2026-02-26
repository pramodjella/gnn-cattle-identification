"""
Verification Pipeline
======================
End-to-end verification: extract embedding → match → accept/reject.
Includes Hungarian algorithm for optimal assignment and threshold calibration.
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from .lightglue_matcher import GraphMatcher


class VerificationPipeline:
    """
    End-to-end biometric verification pipeline.
    """
    
    def __init__(self, model, matcher=None, device='cpu'):
        """
        Args:
            model: Trained CattleGNN model
            matcher: GraphMatcher instance
            device: Computation device
        """
        self.model = model
        self.matcher = matcher or GraphMatcher()
        self.device = device
        self.gallery_embeddings = None
        self.gallery_labels = None
    
    def enroll_gallery(self, gallery_loader):
        """
        Extract embeddings for all gallery images.
        
        Args:
            gallery_loader: DataLoader for gallery set
            
        Returns:
            num_enrolled: Number of enrolled identities
        """
        self.model.eval()
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch in gallery_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                embeddings.append(output['embedding'].cpu())
                labels.append(batch.y.cpu())
        
        self.gallery_embeddings = torch.cat(embeddings)
        self.gallery_labels = torch.cat(labels)
        
        num_identities = len(torch.unique(self.gallery_labels))
        print(f"[INFO] Enrolled {len(self.gallery_embeddings)} samples, {num_identities} identities")
        
        return num_identities
    
    def verify(self, query_data, top_k=5):
        """
        Verify a query against the gallery.
        
        Args:
            query_data: PyG Data object for query
            top_k: Number of top matches
            
        Returns:
            match_result dict
        """
        assert self.gallery_embeddings is not None, "Gallery not enrolled"
        
        self.model.eval()
        with torch.no_grad():
            query_data = query_data.to(self.device)
            output = self.model(query_data)
            query_embedding = output['embedding'].cpu().squeeze(0)
        
        result = self.matcher.identify(
            query_embedding, self.gallery_embeddings,
            self.gallery_labels.tolist(), top_k=top_k
        )
        
        return result
    
    def calibrate_threshold(self, val_loader, target_far=0.01):
        """
        Calibrate acceptance threshold on validation set.
        
        Args:
            val_loader: Validation DataLoader
            target_far: Target False Accept Rate
            
        Returns:
            optimal_threshold: Calibrated threshold
        """
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                all_embeddings.append(output['embedding'].cpu())
                all_labels.append(batch.y.cpu())
        
        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)
        
        # Compute all pairwise similarities
        sim_matrix = torch.mm(all_embeddings, all_embeddings.t())
        
        # Get genuine and impostor scores
        genuine_scores = []
        impostor_scores = []
        
        n = len(all_labels)
        for i in range(n):
            for j in range(i + 1, n):
                score = sim_matrix[i, j].item()
                if all_labels[i] == all_labels[j]:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)
        
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        # Find threshold at target FAR
        thresholds = np.linspace(0, 1, 1000)
        best_threshold = 0.5
        best_far_diff = float('inf')
        
        for t in thresholds:
            far = np.mean(impostor_scores >= t)
            diff = abs(far - target_far)
            if diff < best_far_diff:
                best_far_diff = diff
                best_threshold = t
        
        # Set the calibrated threshold
        self.matcher.threshold = best_threshold
        
        actual_far = np.mean(impostor_scores >= best_threshold)
        actual_tar = np.mean(genuine_scores >= best_threshold)
        
        print(f"[INFO] Calibrated threshold: {best_threshold:.4f}")
        print(f"  TAR: {actual_tar:.4f}, FAR: {actual_far:.4f}")
        
        return best_threshold
    
    def hungarian_matching(self, cost_matrix):
        """
        Optimal assignment using Hungarian algorithm.
        
        Args:
            cost_matrix: (N, M) cost matrix (lower = better match)
            
        Returns:
            row_indices, col_indices: Optimal assignment
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind
