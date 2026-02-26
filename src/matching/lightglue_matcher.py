"""
Graph Matching Module
======================
Implements graph-level and keypoint-level matching for cattle identification.

Graph-level: Cosine similarity between GNN embeddings
Keypoint-level: Cross-attention matching inspired by LightGlue
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


class GraphMatcher:
    """
    Matcher for cattle identification using GNN embeddings.
    
    Two matching strategies:
    1. Graph-level: Compare graph embeddings directly (fast)
    2. Keypoint-level: Match individual keypoint descriptors (detailed)
    """
    
    def __init__(self, method='cosine', threshold=0.7):
        """
        Args:
            method: Matching method ('cosine', 'euclidean')
            threshold: Similarity threshold for accept/reject
        """
        self.method = method
        self.threshold = threshold
    
    def compute_similarity_matrix(self, query_embeddings, gallery_embeddings):
        """
        Compute similarity matrix between query and gallery embeddings.
        
        Args:
            query_embeddings: (Q, D) query embeddings
            gallery_embeddings: (G, D) gallery embeddings
            
        Returns:
            sim_matrix: (Q, G) similarity scores
        """
        if self.method == 'cosine':
            # Both should be L2-normalized already
            sim_matrix = torch.mm(query_embeddings, gallery_embeddings.t())
        elif self.method == 'euclidean':
            # Convert to similarity: 1 / (1 + distance)
            diffs = query_embeddings.unsqueeze(1) - gallery_embeddings.unsqueeze(0)
            distances = torch.norm(diffs, dim=-1)
            sim_matrix = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return sim_matrix
    
    def identify(self, query_embedding, gallery_embeddings, gallery_labels, top_k=5):
        """
        Identify a query against a gallery.
        
        Args:
            query_embedding: (D,) single query embedding
            gallery_embeddings: (G, D) gallery embeddings
            gallery_labels: (G,) gallery labels
            top_k: Number of top matches to return
            
        Returns:
            dict with top-k matches, scores, and predicted identity
        """
        query_embedding = query_embedding.unsqueeze(0)
        sim_matrix = self.compute_similarity_matrix(query_embedding, gallery_embeddings)
        similarities = sim_matrix.squeeze(0)
        
        # Get top-k matches
        top_k = min(top_k, len(gallery_labels))
        top_scores, top_indices = torch.topk(similarities, top_k)
        
        top_labels = [gallery_labels[i.item()] for i in top_indices]
        
        return {
            'predicted_label': top_labels[0],
            'predicted_score': top_scores[0].item(),
            'top_k_labels': top_labels,
            'top_k_scores': top_scores.tolist(),
            'accepted': top_scores[0].item() >= self.threshold,
        }
    
    def match_keypoints(self, desc_a, desc_b, pos_a=None, pos_b=None, 
                        ratio_threshold=0.8):
        """
        Keypoint-level matching using mutual nearest neighbors.
        Inspired by LightGlue's cross-attention approach.
        
        Args:
            desc_a: (N, D) descriptors from image A
            desc_b: (M, D) descriptors from image B
            pos_a: (N, 2) positions from image A (optional)
            pos_b: (M, 2) positions from image B (optional)
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            dict with matches, scores, and quality metrics
        """
        # Compute cross-similarity
        sim = torch.mm(
            F.normalize(desc_a, dim=-1), 
            F.normalize(desc_b, dim=-1).t()
        )
        
        # Forward matches: A -> B
        fwd_scores, fwd_matches = sim.max(dim=1)
        
        # Backward matches: B -> A  
        bwd_scores, bwd_matches = sim.max(dim=0)
        
        # Mutual nearest neighbor filter
        mutual_matches = []
        match_scores = []
        
        for i in range(len(desc_a)):
            j = fwd_matches[i].item()
            if bwd_matches[j].item() == i:
                mutual_matches.append((i, j))
                match_scores.append(fwd_scores[i].item())
        
        # Ratio test (if enough matches)
        if len(mutual_matches) > 2:
            filtered_matches = []
            filtered_scores = []
            
            for idx, (i, j) in enumerate(mutual_matches):
                row_scores = sim[i].sort(descending=True)[0]
                if len(row_scores) >= 2:
                    ratio = row_scores[0] / (row_scores[1] + 1e-8)
                    if ratio > 1.0 / ratio_threshold:
                        filtered_matches.append((i, j))
                        filtered_scores.append(match_scores[idx])
                else:
                    filtered_matches.append((i, j))
                    filtered_scores.append(match_scores[idx])
            
            mutual_matches = filtered_matches
            match_scores = filtered_scores
        
        result = {
            'matches': mutual_matches,
            'scores': match_scores,
            'num_matches': len(mutual_matches),
            'avg_score': float(np.mean(match_scores)) if match_scores else 0.0,
            'match_ratio': len(mutual_matches) / max(min(len(desc_a), len(desc_b)), 1),
        }
        
        # Geometric consistency check (if positions provided)
        if pos_a is not None and pos_b is not None and len(mutual_matches) >= 4:
            inlier_ratio = self._geometric_verification(
                pos_a, pos_b, mutual_matches
            )
            result['geometric_inlier_ratio'] = inlier_ratio
        
        return result
    
    def _geometric_verification(self, pos_a, pos_b, matches):
        """
        Simple geometric verification using position consistency.
        """
        if len(matches) < 4:
            return 0.0
        
        # Extract matched positions
        pts_a = np.array([pos_a[m[0]].numpy() if torch.is_tensor(pos_a) else pos_a[m[0]] for m in matches])
        pts_b = np.array([pos_b[m[1]].numpy() if torch.is_tensor(pos_b) else pos_b[m[1]] for m in matches])
        
        # Compute pairwise relative positions
        diffs = pts_b - pts_a
        
        # Check consistency: all displacement vectors should be similar
        mean_diff = np.mean(diffs, axis=0)
        residuals = np.linalg.norm(diffs - mean_diff, axis=1)
        threshold = np.median(residuals) * 2 + 1e-6
        
        inliers = np.sum(residuals < threshold)
        return float(inliers / len(matches))
