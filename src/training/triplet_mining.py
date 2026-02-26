"""
Triplet Mining Module
=====================
Implements online triplet mining strategies for metric learning.
"""

import torch
import numpy as np
from collections import defaultdict


class TripletMiner:
    """
    Online triplet miner for generating informative training pairs.
    
    Strategies:
    - hard: Selects hardest positive and hardest negative per anchor
    - semi-hard: Selects semi-hard negatives (farther than positive, within margin)
    - all: Returns all valid triplets
    """
    
    def __init__(self, margin=0.5, mining_type='hard'):
        self.margin = margin
        self.mining_type = mining_type
        self.stats_history = []
    
    def mine_triplets(self, embeddings, labels):
        """
        Mine triplets from a batch of embeddings.
        
        Args:
            embeddings: (B, D) embeddings
            labels: (B,) labels
            
        Returns:
            anchor_idx, positive_idx, negative_idx: triplet indices
        """
        dist_matrix = self._pairwise_distances(embeddings)
        labels = labels.view(-1)
        
        anchor_idx = []
        positive_idx = []
        negative_idx = []
        
        for i in range(len(labels)):
            # Find positives (same class, different sample)
            pos_mask = (labels == labels[i]) & (torch.arange(len(labels), device=labels.device) != i)
            neg_mask = (labels != labels[i])
            
            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            if self.mining_type == 'hard':
                # Hardest positive
                p_idx = pos_indices[dist_matrix[i, pos_indices].argmax()]
                # Hardest negative
                n_idx = neg_indices[dist_matrix[i, neg_indices].argmin()]
            elif self.mining_type == 'semi-hard':
                p_idx = pos_indices[dist_matrix[i, pos_indices].argmax()]
                p_dist = dist_matrix[i, p_idx]
                
                # Semi-hard negatives
                neg_dists = dist_matrix[i, neg_indices]
                semi_hard = (neg_dists > p_dist) & (neg_dists < p_dist + self.margin)
                
                if semi_hard.sum() > 0:
                    semi_neg_indices = neg_indices[semi_hard]
                    n_idx = semi_neg_indices[dist_matrix[i, semi_neg_indices].argmax()]
                else:
                    n_idx = neg_indices[neg_dists.argmin()]
            else:  # random
                p_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
                n_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
            
            anchor_idx.append(i)
            positive_idx.append(p_idx.item())
            negative_idx.append(n_idx.item())
        
        return anchor_idx, positive_idx, negative_idx
    
    def _pairwise_distances(self, embeddings):
        """Compute pairwise distances."""
        dot = torch.mm(embeddings, embeddings.t())
        sq_norms = torch.diag(dot)
        distances = sq_norms.unsqueeze(0) - 2 * dot + sq_norms.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)
        return torch.sqrt(distances + 1e-8)
