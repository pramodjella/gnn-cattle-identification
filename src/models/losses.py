"""
Loss Functions for Metric Learning
====================================
Implements triplet loss with online hard negative mining and
optional ArcFace loss for cattle identification.

Triplet Loss: L = max(0, d(a,p) - d(a,n) + margin)
where (a,p) is a positive pair (same animal) and (a,n) is negative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TripletLossWithMining(nn.Module):
    """
    Triplet loss with online hard/semi-hard negative mining.
    
    Mining strategies:
    - 'hard': Hardest positive + hardest negative per anchor
    - 'semi-hard': Hardest negative that is farther than the positive
    - 'random': Random positive + random negative
    """
    
    def __init__(self, margin=0.5, mining_type='hard'):
        """
        Args:
            margin: Triplet margin
            mining_type: Mining strategy ('hard', 'semi-hard', 'random')
        """
        super().__init__()
        self.margin = margin
        self.mining_type = mining_type
        self.stats = {
            'total_triplets': 0,
            'active_triplets': 0,
            'avg_positive_dist': 0,
            'avg_negative_dist': 0,
        }
    
    def forward(self, embeddings, labels):
        """
        Compute triplet loss with online mining.
        
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) integer class labels
            
        Returns:
            loss: Scalar triplet loss
            stats: Dictionary with mining statistics
        """
        # Compute pairwise distance matrix
        dist_matrix = self._pairwise_distances(embeddings)
        
        # Get positive and negative masks
        labels = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
        
        # Remove self-comparisons from positive mask
        identity = torch.eye(len(labels), device=labels.device)
        pos_mask = pos_mask - identity
        
        if self.mining_type == 'hard':
            loss, stats = self._hard_mining(dist_matrix, pos_mask, neg_mask)
        elif self.mining_type == 'semi-hard':
            loss, stats = self._semi_hard_mining(dist_matrix, pos_mask, neg_mask)
        else:
            loss, stats = self._random_mining(dist_matrix, pos_mask, neg_mask)
        
        self.stats = stats
        return loss, stats
    
    def _pairwise_distances(self, embeddings):
        """Compute pairwise Euclidean distance matrix."""
        # For L2-normalized embeddings: d^2 = 2 - 2*cos_sim
        dot = torch.mm(embeddings, embeddings.t())
        sq_norms = torch.diag(dot)
        distances = sq_norms.unsqueeze(0) - 2 * dot + sq_norms.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)
        distances = torch.sqrt(distances + 1e-8)
        return distances
    
    def _hard_mining(self, dist_matrix, pos_mask, neg_mask):
        """Hard triplet mining: hardest positive + hardest negative."""
        # Hardest positive: max distance among positives
        pos_dist = dist_matrix * pos_mask
        hardest_pos_dist = pos_dist.max(dim=1)[0]
        
        # Hardest negative: min distance among negatives
        # Mask negatives with large value, then take min
        neg_dist = dist_matrix + (1 - neg_mask) * 1e6
        hardest_neg_dist = neg_dist.min(dim=1)[0]
        
        # Filter anchors that have both positive and negative
        valid_anchors = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=dist_matrix.device, requires_grad=True), {
                'total_triplets': 0, 'active_triplets': 0,
                'avg_positive_dist': 0, 'avg_negative_dist': 0,
            }
        
        hardest_pos_dist = hardest_pos_dist[valid_anchors]
        hardest_neg_dist = hardest_neg_dist[valid_anchors]
        
        # Triplet loss
        losses = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        
        active = (losses > 0).sum().item()
        total = len(losses)
        
        stats = {
            'total_triplets': total,
            'active_triplets': active,
            'active_ratio': active / max(total, 1),
            'avg_positive_dist': hardest_pos_dist.mean().item(),
            'avg_negative_dist': hardest_neg_dist.mean().item(),
            'margin': self.margin,
        }
        
        return losses.mean(), stats
    
    def _semi_hard_mining(self, dist_matrix, pos_mask, neg_mask):
        """Semi-hard mining: hardest negative that's farther than positive."""
        pos_dist = dist_matrix * pos_mask
        hardest_pos_dist = pos_dist.max(dim=1)[0]
        
        # Semi-hard: negatives that are farther than positive but within margin
        neg_dist = dist_matrix * neg_mask
        
        losses = []
        for i in range(len(dist_matrix)):
            p_dist = hardest_pos_dist[i]
            n_dists = neg_dist[i][neg_mask[i] > 0]
            
            if len(n_dists) == 0 or p_dist == 0:
                continue
            
            # Semi-hard: p_dist < n_dist < p_dist + margin
            semi_hard_mask = (n_dists > p_dist) & (n_dists < p_dist + self.margin)
            
            if semi_hard_mask.sum() > 0:
                hardest_semi = n_dists[semi_hard_mask].max()
                loss = F.relu(p_dist - hardest_semi + self.margin)
                losses.append(loss)
            else:
                # Fall back to hardest negative
                hardest_neg = n_dists.min()
                loss = F.relu(p_dist - hardest_neg + self.margin)
                losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=dist_matrix.device, requires_grad=True), {
                'total_triplets': 0, 'active_triplets': 0,
                'avg_positive_dist': 0, 'avg_negative_dist': 0,
            }
        
        losses = torch.stack(losses)
        active = (losses > 0).sum().item()
        
        stats = {
            'total_triplets': len(losses),
            'active_triplets': active,
            'active_ratio': active / max(len(losses), 1),
            'avg_positive_dist': hardest_pos_dist.mean().item(),
            'avg_negative_dist': 0,
            'margin': self.margin,
        }
        
        return losses.mean(), stats
    
    def _random_mining(self, dist_matrix, pos_mask, neg_mask):
        """Random triplet sampling."""
        losses = []
        
        for i in range(len(dist_matrix)):
            pos_indices = torch.where(pos_mask[i] > 0)[0]
            neg_indices = torch.where(neg_mask[i] > 0)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            p_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
            n_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
            
            p_dist = dist_matrix[i, p_idx]
            n_dist = dist_matrix[i, n_idx]
            
            loss = F.relu(p_dist - n_dist + self.margin)
            losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=dist_matrix.device, requires_grad=True), {
                'total_triplets': 0, 'active_triplets': 0,
                'avg_positive_dist': 0, 'avg_negative_dist': 0,
            }
        
        losses = torch.stack(losses)
        active = (losses > 0).sum().item()
        
        return losses.mean(), {
            'total_triplets': len(losses),
            'active_triplets': active,
            'active_ratio': active / max(len(losses), 1),
            'avg_positive_dist': 0,
            'avg_negative_dist': 0,
        }


class CombinedLoss(nn.Module):
    """
    Combined loss: Triplet Loss + Cross-Entropy Loss.
    
    Using both metric learning (triplet) and classification (CE) losses
    is a common practice that provides complementary training signals.
    """
    
    def __init__(self, margin=0.5, mining_type='hard', ce_weight=0.5):
        super().__init__()
        self.triplet_loss = TripletLossWithMining(margin=margin, mining_type=mining_type)
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
    
    def forward(self, embeddings, logits, labels):
        """
        Compute combined loss.
        
        Args:
            embeddings: (B, D) embeddings from GNN
            logits: (B, C) classification logits
            labels: (B,) integer labels
            
        Returns:
            total_loss: Combined loss
            stats: Dictionary with component losses
        """
        triplet_loss, triplet_stats = self.triplet_loss(embeddings, labels)
        
        if logits is not None:
            ce_loss = self.ce_loss(logits, labels)
            total_loss = triplet_loss + self.ce_weight * ce_loss
        else:
            ce_loss = torch.tensor(0.0)
            total_loss = triplet_loss
        
        stats = {
            **triplet_stats,
            'triplet_loss': triplet_loss.item(),
            'ce_loss': ce_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, stats
