"""
ArcFace Loss (Additive Angular Margin Loss)
============================================
The gold standard metric learning loss for biometric identification.

Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
CVPR 2019. https://arxiv.org/abs/1801.07698

Core idea: For the true class y_i, replace cos(θ) with cos(θ + m), adding an
additive angular margin m (default 0.5 rad = 28.6°) in the hyperspherical space.
This compresses intra-class variance and enlarges inter-class margins simultaneously.

L = -log( e^(s * cos(θ_yi + m)) / (e^(s * cos(θ_yi + m)) + Σ_{j≠y_i} e^(s * cos(θ_j))) )

vs Triplet Loss which only enforces: d(a,p) + margin < d(a,n).

ArcFace directly optimizes ALL pairwise relationships in the class space,
not just sampled triplets — hence dramatically faster convergence and
higher final accuracy in biometric benchmarks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head for metric learning.

    This replaces the standard Linear + CrossEntropy combination.
    It operates on L2-normalized embeddings and maintains a unit-norm
    weight matrix (class prototype vectors on the unit hypersphere).

    Usage:
        arcface = ArcFaceHead(embedding_dim=256, num_classes=260)
        # During training:
        logits = arcface(embeddings, labels)
        loss = F.cross_entropy(logits, labels)
        # During inference (embedding extraction only):
        # Just use the embeddings from the backbone — no ArcFace needed.
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 margin: float = 0.5, scale: float = 64.0,
                 easy_margin: bool = False):
        """
        Args:
            embedding_dim: Dimension of L2-normalized input embeddings.
            num_classes: Number of identity classes (260 for this dataset).
            margin: Angular margin in radians (default 0.5 ≈ 28.6°).
            scale: Feature scale / temperature (default 64.0).
            easy_margin: Use easy margin variant (more stable early training).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        # Class prototype vectors — each class has a unit-norm weight vector
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute margin cosine values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)   # Threshold for easy margin
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized embedding vectors.
            labels: (B,) ground-truth class indices. Required during training.
                    If None, returns raw cosine similarities (inference mode).

        Returns:
            logits: (B, C) scaled cosine similarities with angular margin
                    applied to the true class. Feed directly into cross_entropy.
        """
        # Normalize weight matrix (class prototypes on unit sphere)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: (B, C)
        cosine = F.linear(embeddings, weight_norm)  # embeddings already normalized

        if labels is None:
            # Inference: return scaled cosine similarities
            return cosine * self.scale

        # --- Training: apply additive angular margin to true class ---
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # Numerical stability

        # sin(θ) = sqrt(1 - cos²(θ))
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # Easy margin: only apply margin when cos(θ) > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Standard ArcFace: ensure monotonicity
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply margin only to the true class, keep cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale (temperature) — amplifies the logit range for sharper softmax
        output = output * self.scale

        return output

    def extra_repr(self):
        return (f"embedding_dim={self.embedding_dim}, num_classes={self.num_classes}, "
                f"margin={self.margin:.3f} ({math.degrees(self.margin):.1f}°), "
                f"scale={self.scale}")


class ArcFaceLoss(nn.Module):
    """
    Complete ArcFace training loss with optional auxiliary triplet loss.

    Combines:
        1. ArcFace (primary): Angular margin softmax — optimizes class separation
        2. Triplet auxiliary (optional): Ensures embedding space geometry

    For this dataset: Primary ArcFace + 0.1× auxiliary triplet works best.
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 margin: float = 0.5, scale: float = 64.0,
                 triplet_weight: float = 0.1, triplet_margin: float = 0.3,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.arcface_head = ArcFaceHead(embedding_dim, num_classes, margin, scale)
        self.triplet_weight = triplet_weight
        self.triplet_margin = triplet_margin
        self.label_smoothing = label_smoothing

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings.
            labels: (B,) ground-truth labels.

        Returns:
            loss: Scalar combined loss.
            stats: Dict with component losses and active triplet ratio.
        """
        # Primary: ArcFace with optional label smoothing
        logits = self.arcface_head(embeddings, labels)
        arcface_loss = F.cross_entropy(
            logits, labels.long(),
            label_smoothing=self.label_smoothing,
        )

        stats = {
            'arcface_loss': arcface_loss.item(),
            'total_triplets': 0,
            'active_triplets': 0,
            'active_ratio': 0.0,
        }

        total_loss = arcface_loss

        # Auxiliary: triplet loss on normalized embeddings
        if self.triplet_weight > 0:
            triplet_loss, triplet_stats = self._batch_hard_triplet(embeddings, labels)
            total_loss = arcface_loss + self.triplet_weight * triplet_loss
            stats.update({
                'triplet_loss': triplet_loss.item(),
                'active_ratio': triplet_stats['active_ratio'],
                'total_triplets': triplet_stats['total_triplets'],
                'active_triplets': triplet_stats['active_triplets'],
            })

        stats['total_loss'] = total_loss.item()
        return total_loss, stats

    def _batch_hard_triplet(self, embeddings, labels):
        """Batch hard triplet mining on cosine distances."""
        # Cosine distance = 1 - cosine_similarity (embeddings are L2-normalized)
        sim = torch.mm(embeddings, embeddings.t())
        dist = 1.0 - sim

        labels = labels.view(-1)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
        eye = torch.eye(len(labels), device=labels.device)
        pos_mask = pos_mask - eye

        # Hardest positive (max cosine distance = least similar same-class)
        pos_dist = dist * pos_mask
        hardest_pos = pos_dist.max(dim=1)[0]

        # Hardest negative (min cosine distance = most similar different-class)
        neg_dist = dist + (1 - neg_mask) * 1e6
        hardest_neg = neg_dist.min(dim=1)[0]

        valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True), {
                'total_triplets': 0, 'active_triplets': 0, 'active_ratio': 0.0
            }

        hp = hardest_pos[valid]
        hn = hardest_neg[valid]
        losses = F.relu(hp - hn + self.triplet_margin)

        active = (losses > 0).sum().item()
        total = len(losses)
        return losses.mean(), {
            'total_triplets': total,
            'active_triplets': active,
            'active_ratio': active / max(total, 1),
        }

    def get_head(self) -> ArcFaceHead:
        """Return the ArcFace head (for weight reuse or fine-tuning)."""
        return self.arcface_head
