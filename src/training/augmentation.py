"""
Augmentation Pipelines
=======================
Shared augmentation strategies for all three model types:
  - ImageAugmentation: For CNN baseline and Hybrid CNN-GNN (operates on PIL/tensor images)
  - GraphAugmentation: For GNN+ and Hybrid CNN-GNN (operates on PyG Data objects)

Design philosophy:
  Augmentations simulate real-world cattle scanning variability:
  - Lighting changes (barn lighting, outdoor sun)
  - Head pose (animal moves slightly during scan)
  - Partial occlusion (dirt, wet muzzle, hair)
  - Sensor noise (varying camera quality)
"""

import random
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# Custom Transforms
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNoise(torch.nn.Module):
    """
    Adds additive Gaussian noise to a tensor image.
    Simulates sensor noise from varying camera quality in farm environments.
    Applied AFTER ToTensor() and Normalize().
    """
    def __init__(self, std: float = 0.01, p: float = 0.2):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(img) * self.std
            return (img + noise).clamp(-3.0, 3.0)  # stay in normalized range
        return img


# ─────────────────────────────────────────────────────────────────────────────
# Image Augmentation (CNN & Hybrid)
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transform(image_size: int = 256):
    """
    Enhanced training augmentation pipeline for muzzle images. [TUNED for 98%+]

    Designed for cattle biometrics with literature-backed augmentations:
    - RandomPerspective: camera angle variation in farm settings
    - Stronger ColorJitter: barn vs outdoor vs overcast lighting
    - RandomGrayscale: forces model to rely on texture over color cues
    - GaussianNoise: sensor noise from varying camera quality
    - Random erasing: simulates mud/occlusion on muzzle
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08)
        ], p=0.85),
        T.RandomApply([
            T.RandomAffine(degrees=15, translate=(0.10, 0.10), scale=(0.90, 1.10))
        ], p=0.5),
        T.RandomApply([
            T.RandomPerspective(distortion_scale=0.12, p=1.0)
        ], p=0.3),
        T.RandomGrayscale(p=0.10),
        T.RandomApply([
            T.GaussianBlur(kernel_size=5, sigma=(0.3, 2.0))
        ], p=0.25),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Gaussian noise (applied to tensor)
        GaussianNoise(std=0.01, p=0.2),
        T.RandomErasing(p=0.35, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0),
    ])


def build_val_transform(image_size: int = 256):
    """
    Validation/test transform — no augmentation, just normalize.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_tta_transforms(image_size: int = 256):
    """
    Test-Time Augmentation transforms for inference.
    Returns a list of transforms (one per TTA view).
    Average the embeddings from all views before nearest-neighbor matching.

    5 views: original + hflip + 5-crop (center + 4 corners sampled at 88%)
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop_size = int(image_size * 0.88)
    return [
        # View 1: original (no augmentation)
        T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), normalize]),
        # View 2: horizontal flip
        T.Compose([T.Resize((image_size, image_size)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), normalize]),
        # View 3: center crop (slight zoom-in)
        T.Compose([T.Resize((int(image_size * 1.1), int(image_size * 1.1))), T.CenterCrop(image_size), T.ToTensor(), normalize]),
        # View 4: slight brightness increase  (brightness factor in [0.8, 1.2])
        T.Compose([T.Resize((image_size, image_size)), T.ColorJitter(brightness=0.2), T.ToTensor(), normalize]),
        # View 5: slight brightness decrease  (brightness factor in [0.7, 0.9])
        T.Compose([T.Resize((image_size, image_size)), T.ColorJitter(brightness=(0.7, 0.9)), T.ToTensor(), normalize]),
    ]


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Augmentation (GNN+ & Hybrid)
# ─────────────────────────────────────────────────────────────────────────────

class GraphAugmentation:
    """
    Stochastic augmentation of graph-structured muzzle data.

    Applied only during training. Each call randomly applies one or more
    of the following transforms with the specified probabilities:
      1. KeypointDropout:  Randomly removes nodes (simulates detection failures)
      2. FeatureJitter:    Adds noise to node features (descriptor uncertainty)
      3. EdgeDropout:      Randomly removes edges (simulates spatial occlusion)
      4. PositionJitter:   Small spatial perturbation of keypoint locations
    """

    def __init__(self,
                 dropout_prob: float = 0.5,    # Prob of applying keypoint dropout
                 drop_rate: float = 0.15,        # Fraction of nodes to drop
                 jitter_prob: float = 0.5,
                 jitter_sigma: float = 0.02,    # Relative to descriptor magnitude
                 edge_drop_prob: float = 0.3,
                 edge_drop_rate: float = 0.10,
                 pos_jitter_prob: float = 0.4,
                 pos_jitter_sigma: float = 0.005):  # Normalized position space
        self.dropout_prob = dropout_prob
        self.drop_rate = drop_rate
        self.jitter_prob = jitter_prob
        self.jitter_sigma = jitter_sigma
        self.edge_drop_prob = edge_drop_prob
        self.edge_drop_rate = edge_drop_rate
        self.pos_jitter_prob = pos_jitter_prob
        self.pos_jitter_sigma = pos_jitter_sigma

    def __call__(self, data: Data) -> Data:
        """Apply stochastic augmentations to a graph."""
        # Work on a shallow copy to avoid modifying cached data
        data = data.clone()

        if random.random() < self.jitter_prob:
            data = self._feature_jitter(data)

        if random.random() < self.pos_jitter_prob and hasattr(data, 'pos') and data.pos is not None:
            data = self._position_jitter(data)

        if random.random() < self.edge_drop_prob:
            data = self._edge_dropout(data)

        if random.random() < self.dropout_prob:
            data = self._keypoint_dropout(data)

        return data

    def _feature_jitter(self, data: Data) -> Data:
        """Add Gaussian noise to node feature vectors."""
        if data.x is not None:
            noise = torch.randn_like(data.x) * self.jitter_sigma
            data.x = data.x + noise
            # Re-normalize to unit norm (SuperPoint descriptors are unit-normed)
            norms = data.x.norm(dim=1, keepdim=True).clamp(min=1e-8)
            data.x = data.x / norms
        return data

    def _position_jitter(self, data: Data) -> Data:
        """Small spatial perturbation of keypoint coordinates."""
        noise = torch.randn_like(data.pos) * self.pos_jitter_sigma
        data.pos = (data.pos + noise).clamp(0.0, 1.0)
        return data

    def _edge_dropout(self, data: Data) -> Data:
        """Randomly drop edges from the graph."""
        E = data.edge_index.shape[1]
        if E == 0:
            return data
        keep_mask = torch.rand(E, device=data.edge_index.device) > self.edge_drop_rate
        # Always keep at least 50% of edges
        if keep_mask.sum() < E * 0.5:
            keep_mask = torch.rand(E, device=data.edge_index.device) > 0.2
        data.edge_index = data.edge_index[:, keep_mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[keep_mask]
        return data

    def _keypoint_dropout(self, data: Data) -> Data:
        """Remove random subset of keypoints and their incident edges."""
        N = data.x.shape[0]
        if N <= 10:
            return data

        num_keep = max(10, int(N * (1.0 - self.drop_rate)))
        keep_idx = torch.randperm(N, device=data.x.device)[:num_keep].sort()[0]

        # Remap index space
        remap = torch.full((N,), -1, dtype=torch.long, device=data.x.device)
        remap[keep_idx] = torch.arange(num_keep, device=data.x.device)

        # Filter nodes
        data.x = data.x[keep_idx]
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos[keep_idx]
        if hasattr(data, 'keypoint_scores') and data.keypoint_scores is not None:
            data.keypoint_scores = data.keypoint_scores[keep_idx]

        # Filter edges: keep only edges where both endpoints survived
        src, dst = data.edge_index
        new_src = remap[src]
        new_dst = remap[dst]
        valid = (new_src >= 0) & (new_dst >= 0)
        data.edge_index = torch.stack([new_src[valid], new_dst[valid]], dim=0)
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[valid]

        data.num_keypoints = num_keep
        return data


# ─────────────────────────────────────────────────────────────────────────────
# No-op augmentation (for val/test)
# ─────────────────────────────────────────────────────────────────────────────

class IdentityGraphAugmentation:
    """Pass-through augmentation for validation/test sets."""
    def __call__(self, data: Data) -> Data:
        return data


class SubgraphCrop:
    """
    Randomly sample a spatially contiguous subgraph.
    
    Simulates partial muzzle views (e.g., camera only captures part of the muzzle).
    Selects a random seed node, then grabs its spatial neighbors within a radius.
    
    From GCL literature: subgraph sampling is one of the most effective
    augmentations for graph classification (You et al., 2020).
    """
    
    def __init__(self, min_keep_ratio: float = 0.6, prob: float = 0.3):
        """
        Args:
            min_keep_ratio: Minimum fraction of nodes to keep (0.6 = keep at least 60%)
            prob: Probability of applying this augmentation
        """
        self.min_keep_ratio = min_keep_ratio
        self.prob = prob
    
    def __call__(self, data: Data) -> Data:
        if random.random() > self.prob:
            return data
        if not hasattr(data, 'pos') or data.pos is None:
            return data
        
        N = data.x.shape[0]
        if N <= 10:
            return data
        
        data = data.clone()
        
        # Pick a random seed node
        seed = random.randint(0, N - 1)
        seed_pos = data.pos[seed]
        
        # Compute distances from seed in spatial space
        dists = (data.pos - seed_pos.unsqueeze(0)).norm(dim=1)
        
        # Keep the closest nodes (at least min_keep_ratio)
        num_keep = max(10, int(N * self.min_keep_ratio))
        # Add some randomness to the crop size
        num_keep = min(N, max(num_keep, int(N * random.uniform(self.min_keep_ratio, 0.95))))
        
        _, keep_idx = dists.topk(num_keep, largest=False)
        keep_idx = keep_idx.sort()[0]
        
        # Remap
        remap = torch.full((N,), -1, dtype=torch.long, device=data.x.device)
        remap[keep_idx] = torch.arange(num_keep, device=data.x.device)
        
        data.x = data.x[keep_idx]
        data.pos = data.pos[keep_idx]
        
        if hasattr(data, 'keypoint_scores') and data.keypoint_scores is not None:
            data.keypoint_scores = data.keypoint_scores[keep_idx]
        
        # Filter edges
        src, dst = data.edge_index
        new_src = remap[src]
        new_dst = remap[dst]
        valid = (new_src >= 0) & (new_dst >= 0)
        data.edge_index = torch.stack([new_src[valid], new_dst[valid]], dim=0)
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[valid]
        
        return data


class FeatureMixup:
    """
    Feature-level mixup between random node pairs within the same graph.
    
    For each selected node, interpolate its features with a random neighbor's
    features using a Beta-distributed mixing coefficient. This creates
    'virtual' intermediate descriptors that regularize the feature space.
    
    Inspired by manifold mixup (Verma et al., 2019) adapted for graphs.
    """
    
    def __init__(self, prob: float = 0.3, alpha: float = 0.2, mix_ratio: float = 0.15):
        """
        Args:
            prob: Probability of applying mixup
            alpha: Beta distribution parameter (smaller = more extreme mixing)
            mix_ratio: Fraction of nodes to apply mixup to
        """
        self.prob = prob
        self.alpha = alpha
        self.mix_ratio = mix_ratio
    
    def __call__(self, data: Data) -> Data:
        if random.random() > self.prob:
            return data
        
        N = data.x.shape[0]
        if N <= 5:
            return data
        
        data = data.clone()
        
        # Select nodes to mix
        num_mix = max(1, int(N * self.mix_ratio))
        mix_idx = torch.randperm(N, device=data.x.device)[:num_mix]
        
        # Random partner for each
        partner_idx = torch.randint(0, N, (num_mix,), device=data.x.device)
        
        # Beta-distributed mixing coefficient
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((num_mix, 1)).to(data.x.device)
        
        # Mix features
        data.x[mix_idx] = lam * data.x[mix_idx] + (1 - lam) * data.x[partner_idx]
        
        # Re-normalize (descriptors should be unit-normed)
        norms = data.x[mix_idx].norm(dim=1, keepdim=True).clamp(min=1e-8)
        data.x[mix_idx] = data.x[mix_idx] / norms
        
        return data


class EnhancedGraphAugmentation:
    """
    Full augmentation pipeline combining all graph transforms.
    
    Extends the base GraphAugmentation with SubgraphCrop and FeatureMixup
    for improved generalization on small datasets (260 classes, ~19 imgs/class).
    """
    
    def __init__(self,
                 # Base augmentations
                 dropout_prob: float = 0.5,
                 drop_rate: float = 0.15,
                 jitter_prob: float = 0.5,
                 jitter_sigma: float = 0.02,
                 edge_drop_prob: float = 0.3,
                 edge_drop_rate: float = 0.10,
                 pos_jitter_prob: float = 0.4,
                 pos_jitter_sigma: float = 0.005,
                 # New augmentations
                 subgraph_crop_prob: float = 0.3,
                 subgraph_min_keep: float = 0.6,
                 feature_mixup_prob: float = 0.3,
                 feature_mixup_alpha: float = 0.2):
        
        self.base_aug = GraphAugmentation(
            dropout_prob=dropout_prob,
            drop_rate=drop_rate,
            jitter_prob=jitter_prob,
            jitter_sigma=jitter_sigma,
            edge_drop_prob=edge_drop_prob,
            edge_drop_rate=edge_drop_rate,
            pos_jitter_prob=pos_jitter_prob,
            pos_jitter_sigma=pos_jitter_sigma,
        )
        self.subgraph_crop = SubgraphCrop(
            min_keep_ratio=subgraph_min_keep,
            prob=subgraph_crop_prob,
        )
        self.feature_mixup = FeatureMixup(
            prob=feature_mixup_prob,
            alpha=feature_mixup_alpha,
        )
    
    def __call__(self, data: Data) -> Data:
        # Apply subgraph crop first (before other transforms)
        data = self.subgraph_crop(data)
        # Then base augmentations
        data = self.base_aug(data)
        # Finally feature mixup
        data = self.feature_mixup(data)
        return data
