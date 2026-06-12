"""
Cattle Muzzle Graph Dataset
============================
PyTorch Geometric dataset for loading pre-built muzzle graphs.
Implements PK sampling for triplet training (P classes, K samples each).
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict


class CattleMuzzleGraphDataset(InMemoryDataset):
    """
    PyG Dataset for cattle muzzle graphs.
    Loads pre-built graphs from .pt files.
    """
    
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        """
        Args:
            root: Root directory containing graph files
            split: 'train', 'val', or 'test'
            transform: Optional transform
            pre_transform: Optional pre-transform
        """
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [f'{self.split}_graphs.pt']
    
    @property
    def processed_file_names(self):
        return [f'{self.split}_processed.pt']
    
    def download(self):
        pass  # Data already prepared by scripts 01-04
    
    def process(self):
        """Load and process graph data."""
        raw_path = os.path.join(self.raw_dir, f'{self.split}_graphs.pt')
        
        if not os.path.exists(raw_path):
            # Try parent directory
            raw_path = os.path.join(self.root, f'{self.split}_graphs.pt')
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Graph file not found: {raw_path}")
        
        data_list = torch.load(raw_path, weights_only=False)
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        self.save(data_list, self.processed_paths[0])
    
    def get_labels(self):
        """Get all labels in the dataset."""
        return [data.y.item() for data in self]
    
    def get_class_counts(self):
        """Get number of samples per class."""
        labels = self.get_labels()
        counts = defaultdict(int)
        for l in labels:
            counts[l] += 1
        return dict(counts)


class PKSampler(Sampler):
    """
    PK Sampler for triplet training.
    Each batch contains P random classes with K samples each.
    Ensures every batch has valid triplet pairs.
    """
    
    def __init__(self, labels, p=16, k=4):
        """
        Args:
            labels: List of integer labels for all samples
            p: Number of classes per batch
            k: Number of samples per class per batch
        """
        self.labels = labels
        self.p = p
        self.k = k
        self.batch_size = p * k
        
        # Group indices by label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
        # Filter classes with at least k samples
        self.valid_labels = [
            label for label, indices in self.label_to_indices.items()
            if len(indices) >= k
        ]
        
        if len(self.valid_labels) < p:
            # Relax: allow classes with at least 2 samples
            self.valid_labels = [
                label for label, indices in self.label_to_indices.items()
                if len(indices) >= 2
            ]
            self.k = min(k, 2)
        
        self.num_batches = max(1, len(labels) // self.batch_size)
    
    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Sample P classes
            if len(self.valid_labels) >= self.p:
                selected_labels = random.sample(self.valid_labels, self.p)
            else:
                selected_labels = self.valid_labels.copy()
                # Pad with random repeats
                while len(selected_labels) < self.p:
                    selected_labels.append(random.choice(self.valid_labels))
            
            # For each class, sample K instances
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.k:
                    selected = random.sample(indices, self.k)
                else:
                    # Oversample
                    selected = [random.choice(indices) for _ in range(self.k)]
                batch_indices.extend(selected)
            
            yield from batch_indices
    
    def __len__(self):
        return self.num_batches * self.batch_size


class RandomKeypointDropout:
    """
    Data augmentation: Randomly drop keypoints from graphs.
    Simulates occlusion and quality degradation.
    """
    
    def __init__(self, drop_rate=0.1):
        self.drop_rate = drop_rate
    
    def __call__(self, data):
        if random.random() > 0.5:  # Apply 50% of the time
            return data
        
        n = data.x.shape[0]
        if n <= 5:  # Don't drop from very small graphs
            return data
        
        # Randomly select nodes to keep
        num_keep = max(5, int(n * (1 - self.drop_rate)))
        keep_mask = sorted(random.sample(range(n), num_keep))
        
        # Remap node indices
        remap = {old: new for new, old in enumerate(keep_mask)}
        
        # Filter nodes
        data.x = data.x[keep_mask]
        data.pos = data.pos[keep_mask] if hasattr(data, 'pos') and data.pos is not None else None
        
        if hasattr(data, 'keypoint_scores') and data.keypoint_scores is not None:
            data.keypoint_scores = data.keypoint_scores[keep_mask]
        
        # Filter and remap edges
        keep_set = set(keep_mask)
        new_edge_sources = []
        new_edge_targets = []
        new_edge_attrs = []
        
        for e in range(data.edge_index.shape[1]):
            src = data.edge_index[0, e].item()
            dst = data.edge_index[1, e].item()
            if src in keep_set and dst in keep_set:
                new_edge_sources.append(remap[src])
                new_edge_targets.append(remap[dst])
                if data.edge_attr is not None:
                    new_edge_attrs.append(data.edge_attr[e])
        
        if new_edge_sources:
            data.edge_index = torch.tensor([new_edge_sources, new_edge_targets], dtype=torch.long)
            if data.edge_attr is not None and new_edge_attrs:
                data.edge_attr = torch.stack(new_edge_attrs)
        else:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            if data.edge_attr is not None:
                data.edge_attr = torch.zeros((0, data.edge_attr.shape[1]))
        
        data.num_keypoints = num_keep
        
        return data


class TransformListDataset(torch.utils.data.Dataset):
    """Wrapper to apply transforms dynamically on list elements."""
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data.clone())
        return data


def create_data_loaders(graph_dir, config, augment_train=True):
    """
    Create data loaders for train/val/test splits.
    
    Args:
        graph_dir: Directory containing graph .pt files
        config: Configuration dict
        augment_train: Whether to apply augmentation to training data
        
    Returns:
        dict of DataLoaders: {'train': ..., 'val': ..., 'test': ...}
    """
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        graph_file = os.path.join(graph_dir, f'{split}_graphs.pt')
        
        if not os.path.exists(graph_file):
            print(f"[WARNING] {graph_file} not found, skipping {split} split")
            continue
        
        data_list = torch.load(graph_file, weights_only=False)
        
        if not data_list:
            print(f"[WARNING] Empty data list for {split}")
            continue
        
        # Apply augmentation to training data
        if split == 'train' and augment_train:
            transform = RandomKeypointDropout(drop_rate=0.1)
        else:
            transform = None
        
        dataset = TransformListDataset(data_list, transform)
        batch_size = config['training']['batch_size']
        
        if split == 'train':
            # Use PK sampling for training
            labels = [d.y.item() for d in data_list]
            pk_config = config['training'].get('triplet', {})
            k_per_class = pk_config.get('samples_per_class', 4)
            p_classes = max(2, batch_size // k_per_class)

            sampler = PKSampler(labels, p=p_classes, k=k_per_class)

            loader = DataLoader(
                dataset, batch_size=batch_size,
                sampler=sampler, drop_last=True,
                num_workers=0,
                pin_memory=True,
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False, drop_last=False,
                num_workers=0,
                pin_memory=True,
            )

        
        loaders[split] = loader
        print(f"[INFO] {split.capitalize()} loader: {len(data_list)} graphs, batch_size={batch_size}")
    
    return loaders
