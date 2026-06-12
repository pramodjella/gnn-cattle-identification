"""
Image Dataset for CNN and Hybrid CNN-GNN models
================================================
Loads preprocessed muzzle images paired with their graph data.

Two dataset classes:
  - MuzzleImageDataset: Images only (for CNN baseline)
  - MuzzleImageGraphDataset: Images + graphs (for Hybrid CNN-GNN)
"""

import os
import json
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random


class MuzzleImageDataset(Dataset):
    """
    Dataset of preprocessed muzzle images for CNN baseline training.

    Handles the actual data format:
      - split JSON has: image_path (raw), animal_id (string like 'cattle_0100')
      - preprocessed images live in: data/preprocessed/images/{split}/{animal_id}/

    Returns (image_tensor, label) pairs where label is an integer index.
    """

    def __init__(self, split_json_path: str, transform=None,
                 preprocessed_images_dir: str = None):
        """
        Args:
            split_json_path: Path to train/val/test split JSON file.
            transform: torchvision transform (from augmentation.py).
            preprocessed_images_dir: Root of preprocessed images.
                                     Auto-detected from split JSON location if None.
        """
        self.transform = transform
        self.samples = []  # List of (image_path, int_label)

        split_json_path = Path(split_json_path)
        split_name = split_json_path.stem.replace('_split', '')  # 'train', 'val', 'test'

        # Auto-detect preprocessed images root
        if preprocessed_images_dir is None:
            # split JSON is at data/preprocessed/{split}_split.json
            # images are at data/preprocessed/images/{split}/{animal_id}/
            preprocessed_images_dir = split_json_path.parent / 'images' / split_name

        preprocessed_images_dir = Path(preprocessed_images_dir)

        with open(split_json_path) as f:
            split_data = json.load(f)

        # Build sorted list of unique animal IDs → deterministic integer mapping
        all_ids = sorted(set(item.get('animal_id', item.get('label', '')) for item in split_data))
        id_to_int = {aid: idx for idx, aid in enumerate(all_ids)}

        # Build image path → label mapping
        # Each animal has multiple images in preprocessed_images_dir/{animal_id}/
        animal_images = {}  # animal_id → list of preprocessed image paths
        if preprocessed_images_dir.exists():
            for animal_dir in preprocessed_images_dir.iterdir():
                if animal_dir.is_dir():
                    imgs = sorted(list(animal_dir.glob('*.png')) + list(animal_dir.glob('*.jpg')))
                    if imgs:
                        animal_images[animal_dir.name] = imgs

        # Match split entries to preprocessed images
        seen = {}  # track which preprocessed image files we've assigned
        for item in split_data:
            animal_id = item.get('animal_id', item.get('label', ''))
            if not animal_id or animal_id not in id_to_int:
                continue
            int_label = id_to_int[animal_id]

            # Strategy 1: find preprocessed version of the raw image
            raw_path = Path(item.get('image_path', ''))
            raw_stem = raw_path.stem  # e.g. 'cattle_0100_DSCF3858'

            # Look for preprocessed file with same stem
            found = False
            if animal_id in animal_images:
                for proc_img in animal_images[animal_id]:
                    if proc_img.stem == raw_stem or proc_img.stem.startswith(raw_stem):
                        if str(proc_img) not in seen:
                            self.samples.append((str(proc_img), int_label))
                            seen[str(proc_img)] = True
                            found = True
                            break

            # Strategy 2: if no exact stem match, use any available preprocessed image for this animal
            if not found and animal_id in animal_images:
                for proc_img in animal_images[animal_id]:
                    if str(proc_img) not in seen:
                        self.samples.append((str(proc_img), int_label))
                        seen[str(proc_img)] = True
                        break

        # Strategy 3: if preprocessed dir doesn't exist, fall back to raw images
        if len(self.samples) == 0:
            print(f"  [WARNING] No preprocessed images found in {preprocessed_images_dir}")
            print(f"  [WARNING] Falling back to raw images (augmentation still applied)")
            for item in split_data:
                animal_id = item.get('animal_id', item.get('label', ''))
                raw_path = item.get('image_path', '')
                if animal_id in id_to_int and raw_path and os.path.exists(raw_path):
                    self.samples.append((raw_path, id_to_int[animal_id]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found for {split_json_path}.\n"
                f"Looked in: {preprocessed_images_dir}\n"
                "Check that preprocessing has been run (02_preprocess.py)."
            )

        num_classes = len(set(s[1] for s in self.samples))
        print(f"  Loaded {len(self.samples)} images, {num_classes} classes "
              f"from {split_json_path.name} [{split_name}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return [s[1] for s in self.samples]


class MuzzleImageGraphDataset(Dataset):
    """
    Dataset pairing muzzle images with their graph representations.
    Used for Hybrid CNN-GNN training.

    Returns (image_tensor, graph, label) triples.
    """

    def __init__(self, split_json_path: str, graph_file: str,
                 transform=None, graph_augment=None):
        """
        Args:
            split_json_path: Path to split JSON file.
            graph_file: Path to the corresponding *_graphs.pt file.
            transform: Image transform.
            graph_augment: Graph augmentation callable.
        """
        self.transform = transform
        self.graph_augment = graph_augment

        # Load graphs
        graphs = torch.load(graph_file, weights_only=False)

        # Load split metadata
        with open(split_json_path) as f:
            split_data = json.load(f)

        split_json_path = Path(split_json_path)
        split_name = split_json_path.stem.replace('_split', '')  # 'train', 'val', 'test'

        # Build sorted animal_id → integer mapping (same as MuzzleImageDataset)
        all_ids = sorted(set(item.get('animal_id', item.get('label', '')) for item in split_data))
        id_to_int = {aid: idx for idx, aid in enumerate(all_ids)}

        # Build preprocessed image lookup: animal_id → list of image paths
        preprocessed_images_dir = split_json_path.parent / 'images' / split_name
        animal_images = {}
        if preprocessed_images_dir.exists():
            for animal_dir in preprocessed_images_dir.iterdir():
                if animal_dir.is_dir():
                    imgs = sorted(list(animal_dir.glob('*.png')) + list(animal_dir.glob('*.jpg')))
                    if imgs:
                        animal_images[animal_dir.name] = imgs

        # Build raw stem → preprocessed path lookup for fast matching
        stem_to_proc = {}
        for aid, imgs in animal_images.items():
            for img in imgs:
                stem_to_proc[img.stem] = str(img)

        # Match graphs to preprocessed images
        self.samples = []  # (image_path, graph, int_label)
        for g in graphs:
            # Get integer label from graph (already encoded during graph building)
            graph_label = g.y.item() if torch.is_tensor(g.y) else int(g.y)
            img_path = getattr(g, 'image_path', None)

            # Try to find preprocessed image from graph's stored image_path
            found_path = None
            if img_path:
                raw_stem = Path(str(img_path)).stem
                if raw_stem in stem_to_proc:
                    found_path = stem_to_proc[raw_stem]
                elif os.path.exists(str(img_path)):
                    found_path = str(img_path)

            # Fallback: match by graph label integer → find any image for that class
            if not found_path:
                # Reverse lookup: graph_label integer → animal_id
                # The graph y-label was set during graph building using sorted animal_ids
                if graph_label < len(all_ids):
                    animal_id = all_ids[graph_label]
                    if animal_id in animal_images and animal_images[animal_id]:
                        found_path = str(animal_images[animal_id][0])

            if found_path and os.path.exists(found_path):
                self.samples.append((found_path, g, graph_label))

        print(f"  Hybrid dataset: {len(self.samples)} paired image+graph samples, "
              f"{len(set(s[2] for s in self.samples))} classes")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, graph, label = self.samples[idx]

        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Augment graph
        if self.graph_augment:
            graph = self.graph_augment(graph)

        return image, graph, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return [s[2] for s in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# PK Sampler (works with both image and image+graph datasets)
# ─────────────────────────────────────────────────────────────────────────────

class PKSamplerForImages:
    """
    PK sampler for image datasets (P classes × K samples per batch).
    Ensures every batch has valid positive pairs for metric learning.
    """

    def __init__(self, labels, p=16, k=8):
        self.p = p
        self.k = k
        self.batch_size = p * k

        from collections import defaultdict
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.valid_labels = [
            lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= k
        ]
        if len(self.valid_labels) < p:
            self.valid_labels = [
                lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= 2
            ]
            self.k = min(k, 2)

        self.num_batches = max(1, len(labels) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            if len(self.valid_labels) >= self.p:
                selected_labels = random.sample(self.valid_labels, self.p)
            else:
                selected_labels = self.valid_labels.copy()
                while len(selected_labels) < self.p:
                    selected_labels.append(random.choice(self.valid_labels))

            indices = []
            for lbl in selected_labels:
                pool = self.label_to_indices[lbl]
                if len(pool) >= self.k:
                    indices.extend(random.sample(pool, self.k))
                else:
                    indices.extend(random.choices(pool, k=self.k))

            yield from indices

    def __len__(self):
        return self.num_batches * self.batch_size


def create_image_loaders(preprocessed_dir: str, config: dict,
                         train_transform=None, val_transform=None):
    """
    Create DataLoaders for CNN baseline training.

    Args:
        preprocessed_dir: Directory containing split JSON files.
        config: Full config dict.
        train_transform: Transform for training images.
        val_transform: Transform for val/test images.

    Returns:
        dict of DataLoaders: {'train': ..., 'val': ..., 'test': ...}
    """
    from src.training.augmentation import build_train_transform, build_val_transform

    image_size = config.get('preprocessing', {}).get('image_size', 256)
    if train_transform is None:
        train_transform = build_train_transform(image_size)
    if val_transform is None:
        val_transform = build_val_transform(image_size)

    cnn_cfg = config.get('cnn', {})
    batch_size = cnn_cfg.get('batch_size', 32)
    k_per_class = config['training'].get('triplet', {}).get('samples_per_class', 8)
    p_classes = max(2, batch_size // k_per_class)

    loaders = {}
    for split in ['train', 'val', 'test']:
        split_json = os.path.join(preprocessed_dir, f'{split}_split.json')
        if not os.path.exists(split_json):
            print(f"  [WARNING] {split_json} not found, skipping")
            continue

        import sys
        is_windows = sys.platform == 'win32'
        num_workers = 0 if is_windows else 4
        persistent_workers = False if is_windows else True
        prefetch_factor = None if is_windows else 2

        if split == 'train':
            ds = MuzzleImageDataset(split_json, transform=train_transform)
            sampler = PKSamplerForImages(ds.get_labels(), p=p_classes, k=k_per_class)
            loaders['train'] = DataLoader(
                ds, batch_size=batch_size, sampler=sampler,
                drop_last=True, num_workers=num_workers, pin_memory=True,
                persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
            )
        else:
            ds = MuzzleImageDataset(split_json, transform=val_transform)
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )

    return loaders


def hybrid_collate_fn(batch):
    """Custom collate for (image, graph, label) triples."""
    from torch_geometric.data import Batch
    images, graphs, labels = zip(*batch)
    return torch.stack(images), Batch.from_data_list(list(graphs)), torch.stack(labels)


def create_hybrid_loaders(preprocessed_dir: str, graph_dir: str, config: dict,
                           train_transform=None, val_transform=None,
                           train_graph_aug=None):
    """Create DataLoaders for Hybrid CNN-GNN training."""
    from src.training.augmentation import (
        build_train_transform, build_val_transform,
        GraphAugmentation, IdentityGraphAugmentation
    )

    image_size = config.get('preprocessing', {}).get('image_size', 256)
    if train_transform is None:
        train_transform = build_train_transform(image_size)
    if val_transform is None:
        val_transform = build_val_transform(image_size)
    if train_graph_aug is None:
        train_graph_aug = GraphAugmentation()

    hybrid_cfg = config.get('hybrid', {})
    batch_size = hybrid_cfg.get('batch_size', 16)
    k_per_class = config['training'].get('triplet', {}).get('samples_per_class', 4)
    p_classes = max(2, batch_size // k_per_class)

    loaders = {}
    for split in ['train', 'val', 'test']:
        split_json = os.path.join(preprocessed_dir, f'{split}_split.json')
        graph_file = os.path.join(graph_dir, f'{split}_graphs.pt')

        if not os.path.exists(split_json) or not os.path.exists(graph_file):
            print(f"  [WARNING] Missing files for {split}, skipping")
            continue

        aug = train_graph_aug if split == 'train' else IdentityGraphAugmentation()
        transform = train_transform if split == 'train' else val_transform

        ds = MuzzleImageGraphDataset(split_json, graph_file, transform=transform,
                                      graph_augment=aug)

        import sys
        is_windows = sys.platform == 'win32'
        num_workers = 0 if is_windows else 4
        persistent_workers = False if is_windows else True

        if split == 'train':
            sampler = PKSamplerForImages(ds.get_labels(), p=p_classes, k=k_per_class)
            loaders['train'] = DataLoader(
                ds, batch_size=batch_size, sampler=sampler,
                drop_last=True, num_workers=num_workers, pin_memory=True,
                collate_fn=hybrid_collate_fn, persistent_workers=persistent_workers,
            )
        else:
            loaders[split] = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
                collate_fn=hybrid_collate_fn, persistent_workers=persistent_workers,
            )

    return loaders
