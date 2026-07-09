"""
External / Cross-Dataset Muzzle Loader
======================================
Generic loader for *any* muzzle dataset organised as one folder per animal:

    <root>/<animal_id>/<image>.{jpg,jpeg,png}

This enables cross-dataset transfer evaluation (train on dataset A, test on
dataset B) — the single most valuable experiment for a top-tier submission,
because it demonstrates that learned muzzle features generalise beyond the
training distribution rather than overfitting one capture setup.

No assumptions are made about identity overlap between datasets: labels are
assigned locally (sorted animal_id -> int), which is exactly what closed-set
self-similarity and open-set enrolment metrics require.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

_IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


class ExternalMuzzleImageDataset(Dataset):
    """Folder-per-animal image dataset for cross-dataset evaluation."""

    def __init__(self, root_dir: str, transform=None,
                 min_images_per_animal: int = 1, clahe=None, exclude_dirs=None):
        """
        Args:
            root_dir:              Path with one subdirectory per animal.
            transform:             torchvision transform (use build_val_transform).
            min_images_per_animal: Drop identities with fewer images.
            clahe:                 Optional dict of CLAHE params
                                   ({'clip_limit':.., 'tile_grid_size':..}) to
                                   match the training preprocessing. Without it,
                                   a model trained on CLAHE-enhanced muzzles sees
                                   a large domain shift on raw images.
        """
        self.transform = transform
        self._clahe = None
        if clahe:
            from src.preprocessing.enhancement import CLAHEEnhancer
            self._clahe = CLAHEEnhancer(
                clip_limit=clahe.get('clip_limit', 3.0),
                tile_grid_size=tuple(clahe.get('tile_grid_size', (8, 8))),
            )
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"External dataset root not found: {root}")

        # Some published sets ship a redundant "master pool" folder that
        # duplicates every animal's images under one label (e.g. the Kaggle
        # 'Cattle Muzzle - DB' ships 'OriginalMaster'). Left in, it makes every
        # image's rank-1 neighbour its own duplicate under the wrong label.
        exclude = {d.lower() for d in (exclude_dirs or ['OriginalMaster', 'Master', 'All'])}

        animal_to_imgs = {}
        for animal_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if animal_dir.name.lower() in exclude:
                print(f"  [ExternalMuzzle] excluding pool folder '{animal_dir.name}'")
                continue
            imgs = sorted(str(p) for p in animal_dir.iterdir()
                          if p.suffix.lower() in _IMG_EXT)
            if len(imgs) >= min_images_per_animal:
                animal_to_imgs[animal_dir.name] = imgs

        if not animal_to_imgs:
            raise RuntimeError(
                f"No animal folders with >= {min_images_per_animal} images under {root}.")

        self.animal_ids = sorted(animal_to_imgs)
        self.id_to_int = {a: i for i, a in enumerate(self.animal_ids)}

        self.samples: List[Tuple[str, int]] = []
        for a, imgs in animal_to_imgs.items():
            for img in imgs:
                self.samples.append((img, self.id_to_int[a]))

        print(f"  [ExternalMuzzle] {len(self.samples)} images, "
              f"{len(self.animal_ids)} identities from {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self._clahe is not None:
            import numpy as np
            import cv2
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            enhanced = self._clahe.enhance(bgr)
            img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def get_labels(self) -> List[int]:
        return [s[1] for s in self.samples]
