"""
Script: Pre-Extract MobileNetV3 Patch Features for GNN++ 
=========================================================
For each graph, loads the original muzzle image and extracts
32x32 CNN patches at every keypoint location using MobileNetV3-Small.

Replaces handcrafted SuperPoint/SIFT 256-d descriptors with
576-d learned CNN features + 2-d normalized position.
Total node feature dim: 578-d  (will be projected in GNN++)

Output: data/graphs/train_graphs_v2.pt, val_graphs_v2.pt, test_graphs_v2.pt
Each graph has the same structure as before but with enhanced node features.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config


# ── Patch extraction config ───────────────────────────────────────────────────
PATCH_SIZE   = 32     # pixels around each keypoint
FEATURE_DIM  = 576    # MobileNetV3-Small penultimate layer output
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_patch_encoder():
    """MobileNetV3-Small with the classifier head removed → 576-d features."""
    model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Remove the final classifier, keep up to the AdaptiveAvgPool → 576-d
    encoder = nn.Sequential(*list(model.children())[:-1])  # remove Linear classifier
    encoder = encoder.eval().to(DEVICE)
    # Freeze – we only use it for feature extraction
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def extract_patches_batch(image: Image.Image, positions: torch.Tensor,
                           patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Extract and encode patches for all keypoints in one image.

    Args:
        image: PIL Image (H, W, 3)
        positions: (N, 2) tensor of normalized keypoint coords in [0, 1]
        patch_size: pixel size of each square patch

    Returns:
        patches: (N, 3, patch_size, patch_size) float tensor, normalized
    """
    W, H = image.size
    half = patch_size // 2

    # Convert to tensor once
    img_t = TF.to_tensor(image)  # (3, H, W)

    patches = []
    for i in range(positions.shape[0]):
        cx = int(positions[i, 0].item() * W)
        cy = int(positions[i, 1].item() * H)

        # Clamp so patch doesn't go out of bounds
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(W, cx + half)
        y2 = min(H, cy + half)

        patch = img_t[:, y1:y2, x1:x2]

        # Pad to fixed size if near border
        ph, pw = patch.shape[1], patch.shape[2]
        if ph < patch_size or pw < patch_size:
            pad_b = patch_size - ph
            pad_r = patch_size - pw
            patch = torch.nn.functional.pad(patch, (0, pad_r, 0, pad_b))

        # Resize to exactly patch_size × patch_size
        patch = TF.resize(patch, [patch_size, patch_size], antialias=True)

        # ImageNet normalize
        patch = TF.normalize(patch,
                              mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        patches.append(patch)

    return torch.stack(patches)  # (N, 3, 32, 32)


def encode_patches(encoder: nn.Module, patches: torch.Tensor,
                    batch_size: int = 128) -> torch.Tensor:
    """
    Run patches through MobileNetV3 in mini-batches.
    Returns (N, 576) feature tensor.
    """
    all_feats = []
    for i in range(0, patches.shape[0], batch_size):
        batch = patches[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            feats = encoder(batch)            # (B, 576, 1, 1) or (B, 576)
            feats = feats.flatten(1)          # (B, 576)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats)              # (N, 576)


def process_split(graphs, split_name, encoder, out_dir, image_size=256):
    """Enhance all graphs in a split with CNN patch features."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enhanced = []
    t0 = time.time()
    missing_image = 0

    for idx, g in enumerate(graphs):
        img_path = getattr(g, 'image_path', None)
        pos = getattr(g, 'pos', None)

        # Try to find the original image
        image = None
        if img_path and Path(str(img_path)).exists():
            try:
                image = Image.open(str(img_path)).convert('RGB')
            except Exception:
                image = None

        # If original image not found, try preprocessed
        if image is None and pos is not None:
            # Look up preprocessed image by matching graph label to folder
            label = g.y.item() if torch.is_tensor(g.y) else int(g.y)
            missing_image += 1

        if image is not None and pos is not None and pos.shape[0] > 0:
            patches = extract_patches_batch(image, pos)       # (N, 3, 32, 32)
            cnn_feats = encode_patches(encoder, patches)      # (N, 576)

            # Concatenate: [CNN(576) | SIFT(256) | pos(2)] = 834-d
            # We keep original SIFT for ablation and add CNN + spatial pos
            sift = g.x                                         # (N, 256)
            pos_feat = pos[:, :2]                              # (N, 2) normalized coords
            enhanced_x = torch.cat([cnn_feats, sift, pos_feat], dim=1)  # (N, 834)

            # Clone graph and replace features
            import copy
            g2 = copy.copy(g)
            g2.x = enhanced_x
            g2.original_x = sift          # keep original for ablation
            g2.cnn_x = cnn_feats          # keep CNN-only for ablation
            enhanced.append(g2)
        else:
            # Keep original graph if no image found
            enhanced.append(g)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (len(graphs) - idx - 1) / rate
            print(f"  [{split_name}] {idx+1}/{len(graphs)} | "
                  f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"  [{split_name}] Done: {len(enhanced)} graphs in {elapsed:.1f}s "
          f"(missing images: {missing_image})")
    return enhanced


def main():
    config = load_config()
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']
    out_dir = graph_dir  # Save alongside originals

    print(f"\n{'='*60}")
    print("  GNN++ FEATURE EXTRACTION (MobileNetV3 Patch Encoder)")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE}")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Feature dim: {FEATURE_DIM}-d CNN + 256-d SIFT + 2-d pos = 834-d")

    # Build encoder (cached weights already downloaded)
    print(f"\n  Loading MobileNetV3-Small...")
    encoder = build_patch_encoder()
    print(f"  Encoder ready on {DEVICE}")

    for split in ['train', 'val', 'test']:
        src_file = graph_dir / f'{split}_graphs.pt'
        dst_file = graph_dir / f'{split}_graphs_v2.pt'

        if dst_file.exists():
            print(f"\n  [{split}] Already exists at {dst_file.name}, skipping.")
            continue

        print(f"\n  Loading {src_file.name}...")
        graphs = torch.load(str(src_file), weights_only=False)
        print(f"  Loaded {len(graphs)} graphs")

        enhanced = process_split(graphs, split, encoder, out_dir)

        print(f"  Saving {dst_file.name}...")
        torch.save(enhanced, str(dst_file))
        print(f"  Saved {len(enhanced)} enhanced graphs")

        # Report feature dim
        if enhanced and hasattr(enhanced[0], 'x'):
            print(f"  Node feature dim: {enhanced[0].x.shape[1]}")

    print(f"\n{'='*60}")
    print("  Feature extraction complete!")
    print("  Ready to run: python scripts/train_gnn_plus_v2.py")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
