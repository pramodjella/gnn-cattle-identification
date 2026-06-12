"""
Script: Attention Heatmap Visualization (Dual Explainability)
=============================================================
Shows what the models are "looking at" when identifying a cow.
Supports GNN+, GNN++, and Hybrid architectures.

For Hybrid:
  1. Captures GAT attention weights from the TRM.
  2. Projects CNN backbone spatial feature activations.
  3. Renders a Dual-Heatmap overlay on the muzzle photo.

This is the primary "explainability" figure for research papers.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.models.gnn_model import CattleGNN
from src.models.gnn_plus_v2 import CattleGNNPlusPlus
from src.models.hybrid_model import HybridCNNGNN
from src.training.image_dataset import create_hybrid_loaders


def get_model_and_config(args, config, device, num_classes):
    """Factory to load the requested model."""
    if args.model == 'gnn':
        model = CattleGNN(config=config)
        ckpt_dir = PROJECT_ROOT / config.get('training', {}).get('checkpoint_dir', 'outputs/gnn')
    elif args.model == 'gnn_plus_v2':
        # Need input_dim
        v2_cfg = config.get('gnn_plus_v2', {})
        model = CattleGNNPlusPlus(config=config, input_dim=v2_cfg.get('input_dim', 834))
        ckpt_dir = PROJECT_ROOT / v2_cfg.get('checkpoint_dir', 'outputs/gnn_plus_v2')
    elif args.model == 'hybrid':
        model = HybridCNNGNN(num_classes=num_classes, config=config, pretrained=False)
        ckpt_dir = PROJECT_ROOT / config.get('hybrid', {}).get('checkpoint_dir', 'outputs/hybrid')
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if hasattr(model, 'set_num_classes') and args.model != 'hybrid':
        model.set_num_classes(num_classes)

    ckpt_path = ckpt_dir / 'best_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Loaded {args.model.upper()} from epoch {checkpoint.get('epoch', '?')} | "
          f"Best Val R1: {checkpoint.get('val_r1', checkpoint.get('best_val_acc', 0)):.4f}")
    return model


def load_data(config, split='test'):
    """Load preprocessed images and graphs."""
    preprocessed_dir = PROJECT_ROOT / config['dataset']['processed_dir']
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']
    
    # We use create_hybrid_loaders to get matched images and graphs
    loaders = create_hybrid_loaders(str(preprocessed_dir), str(graph_dir), config)
    loader = loaders[split]
    print(f"[INFO] Loaded {len(loader.dataset)} {split} samples")
    return loader


def extract_attention(model, image, graph, device, model_type):
    """
    Run model and extract attention.
    Returns: embedding, node_importance, cnn_activation_map
    """
    image = image.unsqueeze(0).to(device)
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([graph]).to(device)

    cnn_map = None
    node_importance = None
    embedding = None

    with torch.no_grad():
        if model_type == 'hybrid':
            # Run CNN
            fmaps = model.cnn_features(image) # (1, 1536, 8, 8)
            # Average across channels to get spatial activation map
            cnn_map = fmaps.mean(dim=1).squeeze(0).cpu().numpy() # (8, 8)
            # Normalize CNN map
            if cnn_map.max() > cnn_map.min():
                cnn_map = (cnn_map - cnn_map.min()) / (cnn_map.max() - cnn_map.min())

            # Run full hybrid
            out = model(image, batch)
            embedding = out['embedding'][0].cpu()
            attn = out['attention']
        else:
            # GNN-only
            out = model(batch)
            embedding = out['embedding'][0].cpu()
            attn = out['attention']

    # Aggregate GNN attention
    if attn is not None:
        attn_weights = attn.cpu().float()
        if attn_weights.dim() == 2:
            attn_weights = attn_weights.mean(dim=1)
            
        # Distribute to nodes
        num_nodes = graph.x.shape[0]
        node_importance = torch.zeros(num_nodes)
        dst_nodes = graph.edge_index[1].cpu()
        for e, dst in enumerate(dst_nodes):
            node_importance[dst] += attn_weights[e]

        if node_importance.max() > 0:
            node_importance = node_importance / node_importance.max()
        node_importance = node_importance.numpy()

    return embedding, node_importance, cnn_map


def render_dual_heatmap(original_image, keypoints_xy, node_importance, cnn_map, title="", save_path=None):
    """
    Overlay GNN attention and CNN activation on the muzzle image.
    If cnn_map is None, it renders just the GNN attention.
    """
    H, W = original_image.shape[:2]
    
    # 1. Image preparation
    if original_image.dtype == np.uint8:
        img_float = original_image[:, :, ::-1].astype(np.float32) / 255.0
    else:
        img_float = original_image.astype(np.float32)
        if img_float.max() > 1:
            img_float /= 255.0

    # 2. GNN Node Heatmap
    gnn_heatmap = np.zeros((H, W), dtype=np.float32)
    radius = max(H, W) // 15
    if node_importance is not None:
        for (x, y), imp in zip(keypoints_xy, node_importance):
            x, y = int(x), int(y)
            x0, x1 = max(0, x - radius), min(W, x + radius + 1)
            y0, y1 = max(0, y - radius), min(H, y + radius + 1)
            xv, yv = np.mgrid[x0:x1, y0:y1]
            gauss = np.exp(-((xv - x)**2 + (yv - y)**2) / (2 * (radius/3)**2))
            gnn_heatmap[y0:y1, x0:x1] += (gauss.T * float(imp))
        if gnn_heatmap.max() > 0:
            gnn_heatmap /= gnn_heatmap.max()

    # 3. Plotting
    num_cols = 4 if cnn_map is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.05)

    try:
        cmap = plt.colormaps.get_cmap('jet')
    except AttributeError:
        cmap = cm.get_cmap('jet')

    # Original
    axes[0].imshow(img_float)
    axes[0].set_title("Original Muzzle Image", fontsize=13)
    axes[0].axis('off')

    # GNN Points
    axes[1].imshow(img_float)
    if node_importance is not None:
        scatter = axes[1].scatter(
            keypoints_xy[:, 0], keypoints_xy[:, 1],
            c=node_importance, cmap='RdYlGn', s=30, alpha=0.9, vmin=0, vmax=1, edgecolors='k', linewidth=0.5
        )
        plt.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04, label='TRM Attention Weight')
    axes[1].set_title("GNN Topological Attention", fontsize=13)
    axes[1].axis('off')

    # GNN Overlay
    gnn_colored = cmap(gnn_heatmap)[:, :, :3]
    gnn_blend = np.clip(0.4 * img_float + 0.6 * gnn_colored, 0, 1)
    axes[2].imshow(gnn_blend)
    axes[2].set_title("GNN Spatial Heatmap", fontsize=13)
    axes[2].axis('off')

    # CNN Overlay (if Hybrid)
    if cnn_map is not None:
        # Resize CNN map to image resolution
        cnn_resized = cv2.resize(cnn_map, (W, H), interpolation=cv2.INTER_CUBIC)
        cnn_colored = cmap(cnn_resized)[:, :, :3]
        cnn_blend = np.clip(0.4 * img_float + 0.6 * cnn_colored, 0, 1)
        axes[3].imshow(cnn_blend)
        axes[3].set_title("CNN Backbone Activation", fontsize=13)
        axes[3].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [SAVED] {save_path}")
    plt.close(fig)


def get_keypoint_positions(graph, image_shape):
    """Extract (x, y) pixel coordinates from the graph."""
    H, W = image_shape[:2]
    if hasattr(graph, 'pos') and graph.pos is not None:
        pos = graph.pos.cpu().numpy()
        if pos.max() <= 1.0:
            pos[:, 0] *= W
            pos[:, 1] *= H
        return pos[:, :2]
    return np.zeros((graph.x.shape[0], 2))


def visualize_samples(model, loader, config, device, args):
    """Generate attention visualizations for a random selection of samples."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Unpack all data
    dataset = loader.dataset
    indices = np.random.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    
    print(f"\n[INFO] Visualizing {len(indices)} samples for {args.model.upper()}...")
    
    for i, idx in enumerate(indices):
        image, graph, label = dataset[idx]
        animal_id = f"animal_{int(label):04d}"
        print(f"  [{i+1}/{len(indices)}] {animal_id}")
        
        # De-normalize image for visualization
        img_np = image.numpy().transpose(1, 2, 0) # (H, W, 3)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = std * img_np + mean
        img_vis = np.clip(img_vis, 0, 1)
        
        # Extract Attention
        embedding, node_importance, cnn_map = extract_attention(model, image, graph, device, args.model)
        keypoints_xy = get_keypoint_positions(graph, img_vis.shape)
        
        num_nodes = graph.x.shape[0]
        title = f"Dual Explainability - {animal_id} (Nodes: {num_nodes})" if args.model == 'hybrid' else f"GNN Explainability - {animal_id} (Nodes: {num_nodes})"
        save_path = os.path.join(args.output_dir, f"{args.model}_attention_{animal_id}_idx{idx}.png")
        
        render_dual_heatmap(img_vis, keypoints_xy, node_importance, cnn_map, title, save_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize Attention Weights")
    parser.add_argument('--model', type=str, default='hybrid', choices=['gnn', 'gnn_plus_v2', 'hybrid'])
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='outputs/figures/attention')
    parser.add_argument('--tsne', action='store_true')
    args = parser.parse_args()

    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure label mapping exists
    label_map_path = PROJECT_ROOT / config['dataset']['graph_dir'] / "label_mapping.json"
    with open(label_map_path) as f:
        num_classes = len(json.load(f))

    # Load Model and Data
    model = get_model_and_config(args, config, device, num_classes)
    loader = load_data(config, split=args.split)
    
    # Visualize
    visualize_samples(model, loader, config, device, args)
    
    if args.tsne:
        print("\n  [WARN] t-SNE plot is available in compare_models.py or requires full embedding extraction.")

if __name__ == "__main__":
    main()
