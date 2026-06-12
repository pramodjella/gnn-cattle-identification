"""
Script: GradCAM Visualization (Explainability for CNN)
========================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to show which parts of the muzzle print are most important
for biometric identification.

Hook targets the last conv layer of EfficientNet-B4 (model.features[-1]).
Saves overlay visualizations to outputs/figures/gradcam/.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, ensure_dirs, set_seed
from src.models.cnn_model import CNNMuzzleModel
from src.training.augmentation import build_val_transform
from src.training.image_dataset import MuzzleImageDataset

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        
        # Forward pass
        # model(input_tensor) returns dict with 'embedding'
        out = self.model(input_tensor)
        emb = out['embedding']
        
        # Compute cosine similarities with class prototype weights
        # to get classification logits
        norm_weight = F.normalize(self.model.arcface.arcface_head.weight, p=2, dim=1)
        logits = torch.mm(emb, norm_weight.t())
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        score = logits[0, class_idx]
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pool gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
            
        return cam, class_idx

def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB colors
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    img_overlay = alpha * img + (1 - alpha) * (heatmap_color / 255.0)
    img_overlay = np.clip(img_overlay, 0, 1)
    
    return img_overlay

def main():
    parser = argparse.ArgumentParser(description="Generate GradCAM for CNN model")
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='outputs/figures/gradcam')
    args = parser.parse_args()

    config = load_config()
    set_seed(config['project']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ensure_dirs(args.output_dir)

    print(f"\n{'='*70}")
    print("  CNN GRAD-CAM VISUALIZATION GENERATOR")
    print(f"{'='*70}")
    print(f"  Device: {device}")

    # Load preprocessed directories
    preprocessed_dir = PROJECT_ROOT / config['dataset']['processed_dir']
    test_json = preprocessed_dir / 'test_split.json'
    
    if not test_json.exists():
        print(f"[ERROR] Test split JSON not found at {test_json}")
        sys.exit(1)

    image_size = config.get('preprocessing', {}).get('image_size', 256)
    val_transform = build_val_transform(image_size)
    dataset = MuzzleImageDataset(str(test_json), transform=val_transform)
    
    # Load Model
    cnn_ckpt = PROJECT_ROOT / 'outputs/cnn/best_model.pt'
    if not cnn_ckpt.exists():
        print(f"[ERROR] CNN best model checkpoint not found at {cnn_ckpt}")
        sys.exit(1)

    ckpt = torch.load(cnn_ckpt, map_location=device, weights_only=False)
    num_classes = ckpt.get('num_classes', 260)
    model = CNNMuzzleModel(
        num_classes=num_classes,
        embedding_dim=ckpt['config'].get('embedding_dim', 512),
        backbone=ckpt['config'].get('backbone', 'efficientnet_b4'),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[INFO] Loaded CNN model from checkpoint.")

    # We target the last convolutional layer block of EfficientNet-B4 features
    # features consists of Sequential blocks. features[-1] is the final conv block.
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    # Select random samples to visualize
    indices = np.random.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    print(f"[INFO] Visualizing {len(indices)} random samples...")

    for i, idx in enumerate(indices):
        image_tensor, label = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Load preprocessed image for overlay plotting (un-normalized)
        img_path, _ = dataset.samples[idx]
        img_pil = Image.open(img_path).convert('RGB')
        img_vis = np.array(img_pil).astype(np.float32) / 255.0
        
        # Generate heatmap
        heatmap, pred_idx = gradcam.generate_heatmap(image_tensor, class_idx=label.item())
        
        # Generate overlay
        overlay = overlay_heatmap(img_vis, heatmap, alpha=0.5)
        
        # Save visualization side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_vis)
        axes[0].set_title("Original Muzzle Image")
        axes[0].axis('off')
        
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM Heatmap (ID: {pred_idx})")
        axes[1].axis('off')
        
        plt.suptitle(f"CNN Explainability - Cattle ID {label.item()}", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = os.path.join(args.output_dir, f"gradcam_cattle_{label.item()}_idx{idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [SAVED] {save_path}")

    print(f"\n✅ Grad-CAM maps successfully generated in: {args.output_dir}")

if __name__ == '__main__':
    main()
