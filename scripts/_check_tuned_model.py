"""Quick sanity check for tuned CNN model."""
import sys
sys.path.insert(0, 'F:/GNN Research/gnn-cattle-identification')

from src.utils import load_config
from src.models.cnn_model import CNNMuzzleModel
import torch

config = load_config()
cnn_cfg = config.get('cnn', {})

print('=== Config Sanity Check ===')
print(f'Backbone:       {cnn_cfg.get("backbone", "NOT SET")}')
print(f'Embedding dim:  {cnn_cfg.get("embedding_dim", "NOT SET")}')
print(f'Epochs:         {cnn_cfg.get("epochs", "NOT SET")}')
print(f'ArcFace scale:  {cnn_cfg.get("arcface_scale", "NOT SET")}')
print(f'ArcFace margin: {cnn_cfg.get("arcface_margin", "NOT SET")}')
print(f'SWA:            {cnn_cfg.get("use_swa", False)} from epoch {cnn_cfg.get("swa_start_epoch", 100)}')
print(f'Mixup:          {cnn_cfg.get("use_mixup", False)}')
print(f'Patience:       {cnn_cfg.get("early_stopping", {}).get("patience", "NOT SET")}')
print()

print('=== Model Creation Test ===')
model = CNNMuzzleModel(
    num_classes=260,
    embedding_dim=512,
    backbone='efficientnet_b4',
    arcface_scale=128.0,
    arcface_margin=0.35,
    label_smoothing=0.05,
)
model.summary()

print()
print('=== Test forward pass ===')
x = torch.randn(2, 3, 256, 256)
labels = torch.tensor([0, 1])
out = model(x, labels)
print(f'Embedding shape: {out["embedding"].shape}')
print(f'Loss:            {out["loss"].item():.4f}')
print('ALL OK')
