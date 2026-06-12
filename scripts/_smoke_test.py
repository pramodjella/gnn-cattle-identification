"""Smoke test for all three model architectures."""
import sys, torch
sys.path.insert(0, '.')
from src.utils import load_config
from src.models.arcface import ArcFaceLoss
from src.models.cnn_model import CNNMuzzleModel
from src.models.hybrid_model import HybridCNNGNN
from src.models.gnn_model import CattleGNN
from torch_geometric.data import Data, Batch

config = load_config()
device = torch.device('cuda')
torch.cuda.empty_cache()

# ── Test 1: ArcFace ──────────────────────────────────────────────────────────
print('[1] Testing ArcFace...')
arc = ArcFaceLoss(256, 260, margin=0.5, scale=64.0, triplet_weight=0.1).to(device)
emb = torch.randn(16, 256, device=device)
emb = emb / emb.norm(dim=1, keepdim=True)
lbl = torch.randint(0, 260, (16,), device=device)
loss, stats = arc(emb, lbl)
print(f'   ArcFace loss: {loss.item():.4f}  active_ratio: {stats["active_ratio"]:.2f} - OK')

# ── Test 2: CNN Model ─────────────────────────────────────────────────────────
print('[2] Testing CNN (EfficientNet-B3 + ArcFace)...')
cnn = CNNMuzzleModel(num_classes=260, embedding_dim=256, pretrained=False).to(device)
imgs = torch.randn(4, 3, 256, 256, device=device)
lbls = torch.randint(0, 260, (4,), device=device)
out = cnn(imgs, lbls)
emb_shape = out['embedding'].shape
loss_val = out['loss'].item()
print(f'   CNN embedding: {emb_shape}  loss: {loss_val:.4f} - OK')
del cnn
torch.cuda.empty_cache()

# ── Test 3: Hybrid CNN-GNN ────────────────────────────────────────────────────
print('[3] Testing Hybrid CNN-GNN...')
hybrid = HybridCNNGNN(num_classes=260, config=config, pretrained=False).to(device)
N = 30
graphs = []
for i in range(4):
    src = torch.randint(0, N, (N*4,))
    dst = torch.randint(0, N, (N*4,))
    g = Data(
        x=torch.randn(N, 256),
        edge_index=torch.stack([src, dst]),
        pos=torch.rand(N, 2),
        y=torch.tensor(i * 10),
    )
    graphs.append(g)
graph_batch = Batch.from_data_list(graphs).to(device)
imgs4 = torch.randn(4, 3, 256, 256, device=device)
lbls4 = torch.randint(0, 260, (4,), device=device)
out = hybrid(imgs4, graph_batch, lbls4)
print(f'   Hybrid embedding: {out["embedding"].shape}  loss: {out["loss"].item():.4f} - OK')
del hybrid
torch.cuda.empty_cache()

# ── Test 4: GNN+ with ArcFace ────────────────────────────────────────────────
print('[4] Testing GNN+ with ArcFace...')
gnn = CattleGNN(config=config)
gnn.set_num_classes(260)
gnn = gnn.to(device)
out = gnn(graph_batch)
arc_loss, _ = arc(out['embedding'], lbls4)
print(f'   GNN+ embedding: {out["embedding"].shape}  arc_loss: {arc_loss.item():.4f} - OK')

print()
print('=' * 50)
print('ALL MODELS VERIFIED SUCCESSFULLY')
print('=' * 50)
