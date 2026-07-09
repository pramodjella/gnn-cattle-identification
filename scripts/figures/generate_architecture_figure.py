"""
Publication-quality Hybrid CNN-GNN architecture diagram.
Replaces the ASCII-art block in the manuscript.
Outputs: outputs/figures/architecture/hybrid_architecture.(png|pdf)
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / 'outputs' / 'figures' / 'architecture'
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8.4, 6.6), dpi=220)
ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis('off')

C = {'in': '#dbeafe', 'cnn': '#bfdbeef', 'sample': '#fde68a', 'gnn': '#bbf7d0',
     'pool': '#e9d5ff', 'loss': '#fecaca'}
C['cnn'] = '#c7ddf5'
EDGE = '#334155'

def box(x, y, w, h, text, fill, fs=10, bold=False):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=1.3, edgecolor=EDGE, facecolor=fill)
    ax.add_patch(b)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color='#0f172a')

def arrow(x1, y1, x2, y2, style='-|>'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 linewidth=1.5, color=EDGE, shrinkA=2, shrinkB=2))

# --- two input branches (top) ---
box(2.6, 11.6, 3.6, 0.9, "Muzzle Image\n(256 x 256 x 3)", C['in'], 9)
box(7.4, 11.6, 3.6, 0.9, "DISK Keypoints\n(N x 2 coords)", C['in'], 9)

box(2.6, 10.0, 3.6, 0.9, "EfficientNet-B3\nBackbone (shared)", C['cnn'], 9)
box(2.6, 8.5, 3.6, 0.85, "Feature Map\n(1536 x 8 x 8)", C['cnn'], 9)

arrow(2.6, 11.15, 2.6, 10.45)
arrow(2.6, 9.55, 2.6, 8.93)

# --- bilinear sampling (merge) ---
box(5.0, 6.9, 5.4, 0.95, "Bilinear Feature Sampling\nat keypoint locations", C['sample'], 10, True)
arrow(2.6, 8.07, 3.9, 7.38)      # feature map -> sampling
arrow(7.4, 11.15, 7.4, 7.9)      # keypoints down
arrow(7.4, 7.9, 6.1, 7.38)       # keypoints -> sampling

# --- GNN head (vertical) ---
box(5.0, 5.5, 4.6, 0.8, "Node Features (1536-d) -> Proj (256-d)", C['gnn'], 9)
box(5.0, 4.2, 4.6, 0.8, "Dynamic EdgeConv Blocks (-> 512-d)", C['gnn'], 9)
box(5.0, 2.9, 4.6, 0.8, "Topological Relation Module\n(GATv2, 4 heads)", C['gnn'], 9)
box(5.0, 1.6, 4.6, 0.8, "Global Mean + Max Pool (512-d)", C['pool'], 9)
box(5.0, 0.45, 4.6, 0.75, "Projection Head -> Embedding (256-d)\n+ ArcFace Loss", C['loss'], 9, True)

for y1, y2 in [(6.42, 5.9), (5.1, 4.6), (3.8, 3.3), (2.5, 2.0), (1.2, 0.83)]:
    arrow(5.0, y1, 5.0, y2)

ax.text(5.0, 12.15, "Hybrid CNN-GNN: CNN feature sampling at keypoints + graph reasoning",
        ha='center', fontsize=11, fontweight='bold', color='#0f172a')

fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'hybrid_architecture.{ext}', bbox_inches='tight')
plt.close(fig)
print('saved', OUT / 'hybrid_architecture.png')
