"""
Figure 1 — Hybrid CNN-GNN architecture (publication quality).
=============================================================
Landscape, column-friendly schematic with two clearly separated input lanes
(image -> backbone; keypoints -> graph) that converge at the bilinear-sampling
step, then a single graph-reasoning chain to the ArcFace embedding.

Design: restrained 3-hue semantic palette (CNN path / sampling bridge / graph
head), consistent box geometry, orthogonal arrow routing, no embedded title
(the LaTeX caption carries it), vector PDF output.

Outputs: outputs/figures/architecture/hybrid_architecture.{pdf,png}
Usage:   python scripts/figures/generate_architecture_figure.py
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / 'outputs' / 'figures' / 'architecture'
OUT.mkdir(parents=True, exist_ok=True)

# ── restrained semantic palette ──────────────────────────────────────────────
CNN_FC, CNN_EC = '#DCE6F2', '#3B6091'      # image / CNN path (blue)
KP_FC,  KP_EC = '#E6E2F2', '#5B4E93'       # keypoint path (violet)
BRG_FC, BRG_EC = '#FBE8CF', '#B5741B'      # the bridge (amber, the key idea)
GNN_FC, GNN_EC = '#DDEDE3', '#2E7150'      # graph head (green)
OUT_FC, OUT_EC = '#EDE6DC', '#6B5744'      # output (neutral)
INK = '#1B2430'
ARROW = '#4A5462'

FS_LABEL, FS_SUB = 8.6, 7.3


def box(ax, x, y, w, h, title, sub=None, fc='#FFF', ec='#333', bold=False, lw=1.1):
    """Rounded box with a title and optional sub-label, vertically laid out so
    multi-line titles and sub-labels never collide."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    cx, cy = x + w / 2, y + h / 2
    if not sub:
        ax.text(cx, cy, title, ha='center', va='center', fontsize=FS_LABEL,
                color=INK, fontweight='bold' if bold else 'normal', zorder=3)
        return
    # allocate space proportional to the number of text lines in each part
    nt, ns = title.count('\n') + 1, sub.count('\n') + 1
    lh_t, lh_s = 0.105, 0.088                      # per-line height in axis units
    block = nt * lh_t + ns * lh_s + 0.035          # total text block height
    top = cy + block / 2
    ax.text(cx, top - nt * lh_t / 2, title, ha='center', va='center',
            fontsize=FS_LABEL, color=INK, linespacing=1.25,
            fontweight='bold' if bold else 'normal', zorder=3)
    ax.text(cx, top - nt * lh_t - 0.035 - ns * lh_s / 2, sub, ha='center',
            va='center', fontsize=FS_SUB, color='#5A6472', linespacing=1.25, zorder=3)


def arrow(ax, p, q, style='-|>', rad=0.0, lw=1.15, color=ARROW):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=9, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(9.6, 3.05), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.05); ax.axis('off')

    H = 0.52          # standard box height
    yT, yB = 2.28, 1.42   # top lane (image), bottom lane (keypoints)

    # ── lane 1: image → backbone → feature map ──
    box(ax, 0.10, yT, 1.34, H, 'Muzzle image', '256 × 256 × 3', CNN_FC, CNN_EC)
    box(ax, 1.72, yT, 1.52, H, 'EfficientNet-B3', 'shared backbone', CNN_FC, CNN_EC)
    box(ax, 3.52, yT, 1.34, H, 'Feature map', '1536 × 8 × 8', CNN_FC, CNN_EC)
    arrow(ax, (1.44, yT + H / 2), (1.72, yT + H / 2))
    arrow(ax, (3.24, yT + H / 2), (3.52, yT + H / 2))

    # ── lane 2: keypoints → k-NN graph ──
    box(ax, 0.10, yB, 1.34, H, 'DISK keypoints', 'N ≤ 128  (x, y)', KP_FC, KP_EC)
    box(ax, 1.72, yB, 1.52, H, 'k-NN graph', 'k = 8, edge attr.', KP_FC, KP_EC)
    box(ax, 3.52, yB, 1.34, H, 'Node positions', 'normalised', KP_FC, KP_EC)
    arrow(ax, (1.44, yB + H / 2), (1.72, yB + H / 2))
    arrow(ax, (3.24, yB + H / 2), (3.52, yB + H / 2))

    # ── the bridge: bilinear sampling (both lanes converge) ──
    bx, by, bw, bh = 5.16, 1.63, 1.46, 1.10
    box(ax, bx, by, bw, bh, 'Bilinear\nsampling', 'features at\nkeypoints',
        BRG_FC, BRG_EC, bold=True, lw=1.5)
    arrow(ax, (4.86, yT + H / 2), (bx, by + bh * 0.72))
    arrow(ax, (4.86, yB + H / 2), (bx, by + bh * 0.28))

    # ── graph head chain ──
    yG = 1.90
    box(ax, 6.86, yG, 1.30, H, 'EdgeConv', 'dynamic k-NN', GNN_FC, GNN_EC)
    box(ax, 8.44, yG, 1.42, H, 'GATv2 relation', '4 heads', GNN_FC, GNN_EC)
    arrow(ax, (bx + bw, by + bh / 2), (6.86, yG + H / 2))
    arrow(ax, (8.16, yG + H / 2), (8.44, yG + H / 2))

    # pooling + embedding (returns along the lower right)
    yE = 0.72
    box(ax, 8.44, yE, 1.42, H, 'Mean + max pool', '512-d', OUT_FC, OUT_EC)
    box(ax, 6.60, yE, 1.56, H, 'Embedding', '512-d · ArcFace', OUT_FC, OUT_EC, bold=True)
    arrow(ax, (9.15, yG), (9.15, yE + H))
    arrow(ax, (8.44, yE + H / 2), (8.16, yE + H / 2))

    # ── annotation: the mechanism the paper interrogates ──
    ax.annotate('node features carry the identity signal\n(geometry is causally inert — §3.6)',
                xy=(bx + bw / 2, by), xytext=(4.30, 0.42),
                fontsize=7.0, color='#7A5320', ha='center', va='center',
                arrowprops=dict(arrowstyle='-', color='#C79A5B', lw=0.8,
                                connectionstyle='arc3,rad=0.22'))

    # lane captions
    ax.text(0.10, yT + H + 0.10, 'I M A G E   P A T H', fontsize=6.8, color=CNN_EC,
            fontweight='bold')
    ax.text(0.10, yB - 0.20, 'K E Y P O I N T   P A T H', fontsize=6.8, color=KP_EC,
            fontweight='bold')

    fig.tight_layout(pad=0.25)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'hybrid_architecture.{ext}', bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print('saved ->', OUT / 'hybrid_architecture.pdf')


if __name__ == '__main__':
    main()
