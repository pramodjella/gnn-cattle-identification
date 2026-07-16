"""
Part 2, Stage 1 of the Research Extension Plan: stratified attribution set.
==========================================================================
Builds the mentor's exact case mix from the test set:
  20 correct matches, 10 false accepts, 10 false rejects,
  10 CNN-correct/GNN-wrong, 10 GNN-correct/CNN-wrong,
and emits (a) a JSON manifest of the selected case indices and (b) a side-by-side
CNN Grad-CAM vs. GNN keypoint-importance montage (one representative per category).

Categories (closed-set self-similarity, self excluded):
  correct        = CNN rank-1 correct
  false_accept   = CNN rank-1 WRONG (nearest neighbour is a different identity)
  false_reject   = genuine mate exists but best genuine score < EER threshold
  cnn_only       = CNN correct & GNN wrong
  gnn_only       = GNN correct & CNN wrong

Outputs: outputs/stats/stage1_cases.json,
         outputs/figures/extension/fig_stage1_attribution.(png|pdf)
Usage: python scripts/experiment_stage1_attribution.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config, save_stats
from src.training.image_dataset import create_hybrid_loaders
from src.evaluation.metrics import BiometricMetrics
from scripts.evaluate_explainability import load_gnn, gradcam_importance, _last_gat_layer_name, _prep
from src.models.explainability import GradCAMGraph

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
GRID = 8
COUNTS = {'correct': 20, 'false_accept': 10, 'false_reject': 10, 'cnn_only': 10, 'gnn_only': 10}


def load_cnn(device):
    from src.models.cnn_model import CNNMuzzleModel
    ck = torch.load(PROJECT_ROOT / 'outputs/cnn/best_model.pt', map_location=device, weights_only=False)
    c = ck.get('config', {})
    m = CNNMuzzleModel(num_classes=ck.get('num_classes', 260), embedding_dim=c.get('embedding_dim', 512),
                       backbone=c.get('backbone', 'efficientnet_b4'), arcface_scale=c.get('arcface_scale', 128.0),
                       arcface_margin=c.get('arcface_margin', 0.35)).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


class CNNGradCAM:
    def __init__(self, model):
        self.model = model; self.act = self.grad = None
        layer = model.features[-1]
        layer.register_forward_hook(lambda m, i, o: setattr(self, 'act', o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grad', go[0].detach()))
        self.proto = F.normalize(model.arcface.arcface_head.weight, p=2, dim=1).detach()

    def cam(self, img, label):
        self.model.zero_grad()
        logit = (self.model(img)['embedding'] @ self.proto.t())[0, label]
        logit.backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1))[0]
        cam = F.interpolate(cam[None, None], size=(256, 256), mode='bilinear', align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy()


@torch.no_grad()
def cnn_embed(model, imgs, device, bs=64):
    out = []
    for i in range(0, imgs.size(0), bs):
        out.append(F.normalize(model.get_embedding(imgs[i:i+bs].to(device)), p=2, dim=-1).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def gnn_embed(model, graphs, device):
    E = []
    for g in graphs:
        E.append(F.normalize(model(_prep(g, device))['embedding'], p=2, dim=-1).squeeze(0).cpu())
    return torch.stack(E)


def eer_threshold(S, lbl):
    from sklearn.metrics import roc_curve
    M = BiometricMetrics(); gen, imp = M._get_score_distributions(S, lbl)
    fpr, tpr, thr = roc_curve([1]*len(gen)+[0]*len(imp), list(gen)+list(imp))
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float(thr[i])


def denorm(img):
    return (img.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = create_hybrid_loaders(str(PROJECT_ROOT / config['dataset']['processed_dir']),
                                    str(PROJECT_ROOT / config['dataset']['graph_dir']), config)
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']
    num_classes = len(json.load(open(graph_dir / 'label_mapping.json')))

    imgs = torch.cat([b[0] for b in loaders['test']])
    graphs = [g for b in loaders['test'] for g in b[1].to_data_list()]
    labels = torch.cat([b[2] for b in loaders['test']]).numpy()

    cnn = load_cnn(device); gnn = load_gnn('gnn_v3', config, device, num_classes)
    ce = cnn_embed(cnn, imgs, device); ge = gnn_embed(gnn, graphs, device)
    Sc = (ce @ ce.t()).numpy(); Sg = (ge @ ge.t()).numpy()
    np.fill_diagonal(Sc, -1e9); np.fill_diagonal(Sg, -1e9)

    cnn_pred = labels[Sc.argmax(1)]; gnn_pred = labels[Sg.argmax(1)]
    cnn_ok = cnn_pred == labels; gnn_ok = gnn_pred == labels
    thr = eer_threshold(Sc.copy(), labels)

    # best genuine score per probe (same identity, self excluded)
    best_gen = np.full(len(labels), -1e9)
    has_mate = np.zeros(len(labels), bool)
    for i in range(len(labels)):
        same = np.where(labels == labels[i])[0]
        same = same[same != i]
        if len(same):
            has_mate[i] = True; best_gen[i] = Sc[i, same].max()

    cats = {
        'correct': np.where(cnn_ok)[0],
        'false_accept': np.where(~cnn_ok)[0],
        'false_reject': np.where(has_mate & (best_gen < thr))[0],
        'cnn_only': np.where(cnn_ok & ~gnn_ok)[0],
        'gnn_only': np.where(gnn_ok & ~cnn_ok)[0],
    }
    rng = np.random.default_rng(0)
    manifest = {}
    for k, idx in cats.items():
        take = min(COUNTS[k], len(idx))
        manifest[k] = sorted(int(x) for x in rng.choice(idx, size=take, replace=False)) if take else []
        print(f"  {k:14s} available={len(idx):4d}  selected={len(manifest[k])}")
    save_stats({'eer_threshold': thr, 'counts_requested': COUNTS,
                'counts_selected': {k: len(v) for k, v in manifest.items()},
                'cases': manifest}, str(PROJECT_ROOT / 'outputs/stats/stage1_cases.json'))

    # montage: one representative per category, CNN Grad-CAM vs GNN keypoint importance
    cnn_gc = CNNGradCAM(cnn)
    gnn_gc = GradCAMGraph(gnn, target_layer_name=_last_gat_layer_name(gnn))
    rows = [k for k in COUNTS if manifest[k]]
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.5, 2.4 * len(rows)), dpi=170)
    if len(rows) == 1:
        axes = axes[None, :]
    for r, cat in enumerate(rows):
        i = manifest[cat][0]
        base = denorm(imgs[i])
        cam = cnn_gc.cam(imgs[i:i+1].to(device), int(labels[i]))
        imp = gnn_gc.attribute(_prep(graphs[i], device)).detach().cpu().numpy()
        pos = graphs[i].pos[:, :2].cpu().numpy() if getattr(graphs[i], 'pos', None) is not None else None
        axes[r, 0].imshow(base); axes[r, 0].set_ylabel(cat, fontsize=9)
        axes[r, 1].imshow(base); axes[r, 1].imshow(cam, cmap='jet', alpha=0.5)
        axes[r, 2].imshow(base)
        if pos is not None and len(imp) == len(pos):
            s = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
            axes[r, 2].scatter(pos[:, 0]*256, pos[:, 1]*256, c=s, cmap='jet', s=14, edgecolors='k', linewidths=0.2)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for c, t in enumerate(['muzzle image', 'CNN Grad-CAM', 'GNN keypoint importance']):
        axes[0, c].set_title(t, fontsize=10)
    gnn_gc.remove_hooks()
    fig.suptitle('Stage 1: CNN vs. GNN attribution across case types', fontweight='bold', y=1.005)
    fig.tight_layout()
    outdir = PROJECT_ROOT / 'outputs/figures/extension'; outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'fig_stage1_attribution.{ext}', bbox_inches='tight')
    plt.close(fig)
    print("\nSaved -> outputs/stats/stage1_cases.json + fig_stage1_attribution")


if __name__ == '__main__':
    main()
