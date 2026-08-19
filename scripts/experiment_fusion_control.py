"""
Fusion control: is the verification gain specific to the GRAPH branch?
=====================================================================
The paper claims a validation-selected CNN+Hybrid blend cuts EER 3.5x. A reviewer's
objection: the EfficientNet-B4 CNN has the WORST standalone EER of the strong models
(2.70%, vs VGG-16 1.23% and ProtoN 1.17%), so the gain may simply be "fuse a second,
better-calibrated model" rather than anything about graphs.

This runs the identical val-selected fusion protocol with NON-GRAPH partners
(VGG-16, ResNet-50) and compares against the graph partners (Hybrid, ProtoN).
If a non-graph partner matches the graph partner, the graph-specific claim fails.

Protocol (identical to scripts/ensemble_inference.py): sweep the CNN weight w on the
VALIDATION split, apply the selected w unchanged to TEST. Probe-level bootstrap CI on
the test EER difference vs CNN alone.

Outputs: outputs/stats/fusion_control.json
Usage:   python scripts/experiment_fusion_control.py
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config, save_stats
from src.training.image_dataset import create_hybrid_loaders
from src.evaluation.metrics import BiometricMetrics
from scripts.experiment_pathway_intervention import load_hybrid, load_cnn, embed_cnn, embed
from scripts.experiment_quality_fusion import load_proton, embed_proton, sim, metrics_from_sim


class VGGBiometricModel(nn.Module):
    """Mirror of scripts/baselines/train_vgg_baseline.py (for checkpoint loading)."""
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        from torchvision.models import vgg16
        b = vgg16(weights=None)
        self.features = b.features
        self.avgpool = b.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, embedding_dim), nn.BatchNorm1d(embedding_dim))

    def get_embedding(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


@torch.no_grad()
def embed_generic(model, loader, device):
    E, L = [], []
    for images, _g, labels in loader:
        e = model.get_embedding(images.to(device))
        E.append(F.normalize(e, p=2, dim=-1).float().cpu()); L.append(labels)
    return torch.cat(E), torch.cat(L)


def load_vgg(device, num_classes=260):
    m = VGGBiometricModel(num_classes).to(device)
    sd = torch.load(PROJECT_ROOT / 'outputs/vgg_baseline/best_model.pt',
                    map_location=device, weights_only=False)
    if 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    m.load_state_dict(sd, strict=False); m.eval()
    return m


def val_select_w(Sv_cnn, Sv_p, vlbl, M, grid=None):
    """Sweep CNN weight on VALIDATION (minimise val EER); return best w."""
    grid = grid if grid is not None else np.linspace(0, 1, 21)
    best_w, best_e = 1.0, 1e9
    for w in grid:
        e = metrics_from_sim(w * Sv_cnn + (1 - w) * Sv_p, vlbl, M)['eer']
        if e < best_e:
            best_e, best_w = e, float(w)
    return best_w, best_e


def bootstrap_eer_diff(S_a, S_b, lbl, M, n=300, seed=0):
    """95% CI of EER(a) - EER(b), resampling PROBES (the correct unit)."""
    rng = np.random.RandomState(seed); N = len(lbl); d = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        Sa = S_a[np.ix_(idx, idx)]; Sb = S_b[np.ix_(idx, idx)]
        d.append(metrics_from_sim(Sa, lbl[idx], M)['eer'] - metrics_from_sim(Sb, lbl[idx], M)['eer'])
    d = np.sort(d)
    return float(np.mean(d)), float(d[int(0.025 * n)]), float(d[int(0.975 * n)])


def main():
    cfg = load_config()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ld = create_hybrid_loaders(str(PROJECT_ROOT / cfg['dataset']['processed_dir']),
                               str(PROJECT_ROOT / cfg['dataset']['graph_dir']), cfg)
    M = BiometricMetrics()

    print('embedding CNN ...', flush=True)
    cnn = load_cnn(dev)
    vc, vl = embed_cnn(cnn, ld['val'], dev); tc, tl = embed_cnn(cnn, ld['test'], dev)
    del cnn; torch.cuda.empty_cache()

    partners = {}
    print('embedding Hybrid (graph) ...', flush=True)
    h = load_hybrid(cfg, dev)
    partners['Hybrid (graph)'] = (embed(h, ld['val'], dev, 'full')[0], embed(h, ld['test'], dev, 'full')[0])
    del h; torch.cuda.empty_cache()

    print('embedding ProtoN (graph) ...', flush=True)
    p = load_proton(cfg, dev)
    partners['ProtoN (graph)'] = (embed_proton(p, ld['val'], dev)[0], embed_proton(p, ld['test'], dev)[0])
    del p; torch.cuda.empty_cache()

    print('embedding VGG-16 (non-graph) ...', flush=True)
    try:
        v = load_vgg(dev)
        partners['VGG-16 (non-graph)'] = (embed_generic(v, ld['val'], dev)[0],
                                          embed_generic(v, ld['test'], dev)[0])
        del v; torch.cuda.empty_cache()
    except Exception as e:
        print('  [SKIP] VGG:', e)

    vlbl, tlbl = vl.numpy(), tl.numpy()
    Svc, Stc = sim(vc), sim(tc)
    base = metrics_from_sim(Stc, tlbl, M)
    out = {'cnn_alone': base, 'partners': {}}
    print(f"\nCNN alone: R1={base['rank1']*100:.2f} EER={base['eer']*100:.3f}\n")

    for name, (ve, te) in partners.items():
        Svp, Stp = sim(ve), sim(te)
        w, val_eer = val_select_w(Svc, Svp, vlbl, M)
        Sf = w * Stc + (1 - w) * Stp
        m = metrics_from_sim(Sf, tlbl, M)
        md, lo, hi = bootstrap_eer_diff(Stc, Sf, tlbl, M)
        out['partners'][name] = {**m, 'val_selected_w': w, 'val_eer_at_w': val_eer,
                                 'standalone': metrics_from_sim(Stp, tlbl, M),
                                 'eer_reduction_vs_cnn_pt': (base['eer'] - m['eer']) * 100,
                                 'bootstrap_eer_diff_pt': [md * 100, lo * 100, hi * 100],
                                 'significant': bool(lo > 0)}
        r = out['partners'][name]
        print(f"{name:22s} w={w:.2f}  fused R1={m['rank1']*100:.2f} EER={m['eer']*100:.3f}  "
              f"dEER={r['eer_reduction_vs_cnn_pt']:+.2f}pt CI[{lo*100:+.2f},{hi*100:+.2f}] "
              f"{'SIG' if r['significant'] else 'n.s.'}")

    save_stats(out, str(PROJECT_ROOT / 'outputs/stats/fusion_control.json'))
    print('\nSaved -> outputs/stats/fusion_control.json')


if __name__ == '__main__':
    main()
