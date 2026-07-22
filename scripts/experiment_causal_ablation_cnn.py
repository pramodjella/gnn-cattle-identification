"""
Part 2, Stage 2 (CNN branch) of the Research Extension Plan: causal Grad-CAM
region ablation on the FULL test set.
============================================================================
For every test image we compute a Grad-CAM heatmap (EfficientNet-B4, last conv
block, target = cosine similarity to the matched-identity ArcFace prototype),
then mask the top-k% / random-k% / bottom-k% most important 32x32 regions
(k in {10,20,30}; masked pixels set to the ImageNet mean) and re-embed. A
causally faithful map implies top-region masking hurts more than random/low.
Measures: dcosine, top-1 flip rate, Rank-1 drop, EER increase (+ bootstrap CIs).

Outputs: outputs/stats/causal_ablation_cnn.json
Usage: python scripts/experiment_causal_ablation_cnn.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config, save_stats
from src.training.image_dataset import create_hybrid_loaders
from src.evaluation.metrics import BiometricMetrics

GRID = 8  # Grad-CAM grid resolution (EfficientNet-B4 @ 256 -> 8x8)


def bootstrap_ci(vec, n_boot=2000, seed=0):
    """95% percentile CI of the mean of `vec` by case resampling. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vec, float)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_cnn(device):
    """Load the trained EfficientNet-B4 CNN (with ArcFace head) in eval mode."""
    from src.models.cnn_model import CNNMuzzleModel
    ck = torch.load(PROJECT_ROOT / 'outputs/cnn/best_model.pt', map_location=device, weights_only=False)
    c = ck.get('config', {})
    m = CNNMuzzleModel(num_classes=ck.get('num_classes', 260), embedding_dim=c.get('embedding_dim', 512),
                       backbone=c.get('backbone', 'efficientnet_b4'), arcface_scale=c.get('arcface_scale', 128.0),
                       arcface_margin=c.get('arcface_margin', 0.35)).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


class GradCAM:
    """Per-image Grad-CAM on the CNN's last conv block -> GRIDxGRID map."""
    def __init__(self, model):
        """Register forward/backward hooks on the CNN's last conv block and cache
        the L2-normalised ArcFace class prototypes as the Grad-CAM targets."""
        self.model = model
        self.act = self.grad = None
        layer = model.features[-1]
        layer.register_forward_hook(lambda m, i, o: setattr(self, 'act', o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grad', go[0].detach()))
        self.proto = F.normalize(model.arcface.arcface_head.weight, p=2, dim=1).detach()

    def cam(self, img, label):
        """Grad-CAM for one image w.r.t. its matched-identity prototype.
        In: img (1,3,H,W), label int; Out: (GRID,GRID) importance map (numpy)."""
        self.model.zero_grad()
        emb = self.model(img)['embedding']
        logit = (emb @ self.proto.t())[0, label]
        logit.backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)      # (1,C,1,1)
        cam = F.relu((w * self.act).sum(1))[0]            # (H,W)
        cam = F.interpolate(cam[None, None], size=(GRID, GRID), mode='area')[0, 0]
        return cam.cpu().numpy()


@torch.no_grad()
def embed_batch(model, imgs, device, bs=64):
    """Embed a stack of images in mini-batches. In: (N,3,H,W); Out: (N,D)
    L2-normalised embeddings on CPU."""
    out = []
    for i in range(0, imgs.size(0), bs):
        e = model.get_embedding(imgs[i:i + bs].to(device))
        out.append(F.normalize(e, p=2, dim=-1).float().cpu())
    return torch.cat(out)


def mask_cells(imgs, cams, frac, strat, rng):
    """Zero the top/random/bottom `frac` of GRIDxGRID cells (per image)."""
    N = imgs.size(0); cell = imgs.size(-1) // GRID
    k = max(1, int(round(frac * GRID * GRID)))
    out = imgs.clone()
    for i in range(N):
        flat = cams[i].reshape(-1)
        order = np.argsort(-flat)               # most -> least important
        if strat == 'top':
            cells = order[:k]
        elif strat == 'bottom':
            cells = order[-k:]
        else:
            cells = rng.choice(GRID * GRID, size=k, replace=False)
        for c in cells:
            r, cc = divmod(int(c), GRID)
            out[i, :, r*cell:(r+1)*cell, cc*cell:(cc+1)*cell] = 0.0  # 0 == ImageNet mean
    return out


def main():
    """Grad-CAM region ablation on the full test set: mask top-/random-/bottom-k%
    of 32x32 CNN regions and measure dcos, flip, Rank-1 drop, EER rise (with
    bootstrap CIs); save outputs/stats/causal_ablation_cnn.json."""
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = create_hybrid_loaders(str(PROJECT_ROOT / config['dataset']['processed_dir']),
                                    str(PROJECT_ROOT / config['dataset']['graph_dir']), config)
    model = load_cnn(device)
    gc = GradCAM(model)
    M = BiometricMetrics()

    imgs = torch.cat([b[0] for b in loaders['test']])
    labels = torch.cat([b[2] for b in loaders['test']])
    print(f"[INFO] {imgs.size(0)} test images")

    full_emb = embed_batch(model, imgs, device)
    full_summary = M.compute_all_metrics(full_emb, labels)['summary']
    full_r1, full_eer = full_summary['rank_1_accuracy'], full_summary['eer']
    Sf = (full_emb @ full_emb.t()).numpy(); np.fill_diagonal(Sf, -1e9)
    full_pred = labels[Sf.argmax(1)]

    print("[INFO] computing Grad-CAM maps ...")
    cams = np.stack([gc.cam(imgs[i:i+1].to(device), int(labels[i])) for i in range(imgs.size(0))])

    rng = np.random.default_rng(0)
    results = {'model': 'cnn_efficientnet_b4', 'n': int(imgs.size(0)),
               'full': {'rank1': full_r1, 'eer': full_eer}, 'conditions': {}}
    for frac in (0.10, 0.20, 0.30):
        for strat in ('top', 'random', 'bottom'):
            masked = mask_cells(imgs, cams, frac, strat, rng)
            abl = embed_batch(model, masked, device)
            dcos_vec = (1 - (abl * full_emb).sum(1)).clamp(min=0).numpy()
            Sab = (abl @ full_emb.t()).numpy(); np.fill_diagonal(Sab, -1e9)
            flip_vec = (labels[Sab.argmax(1)] != full_pred).numpy().astype(float)
            s = M.compute_all_metrics(abl, labels)['summary']
            d_lo, d_hi = bootstrap_ci(dcos_vec); f_lo, f_hi = bootstrap_ci(flip_vec)
            key = f'{strat}_{int(frac*100)}'
            results['conditions'][key] = {
                'dcosine': float(dcos_vec.mean()), 'dcosine_ci': [d_lo, d_hi],
                'top1_flip': float(flip_vec.mean()), 'top1_flip_ci': [f_lo, f_hi],
                'rank1_drop': (full_r1 - s['rank_1_accuracy']) * 100,
                'eer_incr': (s['eer'] - full_eer) * 100}
            print(f"  {key:10s} dcos={dcos_vec.mean():.4f} [{d_lo:.4f},{d_hi:.4f}]  "
                  f"flip={flip_vec.mean()*100:5.1f}%  R1drop={results['conditions'][key]['rank1_drop']:+5.1f}  "
                  f"EER+={results['conditions'][key]['eer_incr']:+5.2f}")

    save_stats(results, str(PROJECT_ROOT / 'outputs/stats/causal_ablation_cnn.json'))
    print("\nSaved -> outputs/stats/causal_ablation_cnn.json")


if __name__ == '__main__':
    main()
