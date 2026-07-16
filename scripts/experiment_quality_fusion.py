"""
Part 1 of the Research Extension Plan: Quality-Aware CNN-Hybrid Fusion under corruption.
=======================================================================================
Implements the mentor's plan verbatim:
  score_final = alpha(x) * score_CNN + (1 - alpha(x)) * score_Hybrid
Six models compared: (1) CNN only, (2) Hybrid only, (3) fixed 50:50, (4) val-tuned
fixed alpha, (5) quality-aware rule-based, (6) quality-aware learned (logistic).

Data protocol (NO test leakage): the fixed alpha and the learned gate are chosen
ONLY on the (corrupted) VALIDATION set; the (clean and corrupted) TEST set is
untouched until final scoring. Corruption applies to the IMAGE, so the Hybrid's
CNN backbone degrades too (not a clean-graph oracle) -- keypoint positions remain
from the clean graph (a small, documented optimistic bias under occlusion).

Evaluation: clean + corrupted test (blur, brightness/fog, spatter) at severities 1,3,5.
Metrics: Rank-1, Rank-5, EER, ROC-AUC (self-similarity closed-set).
Outputs: outputs/stats/quality_fusion_results.json (feeds severity curves).

Usage: python scripts/experiment_quality_fusion.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config, save_stats
from src.training.image_dataset import create_hybrid_loaders
from src.evaluation.metrics import BiometricMetrics
from src.evaluation import corruptions as corr
from src.evaluation.quality import image_quality, graph_quality, branch_confidence, to_vector

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SHIFTS = [('clean', 0), ('blur', 1), ('blur', 3), ('blur', 5),
          ('brightness', 1), ('brightness', 3), ('brightness', 5),
          ('spatter', 1), ('spatter', 3), ('spatter', 5)]


def load_models(config, device):
    from src.models.cnn_model import CNNMuzzleModel
    from src.models.hybrid_model import HybridCNNGNN
    ck = torch.load(PROJECT_ROOT / 'outputs/cnn/best_model.pt', map_location=device, weights_only=False)
    mc = ck.get('config', {})
    cnn = CNNMuzzleModel(num_classes=ck.get('num_classes', 260), embedding_dim=mc.get('embedding_dim', 512),
                         backbone=mc.get('backbone', 'efficientnet_b4'), arcface_scale=mc.get('arcface_scale', 128.0),
                         arcface_margin=mc.get('arcface_margin', 0.35)).to(device)
    cnn.load_state_dict(ck['model_state_dict']); cnn.eval()
    hk = torch.load(PROJECT_ROOT / 'outputs/hybrid/best_model.pt', map_location=device, weights_only=False)
    hyb = HybridCNNGNN(num_classes=hk.get('num_classes', 260), config=config, pretrained=False).to(device)
    hyb.load_state_dict(hk['model_state_dict']); hyb.eval()
    return cnn, hyb


def load_proton(config, device, num_classes=260):
    """Load ProtoN (graph-only model) for the CNN+ProtoN fusion ablation."""
    from src.models.proton import CattleProtoN
    ck = torch.load(PROJECT_ROOT / 'outputs/proton/best_model.pt', map_location=device, weights_only=False)
    pc = config.get('proton', {})
    m = CattleProtoN(num_classes=ck.get('num_classes', num_classes),
                     hidden_dim=pc.get('hidden_dim', 128), num_heads=pc.get('num_heads', 4),
                     num_layers=pc.get('num_layers', 4), fusion_dim=pc.get('fusion_dim', 512),
                     embedding_dim=pc.get('embedding_dim', 256), dropout=pc.get('dropout', 0.12)).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


@torch.no_grad()
def embed_proton(model, loader, device):
    """ProtoN embeddings (graph node features are precomputed -> clean-graph)."""
    E, L = [], []
    for _, graphs, labels in loader:
        out = model(graphs.to(device))
        E.append(F.normalize(out['embedding'], p=2, dim=-1).float().cpu()); L.append(labels)
    return torch.cat(E), torch.cat(L)


# feature-group index sets within FEATURE_ORDER (for gate ablations)
IMG_Q_IDX = [0, 1, 2]           # blur, brightness, contrast
GRAPH_Q_IDX = [3, 4, 5, 6]      # node_count, avg_degree, n_components, edge_len_var
DISAGREE_IDX = [11]             # branch disagreement


def corrupt_batch(images_cpu, kind, sev):
    if kind == 'clean' or sev == 0:
        return images_cpu
    out = torch.empty_like(images_cpu)
    for i in range(images_cpu.size(0)):
        x01 = (images_cpu[i] * STD + MEAN).clamp(0, 1)
        out[i] = (corr.apply(x01, kind, sev, seed=i) - MEAN) / STD
    return out


@torch.no_grad()
def embed_shift(cnn, hyb, loader, device, kind, sev):
    """Return CNN emb, Hybrid emb, labels, and per-sample quality-feature vectors."""
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ce, he, lbls, quals = [], [], [], []
    for images, graphs, labels in loader:
        images_c = corrupt_batch(images, kind, sev)
        img = images_c.to(device); g = graphs.to(device)
        c = F.normalize(cnn.get_embedding(img), p=2, dim=-1).float().cpu()
        with autocast(device_type='cuda', dtype=amp, enabled=(device.type == 'cuda')):
            ho = hyb(img, g)
        h = F.normalize(ho['embedding'], p=2, dim=-1).float().cpu()
        ce.append(c); he.append(h); lbls.append(labels)
        # per-sample quality (image + graph); confidence added after sims known
        gcpu = graphs.cpu(); nb = gcpu.batch
        for i in range(images.size(0)):
            q = image_quality(images_c[i])
            mask = nb == i
            sub = _subgraph(gcpu, mask); q.update(graph_quality(sub)); quals.append(q)
    return torch.cat(ce), torch.cat(he), torch.cat(lbls), quals


class _GV: pass
def _subgraph(batch, mask):
    v = _GV(); idx = mask.nonzero(as_tuple=True)[0]
    v.x = batch.x[idx]; remap = -torch.ones(batch.x.size(0), dtype=torch.long); remap[idx] = torch.arange(idx.size(0))
    ei = batch.edge_index; em = mask[ei[0]] & mask[ei[1]]
    v.edge_index = torch.stack([remap[ei[0][em]], remap[ei[1][em]]])
    v.pos = batch.pos[idx] if getattr(batch, 'pos', None) is not None else None
    return v


def sim(emb):
    e = F.normalize(emb, p=2, dim=-1); return (e @ e.t()).numpy()


def metrics_from_sim(S, lbl, m):
    from sklearn.metrics import roc_curve, auc
    cmc, ranks = m._compute_cmc(S, lbl)
    gen, imp = m._get_score_distributions(S, lbl)
    fpr, tpr, _ = roc_curve([1] * len(gen) + [0] * len(imp), list(gen) + list(imp))
    tar1 = float(np.interp(0.01, fpr, tpr))  # TAR @ FAR=1%
    return {'rank1': float(ranks[1]), 'rank5': float(ranks.get(5, 0)),
            'eer': float(m._compute_eer(fpr, tpr)), 'auc': float(auc(fpr, tpr)),
            'tar1': tar1}


def per_sample_alpha_features(quals, Scnn, Shyb):
    """Assemble quality feature vectors incl. branch confidence/disagreement."""
    X = []
    for i in range(len(quals)):
        row = Scnn[i].copy(); row[i] = -1e9
        rowh = Shyb[i].copy(); rowh[i] = -1e9
        quals[i].update(branch_confidence(row, rowh))
        X.append(to_vector(quals[i]))
    return np.array(X)


def branch_correct(S, lbl):
    """Per-sample: is closed-set rank-1 (self excluded) correct?"""
    Sm = S.copy(); np.fill_diagonal(Sm, -1e9)
    return lbl[Sm.argmax(1)] == lbl


def fit_gate_eval(Xv, Xt, disagree, y, Stc, Sth, tln, M, drop_idx=None):
    """Fit the logistic per-sample gate (optionally dropping feature columns) on VAL
    disagreement cases, apply to TEST, return symmetrised fusion metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    keep = [i for i in range(Xv.shape[1]) if drop_idx is None or i not in drop_idx]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
    clf.fit(Xv[disagree][:, keep], y)
    a = clf.predict_proba(Xt[:, keep])[:, 1][:, None]
    S = 0.5 * (a * Stc + (1 - a) * Sth + (a * Stc + (1 - a) * Sth).T)
    return metrics_from_sim(S, tln, M)


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    gd = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    loaders = create_hybrid_loaders(pre, gd, config)
    cnn, hyb = load_models(config, device)
    proton = load_proton(config, device)
    M = BiometricMetrics()

    results = {}
    for kind, sev in SHIFTS:
        tag = 'clean' if kind == 'clean' else f'{kind}_s{sev}'
        print(f"\n=== shift: {tag} ===")
        # VAL: fit fixed-alpha and learned gate here (no test leakage)
        vc, vh, vl, vq = embed_shift(cnn, hyb, loaders['val'], device, kind, sev)
        Svc, Svh = sim(vc), sim(vh); vln = vl.numpy()
        # TEST
        tc, th, tl, tq = embed_shift(cnn, hyb, loaders['test'], device, kind, sev)
        Stc, Sth = sim(tc), sim(th); tln = tl.numpy()

        r = {}
        r['cnn'] = metrics_from_sim(Stc, tln, M)
        r['hybrid'] = metrics_from_sim(Sth, tln, M)
        r['fixed_50'] = metrics_from_sim(0.5 * Stc + 0.5 * Sth, tln, M)

        # (4) val-tuned fixed alpha
        best_a, best_e = 0.5, 1.0
        for a in np.linspace(0, 1, 21):
            e = metrics_from_sim(a * Svc + (1 - a) * Svh, vln, M)['eer']
            if e < best_e:
                best_e, best_a = e, a
        r['val_tuned_alpha'] = {**metrics_from_sim(best_a * Stc + (1 - best_a) * Sth, tln, M), 'alpha': float(best_a)}

        # CNN+ProtoN val-tuned fusion (clean only; ProtoN graph features are corruption-invariant)
        if tag == 'clean':
            pvc, pvl = embed_proton(proton, loaders['val'], device)
            ptc, ptl = embed_proton(proton, loaders['test'], device)
            assert np.array_equal(pvl.numpy(), vln) and np.array_equal(ptl.numpy(), tln), 'loader order mismatch'
            Spvc, Sptc = sim(pvc), sim(ptc)
            ba, be = 0.5, 1.0
            for a in np.linspace(0, 1, 21):
                e = metrics_from_sim(a * Svc + (1 - a) * Spvc, vln, M)['eer']
                if e < be:
                    be, ba = e, a
            r['cnn_proton_val_tuned'] = {**metrics_from_sim(ba * Stc + (1 - ba) * Sptc, tln, M), 'alpha': float(ba)}
            r['proton'] = metrics_from_sim(Sptc, tln, M)

        # (5,6) quality-aware: learn per-sample alpha on VAL, apply to TEST
        Xv = per_sample_alpha_features(vq, Svc, Svh)
        Xt = per_sample_alpha_features(tq, Stc, Sth)
        cnn_ok = branch_correct(Svc, vln); hyb_ok = branch_correct(Svh, vln)
        disagree = cnn_ok != hyb_ok
        if disagree.sum() >= 10 and len(set(cnn_ok[disagree])) > 1:
            y = cnn_ok[disagree].astype(int)  # 1 = trust CNN
            r['quality_learned'] = fit_gate_eval(Xv, Xt, disagree, y, Stc, Sth, tln, M)
            # (5) rule-based: alpha from blur (low blur var -> trust hybrid more)
            blur = Xt[:, 0][:, None]; blur = (blur - blur.min()) / (blur.max() - blur.min() + 1e-8)
            Sqr = blur * Stc + (1 - blur) * Sth
            Sqr = 0.5 * (Sqr + Sqr.T)
            r['quality_rule'] = metrics_from_sim(Sqr, tln, M)
            # feature-group ablations of the learned gate (mentor's Part-1 ablation list)
            r['gate_ablation'] = {
                'full': r['quality_learned'],
                'no_image_quality': fit_gate_eval(Xv, Xt, disagree, y, Stc, Sth, tln, M, IMG_Q_IDX),
                'no_graph_quality': fit_gate_eval(Xv, Xt, disagree, y, Stc, Sth, tln, M, GRAPH_Q_IDX),
                'no_disagreement': fit_gate_eval(Xv, Xt, disagree, y, Stc, Sth, tln, M, DISAGREE_IDX),
            }
        else:
            r['quality_learned'] = r['quality_rule'] = {'note': 'too few disagreements'}

        for k, v in r.items():
            if 'rank1' in v:
                print(f"  {k:18s} R1={v['rank1']*100:5.1f}  R5={v['rank5']*100:5.1f}  EER={v['eer']*100:5.2f}  AUC={v['auc']:.4f}")
        results[tag] = r

    save_stats(results, str(PROJECT_ROOT / 'outputs/stats/quality_fusion_results.json'))
    print("\nSaved -> outputs/stats/quality_fusion_results.json")


if __name__ == '__main__':
    main()
