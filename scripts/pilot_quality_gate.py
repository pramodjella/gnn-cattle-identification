"""
Pilot: does input quality predict WHICH branch is right?  (go/no-go de-risk)
==========================================================================
The whole quality-conditioned fusion contribution collapses if quality features
do NOT predict per-sample branch superiority. This pilot decides that cheaply,
BEFORE building the corruption pipeline or the full gate.

Protocol (val only, no test touched):
  * Gallery = per-identity mean embedding from TRAIN (CNN and GNN separately).
  * Probes  = VAL samples (aligned image+graph via the hybrid loader).
  * Per probe: is CNN rank-1 correct? is GNN rank-1 correct?
  * Keep DISAGREEMENT cases (exactly one branch correct) — that is where a
    router can help. Label = 1 if CNN is the correct branch, 0 if GNN is.
  * Train logistic regression (quality features -> which branch) with 5-fold CV.
  * Report ROC-AUC. AUC ~0.5 => quality can't route => STOP. AUC >> 0.5 => go.

Usage:  python scripts/pilot_quality_gate.py --gnn proton
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.training.image_dataset import create_hybrid_loaders
from src.evaluation.quality import image_quality, graph_quality, branch_confidence, to_vector, FEATURE_ORDER


def load_cnn(config, device):
    from src.models.cnn_model import CNNMuzzleModel
    ck = torch.load(PROJECT_ROOT / 'outputs/cnn/best_model.pt', map_location=device, weights_only=False)
    mc = ck.get('config', {})
    m = CNNMuzzleModel(num_classes=ck.get('num_classes', 260),
                       embedding_dim=mc.get('embedding_dim', 512),
                       backbone=mc.get('backbone', 'efficientnet_b4'),
                       arcface_scale=mc.get('arcface_scale', 128.0),
                       arcface_margin=mc.get('arcface_margin', 0.35)).to(device)
    m.load_state_dict(ck['model_state_dict'])
    return m.eval()


def load_gnn(name, config, device, num_classes):
    if name == 'proton':
        from src.models.proton import CattleProtoN
        ck = torch.load(PROJECT_ROOT / 'outputs/proton/best_model.pt', map_location=device, weights_only=False)
        pc = config.get('proton', {})
        m = CattleProtoN(num_classes=num_classes,
                         hidden_dim=pc.get('hidden_dim', 128),
                         num_heads=pc.get('num_heads', 4),
                         num_layers=pc.get('num_layers', 4),
                         fusion_dim=pc.get('fusion_dim', 512),
                         embedding_dim=pc.get('embedding_dim', 256),
                         dropout=pc.get('dropout', 0.12)).to(device)
    elif name == 'gnn_v3':
        from src.models.gnn_v3 import CattleGNNv3
        ck = torch.load(PROJECT_ROOT / 'outputs/gnn_v3/best_model.pt', map_location=device, weights_only=False)
        m = CattleGNNv3(config=config)
        if hasattr(m, 'set_num_classes'):
            m.set_num_classes(num_classes)
        m = m.to(device)
    else:
        raise ValueError(name)
    m.load_state_dict(ck['model_state_dict'])
    return m.eval()


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _corrupt_batch(images_cpu, kind, severity):
    """De-normalise -> corrupt -> re-normalise a batch (CPU)."""
    from src.evaluation.corruptions import apply
    out = torch.empty_like(images_cpu)
    for i in range(images_cpu.size(0)):
        x01 = (images_cpu[i] * _STD + _MEAN).clamp(0, 1)
        xc = apply(x01, kind, severity, seed=i)
        out[i] = (xc - _MEAN) / _STD
    return out


@torch.no_grad()
def embed_split(cnn, gnn, loader, device, corrupt='clean', severity=0):
    """Return aligned CNN & GNN embeddings, labels, and per-sample quality dicts.

    If corrupt != 'clean', images are corrupted (CNN sees corruption; the GNN
    uses the pre-built clean graph — an optimistic upper bound for the GNN, used
    to test whether CNN degradation alone opens routable disagreement).
    """
    cnn_e, gnn_e, lbls, quals = [], [], [], []
    for images, graphs, labels in loader:
        if corrupt != 'clean' and severity > 0:
            images = _corrupt_batch(images, corrupt, severity)
        images = images.to(device); graphs = graphs.to(device)
        ce = F.normalize(cnn.get_embedding(images), p=2, dim=-1).float().cpu()
        g_out = gnn(graphs)
        ge = F.normalize(g_out['embedding'], p=2, dim=-1).float().cpu()
        cnn_e.append(ce); gnn_e.append(ge); lbls.append(labels.cpu())
        # per-sample quality (image + graph) — confidence added later with sims
        from torch_geometric.utils import unbatch, unbatch_edge_index
        # split the batched graph back to per-sample for graph_quality
        graphs_cpu = graphs.cpu()
        node_batch = graphs_cpu.batch
        for i in range(images.size(0)):
            q = image_quality(images[i].cpu())
            # build a lightweight per-graph view
            mask = node_batch == i
            sub = _subgraph_view(graphs_cpu, mask, i)
            q.update(graph_quality(sub))
            quals.append(q)
    return (torch.cat(cnn_e), torch.cat(gnn_e), torch.cat(lbls), quals)


class _GV:  # minimal graph view
    pass


def _subgraph_view(batch, node_mask, gi):
    v = _GV()
    idx = node_mask.nonzero(as_tuple=True)[0]
    v.x = batch.x[idx]
    remap = -torch.ones(batch.x.size(0), dtype=torch.long)
    remap[idx] = torch.arange(idx.size(0))
    ei = batch.edge_index
    em = node_mask[ei[0]] & node_mask[ei[1]]
    v.edge_index = torch.stack([remap[ei[0][em]], remap[ei[1][em]]])
    v.pos = batch.pos[idx] if getattr(batch, 'pos', None) is not None else None
    return v


def gallery_prototypes(emb, lbl):
    ids = sorted(set(int(x) for x in lbl))
    protos, pid = [], []
    for c in ids:
        m = lbl == c
        protos.append(F.normalize(emb[m].mean(0), p=2, dim=-1)); pid.append(c)
    return torch.stack(protos), np.array(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gnn', default='proton', choices=['proton', 'gnn_v3'])
    ap.add_argument('--corrupt', default='clean', choices=['clean', 'blur', 'brightness', 'spatter'])
    ap.add_argument('--severity', type=int, default=0)
    args = ap.parse_args()

    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])
    pre = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    import json
    with open(os.path.join(graph_dir, 'label_mapping.json')) as f:
        num_classes = len(json.load(f))

    cnn = load_cnn(config, device)
    gnn = load_gnn(args.gnn, config, device, num_classes)
    loaders = create_hybrid_loaders(pre, graph_dir, config)

    print("[pilot] embedding train (gallery) ...")
    cnn_tr, gnn_tr, lbl_tr, _ = embed_split(cnn, gnn, loaders['train'], device)
    print(f"[pilot] embedding val (probes) | corrupt={args.corrupt} sev={args.severity} ...")
    cnn_va, gnn_va, lbl_va, quals = embed_split(cnn, gnn, loaders['val'], device,
                                                corrupt=args.corrupt, severity=args.severity)

    cnn_proto, cpid = gallery_prototypes(cnn_tr, lbl_tr)
    gnn_proto, gpid = gallery_prototypes(gnn_tr, lbl_tr)

    cnn_sims = (cnn_va @ cnn_proto.t()).numpy()   # (Nval, C)
    gnn_sims = (gnn_va @ gnn_proto.t()).numpy()
    lbl_va = lbl_va.numpy()

    cnn_correct = cpid[cnn_sims.argmax(1)] == lbl_va
    gnn_correct = gpid[gnn_sims.argmax(1)] == lbl_va

    # add confidence features per sample
    X, y = [], []
    for i in range(len(lbl_va)):
        quals[i].update(branch_confidence(cnn_sims[i], gnn_sims[i]))
    disagree = cnn_correct != gnn_correct
    print(f"\n  val probes: {len(lbl_va)}")
    print(f"  CNN rank-1: {cnn_correct.mean()*100:.1f}%  GNN rank-1: {gnn_correct.mean()*100:.1f}%")
    print(f"  both correct: {(cnn_correct & gnn_correct).mean()*100:.1f}%  "
          f"both wrong: {(~cnn_correct & ~gnn_correct).mean()*100:.1f}%")
    print(f"  DISAGREEMENT (router matters): {disagree.mean()*100:.1f}%  ({disagree.sum()} cases)")

    for i in np.where(disagree)[0]:
        X.append(to_vector(quals[i])); y.append(1 if cnn_correct[i] else 0)
    X, y = np.array(X), np.array(y)

    if len(y) < 20 or len(set(y)) < 2:
        print("\n  [INCONCLUSIVE] too few / single-class disagreement cases on clean val.")
        print("  -> This is expected on clean data; the real test is on CORRUPTED val.")
        print("  -> Proceed to build corruption pipeline, then re-run this pilot.")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))
    proba = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]
    auc = roc_auc_score(y, proba)

    print("\n" + "=" * 62)
    print(f"  ROUTING AUC (quality -> which branch is right): {auc:.3f}")
    print(f"  base rate CNN-is-right among disagreements: {y.mean():.3f}")
    if auc >= 0.60:
        print("  VERDICT: GO — quality predicts branch superiority (AUC >= 0.60).")
    elif auc >= 0.55:
        print("  VERDICT: WEAK — marginal signal; corruption may strengthen it.")
    else:
        print("  VERDICT: NO-GO on clean data — quality does not route. Test corrupted.")
    print("=" * 62)

    # feature importance (which quality signals route)
    clf.fit(X, y)
    coefs = clf.named_steps['logisticregression'].coef_[0]
    order = np.argsort(-np.abs(coefs))
    print("  top routing features:")
    for j in order[:6]:
        print(f"    {FEATURE_ORDER[j]:16s} coef={coefs[j]:+.3f}")


if __name__ == '__main__':
    main()
