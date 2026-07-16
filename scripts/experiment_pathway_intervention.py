"""
Part 2, Stage 3 of the Research Extension Plan: Hybrid Pathway Intervention.
==========================================================================
Measures what the Hybrid model causally relies on by perturbing its graph
pathway at test time and observing the change in identification/verification:
  * full            -- unperturbed Hybrid
  * randomize_edges -- replace graph edges with random ones (topology destroyed)
  * shuffle_pos     -- permute keypoint positions within each graph (feature
                       sampling locations scrambled)
  * zero_edge_attr  -- zero the geometric edge attributes
  * zero_node_feats -- zero the sampled node features before the GNN head
Derived measures: topology sensitivity (edge randomization), spatial-feature
sensitivity (position shuffle), edge-attribute reliance.

Outputs: outputs/stats/pathway_intervention.json
Usage: python scripts/experiment_pathway_intervention.py
"""
import os, sys, json, copy
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

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def corrupt_batch(images_cpu, kind, sev):
    if kind is None or sev == 0:
        return images_cpu
    out = torch.empty_like(images_cpu)
    for i in range(images_cpu.size(0)):
        x01 = (images_cpu[i] * STD + MEAN).clamp(0, 1)
        out[i] = (corr.apply(x01, kind, sev, seed=i) - MEAN) / STD
    return out


def load_hybrid(config, device):
    from src.models.hybrid_model import HybridCNNGNN
    hk = torch.load(PROJECT_ROOT / 'outputs/hybrid/best_model.pt', map_location=device, weights_only=False)
    m = HybridCNNGNN(num_classes=hk.get('num_classes', 260), config=config, pretrained=False).to(device)
    m.load_state_dict(hk['model_state_dict']); m.eval()
    return m


def intervene(graphs, mode):
    """Return a perturbed copy of the batched graph for the given intervention."""
    g = graphs.clone()
    if mode == 'randomize_edges' and g.edge_index.numel() > 0:
        # random edges within each graph, vectorized (PyG batches nodes contiguously)
        b = g.batch
        counts = torch.bincount(b)                              # nodes per graph
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])  # start index per graph
        eg = b[g.edge_index[0]]                                 # graph id per edge
        sizes = counts[eg].float()
        E = g.edge_index.size(1)
        src = ptr[eg] + (torch.rand(E) * sizes).long()
        dst = ptr[eg] + (torch.rand(E) * sizes).long()
        g.edge_index = torch.stack([src, dst])
    elif mode == 'shuffle_pos' and getattr(g, 'pos', None) is not None:
        b = g.batch
        for gi in b.unique():
            idx = (b == gi).nonzero(as_tuple=True)[0]
            g.pos[idx] = g.pos[idx][torch.randperm(len(idx))]
    elif mode == 'zero_edge_attr' and getattr(g, 'edge_attr', None) is not None:
        g.edge_attr = torch.zeros_like(g.edge_attr)
    return g


@torch.no_grad()
def embed(model, loader, device, mode, corrupt=None):
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    E, L = [], []
    for images, graphs, labels in loader:
        images = corrupt_batch(images, *corrupt) if corrupt else images
        img = images.to(device)
        g = intervene(graphs, mode).to(device)
        with autocast(device_type='cuda', dtype=amp, enabled=(device.type == 'cuda')):
            if mode == 'zero_node_feats':
                out = _forward_zero_nodes(model, img, g)
            else:
                out = model(img, g)
        E.append(F.normalize(out['embedding'], p=2, dim=-1).float().cpu()); L.append(labels)
    return torch.cat(E), torch.cat(L)


@torch.no_grad()
def embed_cnn(cnn, loader, device, corrupt=None):
    E, L = [], []
    for images, graphs, labels in loader:
        images = corrupt_batch(images, *corrupt) if corrupt else images
        e = cnn.get_embedding(images.to(device))
        E.append(F.normalize(e, p=2, dim=-1).float().cpu()); L.append(labels)
    return torch.cat(E), torch.cat(L)


def load_cnn(device):
    from src.models.cnn_model import CNNMuzzleModel
    ck = torch.load(PROJECT_ROOT / 'outputs/cnn/best_model.pt', map_location=device, weights_only=False)
    c = ck.get('config', {})
    m = CNNMuzzleModel(num_classes=ck.get('num_classes', 260), embedding_dim=c.get('embedding_dim', 512),
                       backbone=c.get('backbone', 'efficientnet_b4'), arcface_scale=c.get('arcface_scale', 128.0),
                       arcface_margin=c.get('arcface_margin', 0.35)).to(device)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


def branch_correct(S, lbl):
    Sm = S.copy(); np.fill_diagonal(Sm, -1e9)
    return lbl[Sm.argmax(1)] == lbl


def _forward_zero_nodes(model, images, graph):
    """Hybrid forward with sampled node features zeroed (isolates topology)."""
    edge_index = graph.edge_index; batch_vec = graph.batch
    pos = graph.pos[:, :2] if getattr(graph, 'pos', None) is not None else torch.rand(graph.num_nodes, 2, device=images.device)
    feats = model._sample_cnn_features_at_keypoints(images, pos, batch_vec)
    x = model.node_proj(torch.zeros_like(feats))
    if getattr(model, 'learned_edges', False):
        edge_index, _, _ = model.adaptive_graph(x, edge_index, getattr(graph, 'edge_attr', None))
    x, _ = model.edge_conv(x, batch=batch_vec)
    x, _ = model.trm(x, edge_index, batch=batch_vec)
    from torch_geometric.nn import global_mean_pool, global_max_pool
    xp = torch.cat([global_mean_pool(x, batch_vec), global_max_pool(x, batch_vec)], dim=-1)
    return {'embedding': F.normalize(model.projection_head(xp), p=2, dim=-1)}


MODES = ['full', 'zero_edge_attr', 'shuffle_pos', 'randomize_edges', 'zero_node_feats']


def run_interventions(model, loader, device, M, corrupt=None):
    res, base = {}, None
    for mode in MODES:
        emb, lbl = embed(model, loader, device, mode, corrupt)
        r = M.compute_all_metrics(emb, lbl)['summary']
        m = {'rank1': r['rank_1_accuracy'], 'eer': r['eer'], 'auc': r['roc_auc']}
        res[mode] = m
        if mode == 'full':
            base = m
        print(f"  {mode:16s} R1={m['rank1']*100:5.1f}%  EER={m['eer']*100:5.2f}%  "
              f"(dRank1 {(base['rank1']-m['rank1'])*100:+.1f})")
    res['derived'] = {
        'topology_sensitivity_rank1_drop': (base['rank1'] - res['randomize_edges']['rank1']) * 100,
        'spatial_feature_sensitivity_rank1_drop': (base['rank1'] - res['shuffle_pos']['rank1']) * 100,
        'edge_attr_reliance_rank1_drop': (base['rank1'] - res['zero_edge_attr']['rank1']) * 100,
        'node_feature_reliance_rank1_drop': (base['rank1'] - res['zero_node_feats']['rank1']) * 100}
    return res


def fusion_case_analysis(cnn, hyb, loader, device, M, alpha=0.95):
    """Rescued/harmed by fusion + performance on branch-disagreement subsets.
    alpha = validation-selected CNN weight from Part 1 (0.95)."""
    ce, cl = embed_cnn(cnn, loader, device)
    he, hl = embed(hyb, loader, device, 'full')
    lbl = cl.numpy()
    Sc = (ce @ ce.t()).numpy(); Sh = (he @ he.t()).numpy()
    Sf = alpha * Sc + (1 - alpha) * Sh
    cnn_ok = branch_correct(Sc, lbl); hyb_ok = branch_correct(Sh, lbl); fus_ok = branch_correct(Sf, lbl)
    rescued = int(np.sum(~cnn_ok & fus_ok))      # CNN wrong, fusion right
    harmed = int(np.sum(cnn_ok & ~fus_ok))       # CNN right, fusion wrong
    cnn_only = cnn_ok & ~hyb_ok                    # CNN-correct / Hybrid-wrong
    hyb_only = hyb_ok & ~cnn_ok                    # Hybrid-correct / CNN-wrong
    return {
        'alpha': alpha, 'n': int(len(lbl)),
        'cnn_rank1': float(cnn_ok.mean()), 'hybrid_rank1': float(hyb_ok.mean()),
        'fusion_rank1': float(fus_ok.mean()),
        'rescued_by_fusion': rescued, 'harmed_by_fusion': harmed, 'net_gain': rescued - harmed,
        'n_cnn_correct_hybrid_wrong': int(cnn_only.sum()),
        'fusion_keeps_cnn_only': float(fus_ok[cnn_only].mean()) if cnn_only.sum() else None,
        'n_hybrid_correct_cnn_wrong': int(hyb_only.sum()),
        'fusion_recovers_hybrid_only': float(fus_ok[hyb_only].mean()) if hyb_only.sum() else None}


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = create_hybrid_loaders(str(PROJECT_ROOT / config['dataset']['processed_dir']),
                                    str(PROJECT_ROOT / config['dataset']['graph_dir']), config)
    model = load_hybrid(config, device)
    M = BiometricMetrics()
    out = {}

    print("=== pathway intervention: CLEAN ===")
    out['clean'] = run_interventions(model, loaders['test'], device, M, corrupt=None)
    print("=== pathway intervention: SPATTER s3 (corrupted subset) ===")
    out['spatter_s3'] = run_interventions(model, loaders['test'], device, M, corrupt=('spatter', 3))

    print("=== fusion case analysis (rescued/harmed, disagreement splits) ===")
    cnn = load_cnn(device)
    out['fusion_cases'] = fusion_case_analysis(cnn, model, loaders['test'], device, M)
    fc = out['fusion_cases']
    print(f"  CNN R1={fc['cnn_rank1']*100:.1f}  Hybrid R1={fc['hybrid_rank1']*100:.1f}  "
          f"Fusion R1={fc['fusion_rank1']*100:.1f}")
    print(f"  rescued={fc['rescued_by_fusion']}  harmed={fc['harmed_by_fusion']}  net={fc['net_gain']}")
    print(f"  Hybrid-correct/CNN-wrong n={fc['n_hybrid_correct_cnn_wrong']} "
          f"fusion recovers {fc['fusion_recovers_hybrid_only']}")

    save_stats(out, str(PROJECT_ROOT / 'outputs/stats/pathway_intervention.json'))
    print("\nSaved -> outputs/stats/pathway_intervention.json")


if __name__ == '__main__':
    main()
