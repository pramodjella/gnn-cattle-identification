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
def embed(model, loader, device, mode):
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    E, L = [], []
    for images, graphs, labels in loader:
        img = images.to(device)
        g = intervene(graphs, mode).to(device)
        with autocast(device_type='cuda', dtype=amp, enabled=(device.type == 'cuda')):
            if mode == 'zero_node_feats':
                out = _forward_zero_nodes(model, img, g)
            else:
                out = model(img, g)
        E.append(F.normalize(out['embedding'], p=2, dim=-1).float().cpu()); L.append(labels)
    return torch.cat(E), torch.cat(L)


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


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = create_hybrid_loaders(str(PROJECT_ROOT / config['dataset']['processed_dir']),
                                    str(PROJECT_ROOT / config['dataset']['graph_dir']), config)
    model = load_hybrid(config, device)
    M = BiometricMetrics()

    modes = ['full', 'zero_edge_attr', 'shuffle_pos', 'randomize_edges', 'zero_node_feats']
    out = {}
    base = None
    for mode in modes:
        emb, lbl = embed(model, loaders['test'], device, mode)
        r = M.compute_all_metrics(emb, lbl)['summary']
        m = {'rank1': r['rank_1_accuracy'], 'eer': r['eer'], 'auc': r['roc_auc']}
        out[mode] = m
        if mode == 'full':
            base = m
        drop = (base['rank1'] - m['rank1']) * 100 if base else 0
        print(f"  {mode:16s} R1={m['rank1']*100:5.1f}%  EER={m['eer']*100:5.2f}%  AUC={m['auc']:.4f}"
              f"  (dRank1 {drop:+.1f})")

    out['derived'] = {
        'topology_sensitivity_rank1_drop': (base['rank1'] - out['randomize_edges']['rank1']) * 100,
        'spatial_feature_sensitivity_rank1_drop': (base['rank1'] - out['shuffle_pos']['rank1']) * 100,
        'edge_attr_reliance_rank1_drop': (base['rank1'] - out['zero_edge_attr']['rank1']) * 100,
        'node_feature_reliance_rank1_drop': (base['rank1'] - out['zero_node_feats']['rank1']) * 100,
    }
    save_stats(out, str(PROJECT_ROOT / 'outputs/stats/pathway_intervention.json'))
    print("\nSaved -> outputs/stats/pathway_intervention.json")


if __name__ == '__main__':
    main()
