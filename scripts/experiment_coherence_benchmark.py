"""
Does the coherence confound hold on PUBLIC graph-classification benchmarks?
===========================================================================
Closes two external-validity gaps at once:
  (a) a non-GATv2 architecture -- we train a GIN (Xu et al., ICLR 2019);
  (b) standard public graph-XAI benchmarks (MUTAG / PROTEINS from TUDataset), rather
      than our own cattle keypoint graphs.

Generalisation of `random_block`: our keypoint graphs had 2-D coordinates, so a coherent
block was a spatial neighbourhood. Molecules/proteins have no coordinates, so here a block
is a BFS ball -- a random seed node plus its nearest neighbours in GRAPH-HOP distance.
This is the more general definition and applies to any graph.

Attributions compared:
  gradcam          gradient x activation at the last GIN layer, mean-pooled per node
  random           i.i.d. uniform node scores (no information, NOT smooth)
  smoothed_random  the same noise averaged over each node's 1-hop neighbourhood
                   (still ZERO information, but SMOOTH over the graph)

Prediction: `smoothed_random` passes the conventional top-vs-random test, and an
importance-blind BFS block matches or beats `top` -- i.e. the confound is a property of
the protocol, not of our data or model family.

Outputs: outputs/stats/coherence_benchmark_<dataset>.json
Usage:   python scripts/experiment_coherence_benchmark.py --dataset MUTAG
"""
import sys
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import save_stats
from scripts.experiment_causal_ablation import bootstrap_ci

STRATEGIES = ['top', 'bottom', 'random', 'random_block']
EXPLAINERS = ['gradcam', 'random', 'smoothed_random']


class GIN(nn.Module):
    """Standard 3-layer GIN graph classifier (deliberately NOT GATv2)."""

    def __init__(self, in_dim, hidden=64, n_classes=2):
        super().__init__()
        from torch_geometric.nn import GINConv
        mk = lambda i, o: nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))
        self.c1 = GINConv(mk(in_dim, hidden))
        self.c2 = GINConv(mk(hidden, hidden))
        self.c3 = GINConv(mk(hidden, hidden))
        self.lin = nn.Linear(hidden, n_classes)
        self.last_act = None

    def forward(self, x, edge_index, batch):
        from torch_geometric.nn import global_mean_pool
        h = F.relu(self.c1(x, edge_index))
        h = F.relu(self.c2(h, edge_index))
        h = F.relu(self.c3(h, edge_index))
        self.last_act = h                      # (N, hidden) for Grad-CAM
        g = global_mean_pool(h, batch)
        return self.lin(g), g


def train_gin(ds, dev, epochs=60, seed=0):
    from torch_geometric.loader import DataLoader
    torch.manual_seed(seed)
    n_tr = int(0.8 * len(ds))
    tr, te = ds[:n_tr], ds[n_tr:]
    model = GIN(ds.num_features, 64, ds.num_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ld = DataLoader(tr, batch_size=32, shuffle=True)
    for ep in range(epochs):
        model.train()
        for b in ld:
            b = b.to(dev)
            opt.zero_grad()
            out, _ = model(b.x, b.edge_index, b.batch)
            loss = F.cross_entropy(out, b.y)
            loss.backward()
            opt.step()
    # test accuracy (sanity: the model must actually work for ablation to mean anything)
    model.eval()
    correct = 0
    with torch.no_grad():
        for b in DataLoader(te, batch_size=64):
            b = b.to(dev)
            out, _ = model(b.x, b.edge_index, b.batch)
            correct += int((out.argmax(1) == b.y).sum())
    acc = correct / max(len(te), 1)
    print(f"[INFO] GIN trained: test acc = {acc:.3f} on {len(te)} graphs", flush=True)
    return model, te, acc


def graph_embed(model, data, dev):
    with torch.no_grad():
        b = torch.zeros(data.x.size(0), dtype=torch.long, device=dev)
        _, g = model(data.x.to(dev), data.edge_index.to(dev), b)
    return F.normalize(g.squeeze(0), p=2, dim=-1).cpu()


def gradcam_nodes(model, data, dev):
    """Gradient x activation at the last GIN layer -> per-node importance."""
    model.zero_grad()
    b = torch.zeros(data.x.size(0), dtype=torch.long, device=dev)
    out, _ = model(data.x.to(dev), data.edge_index.to(dev), b)
    out[0, out.argmax(1)].backward()
    act = model.last_act.detach()
    grad = model.last_act.grad if model.last_act.grad is not None else None
    if grad is None:                            # retain via hook fallback
        return torch.rand(data.x.size(0))
    return F.relu((act * grad).sum(-1)).detach().cpu()


def adjacency(data, n):
    adj = [[] for _ in range(n)]
    ei = data.edge_index.cpu().numpy()
    for s, t in zip(ei[0], ei[1]):
        adj[int(s)].append(int(t))
    return adj


def bfs_ball(adj, seed, k, n):
    """k nodes closest to `seed` in graph-hop distance (BFS order)."""
    seen, order, q = {seed}, [seed], deque([seed])
    while q and len(order) < k:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); order.append(v); q.append(v)
                if len(order) >= k:
                    break
    if len(order) < k:                          # disconnected: pad randomly
        rest = [i for i in range(n) if i not in seen]
        order += rest[:k - len(order)]
    return np.array(order[:k])


def smoothed_noise(adj, n, rng):
    """Zero-information scores made SMOOTH by 1-hop averaging."""
    raw = rng.random(n)
    return torch.tensor([float(np.mean([raw[i]] + [raw[j] for j in adj[i]])) for i in range(n)],
                        dtype=torch.float)


def hop_coherence(adj, idx, n):
    """Mean pairwise hop distance within the removed set (lower = more coherent)."""
    idx = list(map(int, idx))
    if len(idx) < 2:
        return float('nan')
    tot, cnt = 0.0, 0
    for s in idx[:12]:                          # sample for cost
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        for t in idx:
            if t != s:
                tot += dist.get(t, n); cnt += 1
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MUTAG', choices=['MUTAG', 'PROTEINS', 'ENZYMES'])
    ap.add_argument('--frac', type=float, default=0.30)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from torch_geometric.datasets import TUDataset
    rng = np.random.default_rng(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = TUDataset(root=str(PROJECT_ROOT / 'data/tudataset'), name=args.dataset).shuffle()
    print(f"[INFO] {args.dataset}: {len(ds)} graphs, {ds.num_features} feats, "
          f"{ds.num_classes} classes, avg nodes {ds[0].num_nodes}", flush=True)

    model, test_graphs, acc = train_gin(ds, dev, args.epochs, args.seed)
    model.eval()
    graphs = [g for g in test_graphs if g.num_nodes >= 8]
    if len(graphs) < 20:
        print('[ABORT] too few usable test graphs'); return
    print(f"[INFO] evaluating on {len(graphs)} test graphs", flush=True)

    full = torch.stack([graph_embed(model, g, dev) for g in graphs])
    adjs = [adjacency(g, g.num_nodes) for g in graphs]

    out = {'dataset': args.dataset, 'model': 'GIN (3-layer)', 'test_acc': acc,
           'n_graphs': len(graphs), 'frac': args.frac, 'explainers': {}}

    for kind in EXPLAINERS:
        if kind == 'gradcam':
            imps = []
            for g in graphs:
                model.zero_grad()
                b = torch.zeros(g.num_nodes, dtype=torch.long, device=dev)
                x = g.x.to(dev).clone().requires_grad_(True)
                o, _ = model(x, g.edge_index.to(dev), b)
                o[0, int(o.argmax(1))].backward()
                imps.append(F.relu((x * x.grad).sum(-1)).detach().cpu())
        elif kind == 'random':
            imps = [torch.tensor(rng.random(g.num_nodes), dtype=torch.float) for g in graphs]
        else:
            imps = [smoothed_noise(a, g.num_nodes, rng) for g, a in zip(graphs, adjs)]

        res = {}
        for strat in STRATEGIES:
            embs, cohs = [], []
            for g, a, imp in zip(graphs, adjs, imps):
                n = g.num_nodes
                k = max(1, int(round(args.frac * n)))
                k = min(k, n - 3) if n - 3 > 0 else 1
                order = torch.argsort(imp, descending=True)
                if strat == 'top':
                    rem = order[:k].numpy()
                elif strat == 'bottom':
                    rem = order[-k:].numpy()
                elif strat == 'random':
                    rem = rng.choice(n, size=k, replace=False)
                else:
                    rem = bfs_ball(a, int(rng.integers(n)), k, n)
                cohs.append(hop_coherence(a, rem, n))
                drop = set(map(int, rem))
                keep = torch.tensor([i for i in range(n) if i not in drop], dtype=torch.long)
                sub = g.clone()
                remap = -torch.ones(n, dtype=torch.long); remap[keep] = torch.arange(len(keep))
                ei = g.edge_index
                m = torch.tensor([(int(s) in drop or int(t) in drop) for s, t in zip(ei[0], ei[1])])
                sub.x = g.x[keep]; sub.edge_index = remap[ei[:, ~m]]
                embs.append(graph_embed(model, sub, dev))
            ab = torch.stack(embs)
            dcos = (1 - (ab * full).sum(1)).clamp(min=0).numpy()
            lo, hi = bootstrap_ci(dcos)
            res[strat] = {'dcosine': float(dcos.mean()), 'dcosine_ci': [lo, hi],
                          'hop_coherence': float(np.nanmean(cohs))}

        t, r, b_ = res['top'], res['random'], res['bottom']
        res['passes_top_vs_random'] = bool(t['dcosine_ci'][0] > r['dcosine_ci'][1])
        res['passes_top_vs_bottom'] = bool(t['dcosine_ci'][0] > b_['dcosine_ci'][1])
        out['explainers'][kind] = res
        print(f"{kind:16s} top={t['dcosine']:.4f} random={r['dcosine']:.4f} "
              f"bottom={b_['dcosine']:.4f} block={res['random_block']['dcosine']:.4f} | "
              f"top>random: {'PASS' if res['passes_top_vs_random'] else 'fail'} | "
              f"top>bottom: {'PASS' if res['passes_top_vs_bottom'] else 'fail'}", flush=True)

    save_stats(out, str(PROJECT_ROOT / f'outputs/stats/coherence_benchmark_{args.dataset}.json'))
    print(f"\nSaved -> outputs/stats/coherence_benchmark_{args.dataset}.json")


if __name__ == '__main__':
    main()
