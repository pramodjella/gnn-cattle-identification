"""
External validity: does the coherence confound hold on a SECOND dataset?
=======================================================================
Our confound result (an importance-blind coherent removal matches the "most important"
condition; zero-information smoothed noise passes the standard top-vs-random test) is so
far established on one dataset. This runs the same test end-to-end on an INDEPENDENT
cattle muzzle set, building its keypoint graphs from scratch with the same DISK + k-NN
pipeline used for the primary data.

Note on interpretation: the GNN is trained on the primary dataset, so on an external set it
operates under domain shift. That is acceptable and arguably the point -- we test whether
the ABLATION PROTOCOL is confounded, not whether the model is accurate here. The model need
only produce a stable embedding for Delta-cos to be meaningful.

Outputs: outputs/stats/coherence_second_dataset.json
Usage:
  python scripts/experiment_coherence_second_dataset.py --images data/external/kaggle25_cropped
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import load_config, save_stats
from src.features.superpoint import SuperPointExtractor
from src.features.graph_builder import GraphBuilder
from src.evaluation.faithfulness import _subgraph
from src.models.explainability import GradCAMGraph
from scripts.evaluate_explainability import gradcam_importance, _last_gat_layer_name
from scripts.experiment_causal_ablation import embed, bootstrap_ci
from scripts.experiment_coherence_control import select_removal, coherence
from scripts.experiment_explainer_sweep import smoothed_random_importance

STRATEGIES = ['top', 'bottom', 'random', 'random_block']
EXPLAINERS = ['gradcam', 'random', 'smoothed_random']
IMG_EXT = ('.jpg', '.jpeg', '.png')


def build_external_graphs(img_dir, n, cfg, size=256, seed=0):
    """DISK keypoints + k-NN graphs for external images, matching the primary pipeline
    (including the same CLAHE preprocessing used at training time)."""
    import cv2
    kp = cfg['keypoints']
    gcfg = cfg['graph']
    ex = SuperPointExtractor(max_keypoints=kp['max_keypoints'],
                             detection_threshold=kp['detection_threshold'],
                             nms_radius=kp['nms_radius'], backend='disk')
    gb = GraphBuilder(knn_k=gcfg['knn_k'],
                      normalize_positions=gcfg['normalize_positions'],
                      use_relative_positions=gcfg['use_relative_positions'])

    paths = sorted([p for p in Path(img_dir).rglob('*') if p.suffix.lower() in IMG_EXT])
    rng = np.random.default_rng(seed)
    if len(paths) > n:
        paths = [paths[i] for i in sorted(rng.choice(len(paths), n, replace=False))]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    graphs = []
    for p in paths:
        img = np.array(Image.open(p).convert('RGB').resize((size, size)))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        r = ex.extract(img)
        if r['keypoints'].shape[0] < 12:
            continue
        graphs.append(gb.build_graph(keypoints=r['keypoints'],
                                     descriptors=r['descriptors'],
                                     scores=r['scores'], image_size=size))
    print(f"[INFO] built {len(graphs)} graphs from {len(paths)} images", flush=True)
    return graphs


def load_model(cfg, name, dev):
    from src.models.gnn_v3 import CattleGNNv3
    lm = PROJECT_ROOT / cfg['dataset']['graph_dir'] / 'label_mapping.json'
    ncls = len(json.load(open(lm))) if lm.exists() else 260
    model = CattleGNNv3(config=cfg)
    if hasattr(model, 'set_num_classes'):
        model.set_num_classes(ncls)
    st = torch.load(PROJECT_ROOT / f'outputs/{name}/best_model.pt',
                    map_location=dev, weights_only=False)
    model.load_state_dict(st['model_state_dict'])
    return model.to(dev).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='data/external/kaggle25_cropped')
    ap.add_argument('--num-images', type=int, default=300)
    ap.add_argument('--model', default='gnn_v3')
    ap.add_argument('--frac', type=float, default=0.30)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graphs = build_external_graphs(PROJECT_ROOT / args.images, args.num_images, cfg)
    if len(graphs) < 30:
        print('[ABORT] too few graphs built')
        return

    model = load_model(cfg, args.model, dev)
    gc = GradCAMGraph(model, target_layer_name=_last_gat_layer_name(model))
    full = torch.stack([embed(model, g, dev) for g in graphs])

    out = {'dataset': str(args.images), 'n_graphs': len(graphs), 'model': args.model,
           'frac': args.frac, 'note': 'model trained on primary dataset; external set is a domain shift',
           'explainers': {}}

    for kind in EXPLAINERS:
        if kind == 'gradcam':
            imps = [gradcam_importance(gc, g, dev).detach().cpu() for g in graphs]
        elif kind == 'random':
            imps = [torch.tensor(rng.random(g.x.size(0)), dtype=torch.float) for g in graphs]
        else:
            imps = [smoothed_random_importance(g, rng) for g in graphs]

        res = {}
        for strat in STRATEGIES:
            embs, cohs = [], []
            for g, imp in zip(graphs, imps):
                rem, _ = select_removal(g, imp, args.frac, strat, rng)
                cohs.append(coherence(g, rem))
                drop = set(int(x) for x in rem)
                keep = torch.tensor([i for i in range(g.x.size(0)) if i not in drop],
                                    dtype=torch.long)
                embs.append(embed(model, _subgraph(g, keep), dev))
            ab = torch.stack(embs)
            dcos = (1 - (ab * full).sum(1)).clamp(min=0).numpy()
            lo, hi = bootstrap_ci(dcos)
            res[strat] = {'dcosine': float(dcos.mean()), 'dcosine_ci': [lo, hi],
                          'coherence': float(np.nanmean(cohs))}

        t, r, b = res['top'], res['random'], res['bottom']
        res['passes_top_vs_random'] = bool(t['dcosine_ci'][0] > r['dcosine_ci'][1])
        res['passes_top_vs_bottom'] = bool(t['dcosine_ci'][0] > b['dcosine_ci'][1])
        out['explainers'][kind] = res
        print(f"{kind:16s} top={t['dcosine']:.4f} random={r['dcosine']:.4f} "
              f"bottom={b['dcosine']:.4f} block={res['random_block']['dcosine']:.4f} | "
              f"top>random: {'PASS' if res['passes_top_vs_random'] else 'fail'} | "
              f"top>bottom: {'PASS' if res['passes_top_vs_bottom'] else 'fail'}", flush=True)

    gc.remove_hooks()
    save_stats(out, str(PROJECT_ROOT / 'outputs/stats/coherence_second_dataset.json'))
    print('\nSaved -> outputs/stats/coherence_second_dataset.json')


if __name__ == '__main__':
    main()
