"""
Honest-corruption control: recompute DISK keypoints FROM the corrupted image.
============================================================================
The main corruption study (scripts/experiment_quality_fusion.py) corrupts the
image but reuses keypoint coordinates extracted from the CLEAN image, which is
optimistic under occlusion (the graph "knows" where undamaged structure was).
This control repeats the two most severe conditions (blur-5, spatter-5) with the
full pipeline rerun on the corrupted image: DISK keypoints are re-detected and
the k-NN graph is rebuilt, so no clean-image information leaks into the graph.

Reports Hybrid Rank-1/EER under clean-keypoint vs recomputed-keypoint graphs
(the CNN is unaffected by keypoints and is included as a reference).

Outputs: outputs/stats/corrupted_keypoints.json
Usage:   python scripts/experiment_corrupted_keypoints.py [--conditions blur:5 spatter:5]
"""
import os, sys, argparse
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
from src.features.superpoint import SuperPointExtractor
from src.features.graph_builder import GraphBuilder

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_models(config, device):
    """Load the trained CNN and Hybrid checkpoints in eval mode."""
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


def corrupt_tensor(img, kind, sev, seed):
    """Corrupt one ImageNet-normalised (3,H,W) tensor; returns normalised tensor."""
    x01 = (img * STD + MEAN).clamp(0, 1)
    return (corr.apply(x01, kind, sev, seed=seed) - MEAN) / STD


def rebuild_graph(img_norm, extractor, builder, template):
    """Re-detect DISK keypoints on the corrupted image and rebuild its k-NN graph.
    Falls back to the clean template graph if detection yields too few points."""
    x01 = (img_norm * STD + MEAN).clamp(0, 1)
    rgb = (x01.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    res = extractor.extract(rgb)
    kps, descs, scores = res['keypoints'], res['descriptors'], res['scores']
    if kps is None or len(kps) < 4:
        return template.clone(), False
    g = builder.build_graph(kps, descs, scores, image_size=rgb.shape[0])
    g.y = template.y
    return g, True


@torch.no_grad()
def embed(cnn, hyb, loader, device, kind, sev, extractor=None, builder=None):
    """Embed the test split under corruption. If `extractor` is given, graphs are
    rebuilt from the corrupted image; otherwise the clean-image graph is reused."""
    from torch_geometric.data import Batch
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ce, he, lbls = [], [], []
    n_rebuilt = n_fallback = 0
    for images, graphs, labels in loader:
        gl = graphs.to_data_list()
        imgs_c, glist = [], []
        for i in range(images.size(0)):
            ic = corrupt_tensor(images[i], kind, sev, seed=i) if kind else images[i]
            imgs_c.append(ic)
            if extractor is not None and kind:
                g, ok = rebuild_graph(ic, extractor, builder, gl[i])
                n_rebuilt += int(ok); n_fallback += int(not ok)
                glist.append(g)
            else:
                glist.append(gl[i])
        img = torch.stack(imgs_c).to(device)
        gb = Batch.from_data_list(glist).to(device)
        ce.append(F.normalize(cnn.get_embedding(img), p=2, dim=-1).float().cpu())
        with autocast(device_type='cuda', dtype=amp, enabled=(device.type == 'cuda')):
            out = hyb(img, gb)
        he.append(F.normalize(out['embedding'], p=2, dim=-1).float().cpu())
        lbls.append(labels)
    return torch.cat(ce), torch.cat(he), torch.cat(lbls), n_rebuilt, n_fallback


def score(emb, lbl, M):
    """Rank-1 / EER / AUC from embeddings."""
    r = M.compute_all_metrics(emb, lbl)['summary']
    return {'rank1': r['rank_1_accuracy'], 'eer': r['eer'], 'auc': r['roc_auc']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', nargs='+', default=['blur:5', 'spatter:5'])
    args = ap.parse_args()

    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders = create_hybrid_loaders(str(PROJECT_ROOT / config['dataset']['processed_dir']),
                                    str(PROJECT_ROOT / config['dataset']['graph_dir']), config)
    cnn, hyb = load_models(config, device)
    M = BiometricMetrics()
    kp = config.get('keypoints', {})
    extractor = SuperPointExtractor(
        max_keypoints=kp.get('max_keypoints', 128),
        detection_threshold=kp.get('detection_threshold', 0.005),
        nms_radius=kp.get('nms_radius', 4),
        backend='disk')
    kcfg = config.get('graph', {})
    builder = GraphBuilder(knn_k=kcfg.get('knn_k', 8))

    out = {}
    ce, he, lbl, _, _ = embed(cnn, hyb, loaders['test'], device, None, 0)
    out['clean'] = {'cnn': score(ce, lbl, M), 'hybrid': score(he, lbl, M)}
    print(f"clean            CNN R1={out['clean']['cnn']['rank1']*100:.1f}  "
          f"Hybrid R1={out['clean']['hybrid']['rank1']*100:.1f}", flush=True)

    for spec in args.conditions:
        kind, sev = spec.split(':'); sev = int(sev)
        # (a) clean keypoints (the protocol used in the main study)
        ce, he, lbl, _, _ = embed(cnn, hyb, loaders['test'], device, kind, sev)
        a = {'cnn': score(ce, lbl, M), 'hybrid': score(he, lbl, M)}
        # (b) keypoints recomputed from the corrupted image (honest)
        ce2, he2, lbl2, nr, nf = embed(cnn, hyb, loaders['test'], device, kind, sev,
                                       extractor=extractor, builder=builder)
        b = {'cnn': score(ce2, lbl2, M), 'hybrid': score(he2, lbl2, M),
             'graphs_rebuilt': nr, 'graphs_fallback': nf}
        out[f'{kind}_s{sev}'] = {'clean_keypoints': a, 'recomputed_keypoints': b,
                                 'hybrid_rank1_delta': (b['hybrid']['rank1'] - a['hybrid']['rank1']) * 100,
                                 'hybrid_eer_delta': (b['hybrid']['eer'] - a['hybrid']['eer']) * 100}
        print(f"{kind}-{sev}  clean-kp Hybrid R1={a['hybrid']['rank1']*100:5.1f} EER={a['hybrid']['eer']*100:5.2f} | "
              f"recomputed-kp R1={b['hybrid']['rank1']*100:5.1f} EER={b['hybrid']['eer']*100:5.2f} "
              f"(rebuilt {nr}, fallback {nf})", flush=True)

    save_stats(out, str(PROJECT_ROOT / 'outputs/stats/corrupted_keypoints.json'))
    print("\nSaved -> outputs/stats/corrupted_keypoints.json")


if __name__ == '__main__':
    main()
