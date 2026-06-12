"""
Script: Re-Evaluate All Existing Model Checkpoints
===================================================
Loads best_model.pt for each model that has a checkpoint but
missing or incomplete results JSON, and fills in the gaps
(cmc_curve, fpr, tpr, etc.) needed by compare_models.py.

Run this once to repair any partially-saved results.
"""

import os
import sys
import json
import torch
from pathlib import Path
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, set_seed
from src.evaluation.metrics import BiometricMetrics


REQUIRED_FIELDS = ['cmc_curve', 'fpr', 'tpr', 'test_rank1', 'test_rank5', 'eer', 'roc_auc']


def needs_reeval(json_path):
    """Check if a results JSON is missing required fields."""
    if not os.path.exists(json_path):
        return True
    with open(json_path) as f:
        data = json.load(f)
    return any(data.get(field) is None for field in REQUIRED_FIELDS)


def eval_gnn_plus(config, device, model_key='gnn_plus'):
    """Re-evaluate GNN+ or base GNN model."""
    from src.models.gnn_model import CattleGNN
    from src.models.arcface import ArcFaceLoss
    from src.training.dataset import create_data_loaders

    cfg = config.get(model_key, {})
    ckpt_dir = PROJECT_ROOT / cfg.get('checkpoint_dir', f'outputs/{model_key}')
    result_path = PROJECT_ROOT / f'outputs/stats/{model_key}_results.json'

    ckpt_path = ckpt_dir / 'best_model.pt'
    if not ckpt_path.exists():
        print(f"  [SKIP] {model_key}: No checkpoint at {ckpt_path}")
        return

    print(f"\n  Evaluating {model_key}...")
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])

    # Load data
    loaders = create_data_loaders(graph_dir, config, augment_train=False)
    labels = [d.y.item() for d in torch.load(
        os.path.join(graph_dir, 'train_graphs.pt'), weights_only=False)]
    num_classes = len(set(labels))

    # Build model
    config_plus = dict(config)
    config_plus['model'] = dict(config['model'])
    config_plus['model']['edge_conv'] = dict(config['model']['edge_conv'])
    config_plus['model']['edge_conv']['k_dynamic'] = cfg.get('edge_conv_k_dynamic', 12)

    model = CattleGNN(config=config_plus)
    model.set_num_classes(num_classes)
    model = model.to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loaders['test']:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            all_emb.append(out['embedding'].cpu())
            all_lbl.append(batch.y.cpu())
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)

    metrics = BiometricMetrics()
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    # Load existing JSON to preserve history
    existing = {}
    if os.path.exists(str(result_path)):
        with open(str(result_path)) as f:
            existing = json.load(f)

    existing.update({
        'model': f'GNN+ (Kornia DISK + ArcFace + EdgeConv(k=12) + GraphAug)',
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
    })

    with open(str(result_path), 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"  [UPDATED] {result_path}")


def eval_gnn_plus_v2(config, device):
    """Re-evaluate GNN++ model."""
    from src.models.gnn_plus_v2 import CattleGNNPlusPlus
    from torch_geometric.loader import DataLoader

    pp_cfg = config.get('gnn_plus_v2', {})
    ckpt_dir = PROJECT_ROOT / pp_cfg.get('checkpoint_dir', 'outputs/gnn_plus_v2')
    result_path = PROJECT_ROOT / 'outputs/stats/gnn_plus_v2_results.json'
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']

    ckpt_path = ckpt_dir / 'best_model.pt'
    if not ckpt_path.exists():
        print(f"  [SKIP] gnn_plus_v2: No checkpoint at {ckpt_path}")
        return

    print(f"\n  Evaluating gnn_plus_v2...")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    input_dim = ckpt.get('input_dim', 834)
    num_classes = ckpt.get('num_classes')

    # Load test graphs
    v2_path = graph_dir / 'test_graphs_v2.pt'
    v1_path = graph_dir / 'test_graphs.pt'
    test_graphs = torch.load(str(v2_path if v2_path.exists() else v1_path), weights_only=False)
    if num_classes is None:
        train_g = torch.load(str(graph_dir / 'train_graphs.pt'), weights_only=False)
        num_classes = len(set(g.y.item() for g in train_g))

    # Check actual input dim from graphs
    if test_graphs and hasattr(test_graphs[0], 'x'):
        actual_dim = test_graphs[0].x.shape[1]
        if actual_dim != input_dim:
            print(f"  [INFO] Overriding input_dim {input_dim} → {actual_dim} from graph data")
            input_dim = actual_dim

    model = CattleGNNPlusPlus(config=config, input_dim=input_dim)
    model.set_num_classes(num_classes)
    model = model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    batch_size = pp_cfg.get('batch_size', 64)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            all_emb.append(model.get_embedding(batch).cpu())
            all_lbl.append(batch.y.cpu())
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)

    metrics = BiometricMetrics()
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    existing = {}
    if os.path.exists(str(result_path)):
        with open(str(result_path)) as f:
            existing = json.load(f)

    existing.update({
        'model': 'GNN++ (MobileNetV3 patches + 4-layer ResEdgeConv + 3-stream pool + ArcFace)',
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies'].get('rank_5', 0),
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
        'input_dim': input_dim,
        'num_classes': num_classes,
    })

    with open(str(result_path), 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"  [UPDATED] {result_path}")


def eval_cnn(config, device):
    """Re-evaluate CNN model."""
    from src.models.cnn_model import CNNMuzzleModel
    from src.training.image_dataset import create_image_loaders
    import json

    cnn_cfg = config.get('cnn', {})
    ckpt_dir = PROJECT_ROOT / cnn_cfg.get('checkpoint_dir', 'outputs/cnn')
    result_path = PROJECT_ROOT / 'outputs/stats/cnn_results.json'

    ckpt_path = ckpt_dir / 'best_model.pt'
    if not ckpt_path.exists():
        print(f"  [SKIP] cnn: No checkpoint at {ckpt_path}")
        return

    print(f"\n  Evaluating CNN...")
    preprocessed_dir = str(PROJECT_ROOT / config['dataset']['processed_dir'])
    loaders = create_image_loaders(preprocessed_dir, config)

    with open(os.path.join(preprocessed_dir, 'train_split.json')) as f:
        train_data = json.load(f)
    num_classes = len(set(
        item.get('animal_id', item.get('label', str(i)))
        for i, item in enumerate(train_data)
    ))

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    num_classes = ckpt.get('num_classes', num_classes)

    model = CNNMuzzleModel(num_classes=num_classes, embedding_dim=256, pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_emb, all_lbl = [], []
    with torch.no_grad():
        for images, labels in loaders['test']:
            images = images.to(device, non_blocking=True)
            emb = model.get_embedding(images)
            all_emb.append(emb.cpu())
            all_lbl.append(labels)
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)

    metrics = BiometricMetrics()
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    existing = {}
    if os.path.exists(str(result_path)):
        with open(str(result_path)) as f:
            existing = json.load(f)

    existing.update({
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
    })

    with open(str(result_path), 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"  [UPDATED] {result_path}")


def eval_gnn_v3_optimized(config, device):
    """Re-evaluate GNN v3 optimized model."""
    from src.models.gnn_v3 import CattleGNNv3
    from src.training.dataset import create_data_loaders

    ckpt_path = PROJECT_ROOT / 'outputs/gnn_v3/best_model.pt'
    result_path = PROJECT_ROOT / 'outputs/stats/gnn_v3_optimized_results.json'
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])

    if not ckpt_path.exists():
        print(f"  [SKIP] gnn_v3_optimized: No checkpoint at {ckpt_path}")
        return

    print(f"\n  Evaluating gnn_v3_optimized...")
    loaders = create_data_loaders(graph_dir, config, augment_train=False)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ckpt_config = ckpt.get('config', {})
    dropout = ckpt_config.get('dropout', 0.25)

    model = CattleGNNv3(
        input_dim=256,
        hidden_dim=ckpt_config.get('hidden_dim', 192),
        num_heads=ckpt_config.get('num_heads', 4),
        num_layers=ckpt_config.get('num_layers', 4),
        edge_enc_dim=ckpt_config.get('edge_enc_dim', 96),
        fusion_dim=ckpt_config.get('fusion_dim', 768),
        projection_hidden=ckpt_config.get('projection_hidden', 512),
        dropout=dropout,
    )
    
    labels = [d.y.item() for d in torch.load(
        os.path.join(graph_dir, 'train_graphs.pt'), weights_only=False)]
    num_classes = len(set(labels))
    model.set_num_classes(num_classes)
    
    model = model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loaders['test']:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            all_emb.append(out['embedding'].cpu())
            all_lbl.append(batch.y.cpu())
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)

    metrics = BiometricMetrics()
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    existing = {}
    if os.path.exists(str(result_path)):
        with open(str(result_path)) as f:
            existing = json.load(f)

    existing.update({
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
    })

    with open(str(result_path), 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"  [UPDATED] {result_path}")


def eval_gnn_v4_enhanced(config, device):
    """Re-evaluate GNN v4 enhanced model."""
    from src.models.gnn_v3 import CattleGNNv3
    from src.training.dataset import create_data_loaders

    ckpt_path = PROJECT_ROOT / 'outputs/gnn_v4/best_model.pt'
    result_path = PROJECT_ROOT / 'outputs/stats/gnn_v4_enhanced_results.json'
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])

    if not ckpt_path.exists():
        print(f"  [SKIP] gnn_v4_enhanced: No checkpoint at {ckpt_path}")
        return

    print(f"\n  Evaluating gnn_v4_enhanced...")
    loaders = create_data_loaders(graph_dir, config, augment_train=False)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ckpt_config = ckpt.get('config', {})
    dropout = ckpt_config.get('dropout', 0.30)

    model = CattleGNNv3(
        input_dim=256,
        hidden_dim=ckpt_config.get('hidden_dim', 128),
        num_heads=ckpt_config.get('num_heads', 8),
        num_layers=ckpt_config.get('num_layers', 4),
        edge_enc_dim=ckpt_config.get('edge_enc_dim', 64),
        fusion_dim=ckpt_config.get('fusion_dim', 512),
        projection_hidden=ckpt_config.get('projection_hidden', 256),
        dropout=dropout,
    )
    
    labels = [d.y.item() for d in torch.load(
        os.path.join(graph_dir, 'train_graphs.pt'), weights_only=False)]
    num_classes = len(set(labels))
    model.set_num_classes(num_classes)
    
    model = model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loaders['test']:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            all_emb.append(out['embedding'].cpu())
            all_lbl.append(batch.y.cpu())
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)

    metrics = BiometricMetrics()
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    existing = {}
    if os.path.exists(str(result_path)):
        with open(str(result_path)) as f:
            existing = json.load(f)

    existing.update({
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
    })

    with open(str(result_path), 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"  [UPDATED] {result_path}")


def eval_proton(config, device):
    """Re-evaluate ProtoN model."""
    from src.models.proton import CattleProtoN
    from src.training.dataset import create_data_loaders

    ckpt_path = PROJECT_ROOT / 'outputs/proton/best_model.pt'
    result_path = PROJECT_ROOT / 'outputs/stats/proton_results.json'
    graph_dir = str(PROJECT_ROOT / config['dataset']['graph_dir'])

    if not ckpt_path.exists():
        print(f"  [SKIP] proton: No checkpoint at {ckpt_path}")
        return

    print(f"\n  Evaluating ProtoN...")
    loaders = create_data_loaders(graph_dir, config, augment_train=False)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    
    ckpt_config = ckpt.get('config', {})
    dropout = ckpt_config.get('dropout', 0.12)

    model = CattleProtoN(
        num_classes=ckpt.get('num_classes', 260),
        input_dim=256,
        hidden_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=dropout,
    )
    
    model = model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loaders['test']:
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            all_emb.append(out['embedding'].cpu())
            all_lbl.append(batch.y.cpu())
    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl)

    metrics = BiometricMetrics()
    results = metrics.compute_all_metrics(emb, lbl)
    metrics.print_summary(results)

    existing = {}
    if os.path.exists(str(result_path)):
        with open(str(result_path)) as f:
            existing = json.load(f)

    existing.update({
        'test_rank1': results['identification']['rank_accuracies']['rank_1'],
        'test_rank5': results['identification']['rank_accuracies']['rank_5'],
        'eer': results['verification']['eer'],
        'roc_auc': results['verification']['roc_auc'],
        'cmc_curve': results['identification']['cmc_curve'],
        'fpr': results['verification']['fpr'],
        'tpr': results['verification']['tpr'],
    })

    with open(str(result_path), 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"  [UPDATED] {result_path}")


def main():
    config = load_config()
    set_seed(config['project']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensure_dirs(str(PROJECT_ROOT / 'outputs/stats'))

    print(f"\n{'='*65}")
    print("  RE-EVALUATE ALL MODELS (fill missing JSON fields)")
    print(f"{'='*65}")
    print(f"  Device: {device}")

    # Check and re-evaluate each model
    gnn_plus_json = str(PROJECT_ROOT / 'outputs/stats/gnn_plus_results.json')
    if needs_reeval(gnn_plus_json):
        print(f"\n  GNN+ results incomplete. Re-evaluating...")
        eval_gnn_plus(config, device, model_key='gnn_plus')
    else:
        print(f"\n  [SKIP] GNN+ results complete.")

    gnn_pp_json = str(PROJECT_ROOT / 'outputs/stats/gnn_plus_v2_results.json')
    if needs_reeval(gnn_pp_json):
        print(f"\n  GNN++ results incomplete. Re-evaluating...")
        eval_gnn_plus_v2(config, device)
    else:
        print(f"\n  [SKIP] GNN++ results complete.")

    gnn_v3_json = str(PROJECT_ROOT / 'outputs/stats/gnn_v3_optimized_results.json')
    if needs_reeval(gnn_v3_json):
        print(f"\n  GNN v3 optimized results incomplete. Re-evaluating...")
        eval_gnn_v3_optimized(config, device)
    else:
        print(f"\n  [SKIP] GNN v3 optimized results complete.")

    gnn_v4_json = str(PROJECT_ROOT / 'outputs/stats/gnn_v4_enhanced_results.json')
    if needs_reeval(gnn_v4_json):
        print(f"\n  GNN v4 enhanced results incomplete. Re-evaluating...")
        eval_gnn_v4_enhanced(config, device)
    else:
        print(f"\n  [SKIP] GNN v4 enhanced results complete.")

    proton_json = str(PROJECT_ROOT / 'outputs/stats/proton_results.json')
    if needs_reeval(proton_json):
        print(f"\n  ProtoN results incomplete. Re-evaluating...")
        eval_proton(config, device)
    else:
        print(f"\n  [SKIP] ProtoN results complete.")

    cnn_json = str(PROJECT_ROOT / 'outputs/stats/cnn_results.json')
    if needs_reeval(cnn_json):
        print(f"\n  CNN results incomplete. Re-evaluating...")
        eval_cnn(config, device)
    else:
        print(f"\n  [SKIP] CNN results complete.")


    print(f"\n{'='*65}")
    print("  Re-evaluation complete. Ready for compare_models.py")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
