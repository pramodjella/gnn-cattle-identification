"""
Script 04: Graph Construction
===============================
Builds KNN graphs from extracted keypoints for GNN processing.

Input:  data/preprocessed/keypoints/ (.npz files)
Output: data/graphs/ (.pt PyG Data files)
Stats:  outputs/stats/graph_stats.json
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_stats, ensure_dirs, setup_logging, set_seed, Timer
from src.features.graph_builder import GraphBuilder


def main():
    print("=" * 70)
    print("PHASE 4: Graph Construction")
    print("=" * 70)
    
    config = load_config()
    set_seed(config['project']['seed'])
    logger = setup_logging(config['outputs']['log_dir'], "04_build_graphs")
    
    processed_dir = PROJECT_ROOT / config['dataset']['processed_dir']
    graph_dir = PROJECT_ROOT / config['dataset']['graph_dir']
    stats_dir = PROJECT_ROOT / config['outputs']['stats_dir']
    figure_dir = PROJECT_ROOT / config['outputs']['figure_dir']
    
    ensure_dirs(str(graph_dir), str(stats_dir), str(figure_dir / "graphs"))
    
    # Initialize graph builder
    graph_config = config['graph']
    builder = GraphBuilder(
        knn_k=graph_config['knn_k'],
        normalize_positions=graph_config['normalize_positions'],
        use_relative_positions=graph_config['use_relative_positions'],
    )
    
    # Find all keypoint files
    kp_dir = processed_dir / "keypoints"
    all_kp_files = []
    
    for split_dir in kp_dir.iterdir():
        if split_dir.is_dir():
            for animal_dir in split_dir.iterdir():
                if animal_dir.is_dir():
                    for kp_file in animal_dir.glob("*.npz"):
                        all_kp_files.append({
                            'kp_path': kp_file,
                            'split': split_dir.name,
                            'animal_id': animal_dir.name,
                        })
    
    print(f"[INFO] Found {len(all_kp_files)} keypoint files")
    
    if len(all_kp_files) == 0:
        print("[WARNING] No keypoint files found. Run 03_extract_keypoints.py first.")
        return
    
    # Build animal_id to integer label mapping
    all_animals = sorted(set(item['animal_id'] for item in all_kp_files))
    animal_to_label = {aid: idx for idx, aid in enumerate(all_animals)}
    
    # Save label mapping
    label_map_path = graph_dir / "label_mapping.json"
    with open(label_map_path, 'w') as f:
        json.dump(animal_to_label, f, indent=2)
    print(f"[INFO] {len(animal_to_label)} unique animals mapped to labels")
    
    # Build graphs
    graph_data = {'train': [], 'val': [], 'test': []}
    sample_vis_count = 0
    max_vis = 15
    
    with Timer("Graph Construction") as timer:
        for item in tqdm(all_kp_files, desc="Building graphs"):
            # Load keypoint data
            kp_data = np.load(str(item['kp_path']), allow_pickle=True)
            
            keypoints = kp_data['keypoints']
            descriptors = kp_data['descriptors']
            scores = kp_data['scores']
            animal_id = item['animal_id']
            
            # Build graph
            label = animal_to_label[animal_id]
            data = builder.build_graph(
                keypoints=keypoints,
                descriptors=descriptors,
                scores=scores,
                image_size=config['preprocessing']['image_size'],
                animal_id=label,
                image_path=str(kp_data.get('image_path', '')),
            )
            
            if data is None:
                logger.warning(f"Skipped (too few keypoints): {item['kp_path']}")
                continue
            
            # Set label
            data.y = torch.tensor(label, dtype=torch.long)
            data.animal_id_str = animal_id
            
            # Add to split
            split = item['split']
            if split in graph_data:
                graph_data[split].append(data)
            else:
                graph_data['train'].append(data)
            
            # Save sample graph visualizations
            if sample_vis_count < max_vis:
                img_path = str(kp_data.get('image_path', ''))
                if img_path and os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    if image is not None:
                        builder.visualize_graph(
                            image, data,
                            output_path=str(figure_dir / "graphs" / f"graph_{sample_vis_count:03d}_{animal_id}.png")
                        )
                        sample_vis_count += 1
    
    # Save graphs per split
    for split_name, graphs in graph_data.items():
        if graphs:
            split_path = graph_dir / f"{split_name}_graphs.pt"
            torch.save(graphs, str(split_path))
            print(f"[INFO] Saved {len(graphs)} {split_name} graphs to {split_path}")
    
    # Save stats
    graph_stats = builder.get_stats()
    graph_stats['processing_time_seconds'] = timer.elapsed
    graph_stats['num_classes'] = len(animal_to_label)
    graph_stats['graphs_per_split'] = {
        split: len(graphs) for split, graphs in graph_data.items()
    }
    
    stats_path = str(stats_dir / "graph_stats.json")
    save_stats(graph_stats, stats_path)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("GRAPH CONSTRUCTION STATISTICS")
    print(f"{'=' * 70}")
    print(f"  Total Graphs:        {graph_stats['total_processed']}")
    print(f"  Skipped (too few):   {graph_stats['skipped_too_few_keypoints']}")
    print(f"  KNN K:               {graph_stats['knn_k']}")
    print(f"  Nodes/Graph:         {graph_stats['nodes_per_graph']['mean']:.1f} ± {graph_stats['nodes_per_graph']['std']:.1f}")
    print(f"  Edges/Graph:         {graph_stats['edges_per_graph']['mean']:.1f} ± {graph_stats['edges_per_graph']['std']:.1f}")
    print(f"  Avg Degree:          {graph_stats['avg_degree']['mean']:.1f}")
    print(f"  Graph Density:       {graph_stats['graph_density']['mean']:.4f}")
    print(f"  Processing Time:     {timer.elapsed:.1f}s")
    
    for split, count in graph_stats['graphs_per_split'].items():
        print(f"  {split.capitalize()} Graphs:     {count}")
    
    print(f"{'=' * 70}")
    print(f"\n[SUCCESS] [OK] Phase 4 complete! Graphs saved to {graph_dir}")
    
    return graph_stats


if __name__ == "__main__":
    main()
