"""
Graph Construction Module
=========================
Builds KNN graphs from detected keypoints for GNN processing.

Each detected keypoint becomes a graph node with:
- Node features: SuperPoint descriptor (256-dim)
- Edge connections: K-Nearest Neighbors based on spatial distance
- Edge features: Relative position, distance, and angle
"""

import numpy as np
import torch
from scipy.spatial import KDTree
from torch_geometric.data import Data


class GraphBuilder:
    """Build KNN graphs from keypoint detections."""
    
    def __init__(self, knn_k=12, normalize_positions=True, use_relative_positions=True):
        """
        Args:
            knn_k: Number of nearest neighbors for graph construction
            normalize_positions: Whether to normalize keypoint positions to [0, 1]
            use_relative_positions: Whether to include relative position as edge features
        """
        self.knn_k = knn_k
        self.normalize_positions = normalize_positions
        self.use_relative_positions = use_relative_positions
        
        self.stats = {
            'total_processed': 0,
            'num_nodes': [],
            'num_edges': [],
            'graph_density': [],
            'avg_degree': [],
            'min_keypoints_seen': float('inf'),
            'skipped_too_few': 0,
        }
    
    def build_graph(self, keypoints, descriptors, scores, image_size=256, 
                    animal_id=None, image_path=None):
        """
        Build a KNN graph from keypoints.
        
        Args:
            keypoints: (N, 2) array of (x, y) positions
            descriptors: (N, 256) array of descriptors
            scores: (N,) array of detection scores
            image_size: Size of the source image (for normalization)
            animal_id: Label for the animal (class ID)
            image_path: Source image path
            
        Returns:
            torch_geometric.data.Data object representing the graph
        """
        self.stats['total_processed'] += 1
        n_keypoints = len(keypoints)
        
        # Handle edge cases
        if n_keypoints < 3:
            self.stats['skipped_too_few'] += 1
            self.stats['min_keypoints_seen'] = min(
                self.stats['min_keypoints_seen'], n_keypoints
            )
            return None
        
        # Actual K for this graph (can't have more neighbors than nodes - 1)
        actual_k = min(self.knn_k, n_keypoints - 1)
        
        # Normalize positions
        positions = keypoints.copy()
        if self.normalize_positions:
            positions = positions / image_size
        
        # Build KNN graph using KDTree
        tree = KDTree(positions)
        distances, indices = tree.query(positions, k=actual_k + 1)  # +1 includes self
        
        # Build edge index (exclude self-loops)
        edge_sources = []
        edge_targets = []
        edge_features_list = []
        
        for i in range(n_keypoints):
            for j_idx in range(1, actual_k + 1):  # Skip self (index 0)
                j = indices[i, j_idx]
                edge_sources.append(i)
                edge_targets.append(j)
                
                if self.use_relative_positions:
                    # Compute edge features
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = distances[i, j_idx]
                    angle = np.arctan2(dy, dx)
                    
                    # Relative scale (ratio of detection scores)
                    if scores[i] > 0:
                        rel_scale = scores[j] / (scores[i] + 1e-8)
                    else:
                        rel_scale = 1.0
                    
                    edge_features_list.append([dx, dy, dist, angle, rel_scale])
        
        # Convert to tensors
        edge_index = torch.tensor(
            [edge_sources, edge_targets], dtype=torch.long
        )
        
        # Node features: descriptors
        x = torch.tensor(descriptors, dtype=torch.float32)
        
        # Node positions
        pos = torch.tensor(positions, dtype=torch.float32)
        
        # Edge features
        if edge_features_list:
            edge_attr = torch.tensor(
                np.array(edge_features_list), dtype=torch.float32
            )
        else:
            edge_attr = None
        
        # Create PyG Data object
        data = Data(
            x=x,                          # Node features (N, 256)
            edge_index=edge_index,         # Edge connections (2, E)
            edge_attr=edge_attr,           # Edge features (E, 5)
            pos=pos,                       # Node positions (N, 2)
        )
        
        # Store metadata
        data.num_keypoints = n_keypoints
        data.keypoint_scores = torch.tensor(scores, dtype=torch.float32)
        
        if animal_id is not None:
            data.animal_id = animal_id
        if image_path is not None:
            data.image_path = str(image_path)
        
        # Record stats
        num_edges = edge_index.shape[1]
        max_edges = n_keypoints * (n_keypoints - 1)
        density = num_edges / max_edges if max_edges > 0 else 0
        avg_degree = num_edges / n_keypoints if n_keypoints > 0 else 0
        
        self.stats['num_nodes'].append(n_keypoints)
        self.stats['num_edges'].append(num_edges)
        self.stats['graph_density'].append(density)
        self.stats['avg_degree'].append(avg_degree)
        
        return data
    
    def visualize_graph(self, image, data, output_path=None):
        """
        Visualize graph overlay on the image.
        
        Args:
            image: Source image (BGR)
            data: PyG Data object
            output_path: Path to save visualization
            
        Returns:
            vis: Visualization image
        """
        import cv2
        
        vis = image.copy()
        if len(vis.shape) == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        h, w = vis.shape[:2]
        
        # Get positions (denormalize if needed)
        pos = data.pos.numpy()
        if self.normalize_positions:
            pos = pos * max(h, w)
        
        # Draw edges
        edge_index = data.edge_index.numpy()
        for e in range(edge_index.shape[1]):
            src = edge_index[0, e]
            dst = edge_index[1, e]
            
            pt1 = (int(pos[src, 0]), int(pos[src, 1]))
            pt2 = (int(pos[dst, 0]), int(pos[dst, 1]))
            
            cv2.line(vis, pt1, pt2, (0, 200, 0), 1, cv2.LINE_AA)
        
        # Draw nodes
        for i in range(len(pos)):
            pt = (int(pos[i, 0]), int(pos[i, 1]))
            cv2.circle(vis, pt, 4, (0, 0, 255), -1)
            cv2.circle(vis, pt, 5, (255, 255, 255), 1)
        
        # Add info text
        n = data.x.shape[0]
        e = data.edge_index.shape[1]
        cv2.putText(vis, f"N={n} E={e}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), vis)
        
        return vis
    
    def get_stats(self):
        """Return graph construction statistics."""
        if not self.stats['num_nodes']:
            return self.stats
        
        return {
            'total_processed': self.stats['total_processed'],
            'skipped_too_few_keypoints': self.stats['skipped_too_few'],
            'knn_k': self.knn_k,
            'nodes_per_graph': {
                'mean': float(np.mean(self.stats['num_nodes'])),
                'std': float(np.std(self.stats['num_nodes'])),
                'min': int(np.min(self.stats['num_nodes'])),
                'max': int(np.max(self.stats['num_nodes'])),
            },
            'edges_per_graph': {
                'mean': float(np.mean(self.stats['num_edges'])),
                'std': float(np.std(self.stats['num_edges'])),
                'min': int(np.min(self.stats['num_edges'])),
                'max': int(np.max(self.stats['num_edges'])),
            },
            'graph_density': {
                'mean': float(np.mean(self.stats['graph_density'])),
                'std': float(np.std(self.stats['graph_density'])),
            },
            'avg_degree': {
                'mean': float(np.mean(self.stats['avg_degree'])),
                'std': float(np.std(self.stats['avg_degree'])),
            },
        }
