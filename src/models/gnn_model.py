"""
CattleGNN: Full Graph Neural Network for Cattle Identification
================================================================
Complete GNN architecture combining:
1. Input projection
2. Dynamic EdgeConv blocks (spatial feature learning)
3. Topological Relation Module (topological invariant learning)
4. Global pooling (graph-level representation)
5. Projection head (normalized embedding for metric learning)

The model produces a 256-dimensional embedding for each muzzle graph,
optimized via triplet loss for maximum inter-animal discrimination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from .edge_conv import DynamicEdgeConvBlock
from .trm import TopologicalRelationModule


class CattleGNN(nn.Module):
    """
    Graph Neural Network for cattle muzzle biometric identification.
    
    Architecture:
        SuperPoint Descriptors (256-d)
            → Input Projection (Linear + BN + ReLU)
            → 3× Dynamic EdgeConv Blocks
            → Topological Relation Module (GAT)
            → Global Pooling (Mean + Max → concatenate)
            → Projection Head (MLP → L2 normalize)
            → 256-d Embedding
    """
    
    def __init__(self, config=None, 
                 input_dim=256, 
                 edge_conv_dims=None,
                 edge_conv_k=12,
                 trm_hidden=256,
                 trm_heads=4,
                 trm_layers=2,
                 embedding_dim=256,
                 projection_hidden=512,
                 dropout=0.3,
                 use_batch_norm=True):
        """
        Args:
            config: Configuration dict (overrides other params)
            input_dim: Input feature dimension (256 for SuperPoint)
            edge_conv_dims: Hidden dims for EdgeConv layers
            edge_conv_k: KNN k for dynamic graph
            trm_hidden: TRM hidden dimension
            trm_heads: TRM attention heads
            trm_layers: TRM layers
            embedding_dim: Output embedding dimension
            projection_hidden: Hidden dim in projection head
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # Parse config if provided
        if config is not None:
            model_cfg = config.get('model', {})
            ec_cfg = model_cfg.get('edge_conv', {})
            trm_cfg = model_cfg.get('trm', {})
            
            edge_conv_dims = ec_cfg.get('hidden_dims', [256, 256, 512])
            edge_conv_k = ec_cfg.get('k_dynamic', 12)
            dropout = ec_cfg.get('dropout', 0.3)
            trm_hidden = trm_cfg.get('hidden_dim', 256)
            trm_heads = trm_cfg.get('num_heads', 4)
            trm_layers = trm_cfg.get('num_layers', 2)
            embedding_dim = model_cfg.get('embedding_dim', 256)
            projection_hidden = model_cfg.get('projection_hidden', 512)
            use_batch_norm = model_cfg.get('use_batch_norm', True)
        
        if edge_conv_dims is None:
            edge_conv_dims = [256, 256, 512]
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # 1. Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, edge_conv_dims[0]),
            nn.BatchNorm1d(edge_conv_dims[0]) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # 2. Dynamic EdgeConv Blocks
        self.edge_conv = DynamicEdgeConvBlock(
            in_dim=edge_conv_dims[0],
            hidden_dims=edge_conv_dims,
            k=edge_conv_k,
            aggr='max',
            dropout=dropout,
        )
        
        # 3. Topological Relation Module
        self.trm = TopologicalRelationModule(
            in_dim=edge_conv_dims[-1],
            hidden_dim=trm_hidden,
            num_heads=trm_heads,
            num_layers=trm_layers,
            dropout=dropout * 0.67,  # Slightly lower dropout in TRM
        )
        
        # 4. Global Pooling combines mean and max
        pool_dim = trm_hidden * 2  # mean + max concatenation
        
        # 5. Projection Head (for metric learning)
        self.projection_head = nn.Sequential(
            nn.Linear(pool_dim, projection_hidden),
            nn.BatchNorm1d(projection_hidden) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(projection_hidden, embedding_dim),
        )
        
        # Optional: Classification head for auxiliary loss
        self._num_classes = None
        self.classifier = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def set_num_classes(self, num_classes):
        """Set up classification head (for auxiliary CE loss)."""
        self._num_classes = num_classes
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Batch object with:
                - x: Node features (N_total, input_dim)
                - edge_index: Edge connections (2, E)
                - batch: Batch vector (N_total,)
                
        Returns:
            dict with:
                - 'embedding': L2-normalized graph embedding (B, embedding_dim)
                - 'logits': Classification logits if classifier set (B, num_classes)
                - 'attention': GAT attention weights from TRM
                - 'node_features': Final node-level features before pooling
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 1. Input projection
        x = self.input_proj(x)
        
        # 2. EdgeConv blocks (dynamic graph recomputation)
        x, intermediates = self.edge_conv(x, batch=batch)
        
        # 3. Topological Relation Module
        x, attention_weights = self.trm(x, edge_index, batch=batch)
        
        node_features = x  # Save for potential keypoint-level matching
        
        # 4. Global Pooling
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        else:
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]
        
        x_pooled = torch.cat([x_mean, x_max], dim=-1)
        
        # 5. Projection head
        embedding = self.projection_head(x_pooled)
        
        # L2 normalize for metric learning
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        result = {
            'embedding': embedding,
            'attention': attention_weights,
            'node_features': node_features,
        }
        
        # Optional classification
        if self.classifier is not None:
            result['logits'] = self.classifier(embedding)
        
        return result
    
    def get_embedding(self, data):
        """Get only the embedding (inference mode)."""
        with torch.no_grad():
            result = self.forward(data)
        return result['embedding']
    
    def summary(self):
        """Print model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = {
            'architecture': 'CattleGNN (EdgeConv + TRM)',
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': total_params * 4 / 1e6,  # Float32
        }
        
        print(f"\n{'=' * 50}")
        print("CattleGNN Model Summary")
        print(f"{'=' * 50}")
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:,.0f}" if isinstance(v, int) else f"  {k}: {v:,.2f}")
            else:
                print(f"  {k}: {v}")
        print(f"{'=' * 50}")
        
        return summary
