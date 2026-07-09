"""
Explainability Module for Cattle GNN Models
=============================================
Post-hoc interpretability tools that answer *why* the model matched
(or failed to match) a particular muzzle pattern.

Provides four complementary approaches:

1. **GNNExplainer** — learnable mask-based explanations (Ying et al., 2019).
2. **AttentionRollout** — propagate multi-layer GAT/GATv2 attention into
   a single per-node importance vector (Abnar & Zuidema, 2020 adaptation).
3. **GradCAMGraph** — gradient-weighted node activation mapping
   (Pope et al., 2019, "Explainability Methods for GNNs").
4. **ExplainabilityVisualizer** — overlay importance heatmaps on muzzle
   images at the original keypoint locations.

Compatible with:  CattleGNN, CattleGNNv3, CattleGNNPlusPlus, HybridCNNGNN.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# matplotlib is an optional runtime dependency — we import lazily in the
# visualiser so that the other three classes stay dependency-light.


# =====================================================================
# 1.  GNNExplainer Wrapper
# =====================================================================

class GNNExplainerWrapper:
    """
    Clean API around ``torch_geometric.explain.Explainer`` with
    the ``GNNExplainer`` algorithm.

    Usage::

        explainer = GNNExplainerWrapper(model, epochs=200)
        node_imp, edge_imp = explainer.explain_graph(data)

    The wrapper handles:
        • setting the model to eval mode,
        • moving data to the same device as the model,
        • extracting importance tensors from the ``Explanation`` object.
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 200,
        lr: float = 0.01,
        node_mask_type: str = 'attributes',
        edge_mask_type: str = 'object',
    ) -> None:
        """
        Args:
            model:          A trained GNN model that follows the codebase
                            convention: ``model(data) → dict`` with key
                            ``'embedding'``.
            epochs:         Number of optimisation steps for the masks.
            lr:             Learning rate for the mask optimiser.
            node_mask_type: ``'object'`` (per-node scalar),
                            ``'common_attributes'`` (shared feature mask),
                            or ``'attributes'`` (per-node-feature).
            edge_mask_type: ``'object'`` (per-edge scalar).
        """
        from torch_geometric.explain import Explainer, GNNExplainer

        self.model = model
        self.device = next(model.parameters()).device

        # Wrap the model so that Explainer can consume it
        self._wrapped_model = _EmbeddingModelWrapper(model)

        self.explainer = Explainer(
            model=self._wrapped_model,
            algorithm=GNNExplainer(epochs=epochs, lr=lr),
            explanation_type='model',
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
            model_config=dict(
                mode='regression',
                task_level='graph',
                return_type='raw',
            ),
        )

    def explain_graph(
        self,
        data: Any,
    ) -> Tuple[Tensor, Tensor]:
        """
        Explain a single graph.

        Note: this must NOT run under ``torch.no_grad`` — GNNExplainer
        optimises the node/edge masks by gradient descent internally.

        Args:
            data: PyG Data/Batch object (single graph).

        Returns:
            node_importance: (N,) per-node importance scores in [0, 1].
            edge_importance: (E,) per-edge importance scores in [0, 1].
        """
        self.model.eval()

        # Ensure single-graph batch vector
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long,
                                     device=data.x.device)

        data = data.to(self.device)
        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
            edge_attr=getattr(data, 'edge_attr', None),
        )

        # Extract masks
        node_mask = explanation.node_mask  # (N, D) or (N,)
        edge_mask = explanation.edge_mask  # (E,)

        # Collapse feature-level mask to per-node importance
        if node_mask is not None and node_mask.dim() == 2:
            node_importance = node_mask.abs().mean(dim=1)
        elif node_mask is not None:
            node_importance = node_mask.abs()
        else:
            node_importance = torch.ones(data.x.size(0), device=self.device)

        # Normalise to [0, 1]
        node_importance = _minmax_normalise(node_importance)

        if edge_mask is not None:
            edge_importance = _minmax_normalise(edge_mask.abs())
        else:
            edge_importance = torch.ones(data.edge_index.size(1),
                                         device=self.device)

        return node_importance, edge_importance


class _EmbeddingModelWrapper(nn.Module):
    """
    Thin wrapper so that ``torch_geometric.explain.Explainer`` receives
    a standard ``(x, edge_index, …) → Tensor`` callable instead of the
    dict-returning models used in this codebase.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, edge_index: Tensor,
                batch: Optional[Tensor] = None,
                edge_attr: Optional[Tensor] = None) -> Tensor:
        from torch_geometric.data import Data, Batch

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if batch is not None:
            # Construct a Batch-like object
            data.batch = batch
            data.num_graphs = int(batch.max().item()) + 1

        result = self.model(data)
        return result['embedding']


# =====================================================================
# 2.  Attention Rollout
# =====================================================================

class AttentionRollout:
    """
    Attention rollout for multi-layer GAT / GATv2 networks.

    Given a list of sparse attention weight tensors (one per layer),
    multiplies them across layers to obtain a single per-node importance
    vector that reflects how much each input node influences the final
    representation.

    Reference:  Abnar & Zuidema (2020), "Quantifying Attention Flow in
    Transformers" — adapted here for GNNs with sparse attention.

    Usage::

        rollout = AttentionRollout()
        result = model(data)
        node_imp = rollout.compute(
            attention_list=result['attention'],
            edge_index=data.edge_index,
            num_nodes=data.x.size(0),
        )
    """

    def __init__(self, add_residual: bool = True) -> None:
        """
        Args:
            add_residual: If True, add an identity matrix (self-connection)
                          before each multiplication step to account for
                          residual / skip connections in the model.
        """
        self.add_residual = add_residual

    @torch.no_grad()
    def compute(
        self,
        attention_list: List[Optional[Tensor]],
        edge_index: Tensor,
        num_nodes: int,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute attention rollout from a list of layer-wise attention
        weight tensors.

        Args:
            attention_list: List of (E_l,) or (E_l, H) attention weight
                            tensors, one per GNN layer. ``None`` entries
                            are treated as uniform attention.
            edge_index:     (2, E) edge index of the *input* graph.
            num_nodes:      Total number of nodes.
            batch:          (N,) batch assignment (for per-graph normalisation).

        Returns:
            node_importance: (N,) rollout importance scores in [0, 1].
        """
        device = edge_index.device

        # Start with identity (each node fully attends to itself)
        rollout = torch.eye(num_nodes, device=device, dtype=torch.float32)

        for alpha in attention_list:
            if alpha is None:
                continue

            # Average across heads if multi-head
            if alpha.dim() == 2:
                alpha = alpha.mean(dim=1)  # (E,)

            # Build sparse attention matrix A  (row-normalised)
            A = torch.zeros(num_nodes, num_nodes, device=device,
                            dtype=torch.float32)
            src, tgt = edge_index[0], edge_index[1]
            # Only use edges that fit (layer may have added self-loops)
            valid = (src < num_nodes) & (tgt < num_nodes)
            e = min(alpha.size(0), valid.size(0))
            valid = valid[:e]
            A[src[:e][valid], tgt[:e][valid]] = alpha[:e][valid].float()

            # Row-normalise
            row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
            A = A / row_sum

            if self.add_residual:
                # Mix with identity (residual connection)
                A = 0.5 * A + 0.5 * torch.eye(num_nodes, device=device,
                                                dtype=torch.float32)

            rollout = rollout @ A

        # Per-node importance = sum of how much each input node contributes
        # to all other nodes (column sum of the rollout matrix)
        node_importance = rollout.sum(dim=0)

        return _minmax_normalise(node_importance)


# =====================================================================
# 3.  GradCAM for Graphs
# =====================================================================

class GradCAMGraph:
    """
    Gradient-weighted Class Activation Mapping for graph neural networks.

    Hooks into a target GNN layer, computes gradients of the predicted
    class score w.r.t. the layer's output, and weights node features
    by the mean gradient to produce per-node importance scores.

    Reference:  Pope et al. (2019), "Explainability Methods for Graph
    Neural Networks".

    Usage::

        gradcam = GradCAMGraph(model, target_layer_name='gat_layers.2')
        node_imp = gradcam.attribute(data, target_class=0)
        gradcam.remove_hooks()  # clean up
    """

    def __init__(self, model: nn.Module, target_layer_name: str) -> None:
        """
        Args:
            model:              Trained GNN model.
            target_layer_name:  Dot-separated path to the target layer,
                                e.g. ``'gat_layers.2'`` or ``'trm.layers.1'``.
        """
        self.model = model
        self.target_layer_name = target_layer_name

        self._activations: Optional[Tensor] = None
        self._gradients: Optional[Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHook] = []

        # Register hooks
        target_layer = self._get_layer(model, target_layer_name)
        self._handles.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    # ---- hook callbacks ----

    def _forward_hook(self, module: nn.Module, input: Any,
                      output: Any) -> None:
        # GATConv / GATv2Conv may return (Tensor, (edge_index, alpha))
        if isinstance(output, tuple):
            self._activations = output[0].detach()
        else:
            self._activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: Any,
                       grad_output: Any) -> None:
        grad = grad_output[0]
        if isinstance(grad, tuple):
            grad = grad[0]
        self._gradients = grad.detach()

    # ---- helpers ----

    @staticmethod
    def _get_layer(model: nn.Module, name: str) -> nn.Module:
        """Resolve a dot-separated layer name to the actual module."""
        parts = name.split('.')
        module = model
        for p in parts:
            if p.isdigit():
                module = module[int(p)]
            else:
                module = getattr(module, p)
        return module

    # ---- main API ----

    def attribute(
        self,
        data: Any,
        target_class: Optional[int] = None,
    ) -> Tensor:
        """
        Compute GradCAM node importance for a single graph.

        Args:
            data:         PyG Data/Batch object (single graph).
            target_class: Class index to explain.  If ``None`` the model's
                          predicted class (argmax of embedding dot-products
                          with ArcFace prototypes, or embedding dim 0) is
                          used as a proxy score.

        Returns:
            node_importance: (N,) importance scores in [0, 1].
        """
        self.model.eval()
        # We need gradients for GradCAM
        for p in self.model.parameters():
            p.requires_grad_(True)

        # Ensure batch vector
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long,
                                     device=data.x.device)

        # Forward
        data.x.requires_grad_(True)
        result = self.model(data)
        embedding = result['embedding']  # (1, D)

        # Build a scalar score to back-prop through
        if target_class is not None and 'logits' in result:
            score = result['logits'][0, target_class]
        else:
            # Proxy: sum of embedding (encourages all dimensions)
            score = embedding.sum()

        # Backward
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                f"Hooks did not fire for layer '{self.target_layer_name}'. "
                "Check that the layer name is correct and is actually "
                "executed during the forward pass."
            )

        # GradCAM:  importance_i = ReLU(Σ_d  α_d · A_id)
        #   where α_d = (1/N) Σ_i ∂score/∂A_id   (mean gradient per feature)
        alpha = self._gradients.mean(dim=0)                  # (D,)
        weighted = (self._activations * alpha.unsqueeze(0))   # (N, D)
        node_importance = F.relu(weighted.sum(dim=1))         # (N,)

        return _minmax_normalise(node_importance.detach())

    def remove_hooks(self) -> None:
        """Remove all registered forward/backward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


# =====================================================================
# 4.  Explainability Visualiser
# =====================================================================

class ExplainabilityVisualizer:
    """
    Overlay node importance heatmaps on original muzzle images.

    Takes per-node importance scores together with the original image
    and keypoint pixel positions, and produces publication-quality
    matplotlib figures.

    Usage::

        viz = ExplainabilityVisualizer()
        fig = viz.visualize_single(
            image=img_array,
            keypoints=kp_xy,
            importance=node_imp.cpu().numpy(),
            title='GradCAM — Animal 42',
        )
        viz.save_figure(fig, 'gradcam_animal42.pdf')
    """

    # Default style settings for publication quality
    DPI: int = 300
    FIGSIZE: Tuple[float, float] = (8.0, 6.0)
    CMAP: str = 'hot'
    POINT_SIZE_BASE: float = 80.0
    POINT_SIZE_SCALE: float = 200.0
    EDGE_COLOR: str = 'white'
    EDGE_WIDTH: float = 0.3
    BG_ALPHA: float = 0.6

    def __init__(
        self,
        dpi: int = 300,
        figsize: Tuple[float, float] = (8.0, 6.0),
        cmap: str = 'hot',
    ) -> None:
        """
        Args:
            dpi:     Output resolution.
            figsize: Default figure size (width, height) in inches.
            cmap:    Matplotlib colourmap name for the heatmap.
        """
        self.DPI = dpi
        self.FIGSIZE = figsize
        self.CMAP = cmap

    # -----------------------------------------------------------------

    def visualize_single(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        importance: np.ndarray,
        title: str = 'Node Importance',
        show_colorbar: bool = True,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Overlay importance heatmap on a single muzzle image.

        Args:
            image:       (H, W, 3) or (H, W) original muzzle image.
            keypoints:   (N, 2) keypoint pixel positions (x, y).
            importance:  (N,) node importance scores in [0, 1].
            title:       Figure title.
            show_colorbar: Whether to add a colourbar.
            ax:          Optional pre-existing matplotlib Axes.

        Returns:
            matplotlib Figure (or Axes if ``ax`` was provided).
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.FIGSIZE, dpi=self.DPI)
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False

        # Show background image
        ax.imshow(image, alpha=self.BG_ALPHA)

        # Scatter keypoints — size + colour encode importance
        sizes = self.POINT_SIZE_BASE + self.POINT_SIZE_SCALE * importance
        norm = Normalize(vmin=0.0, vmax=1.0)
        sc = ax.scatter(
            keypoints[:, 0], keypoints[:, 1],
            c=importance,
            cmap=self.CMAP,
            norm=norm,
            s=sizes,
            edgecolors=self.EDGE_COLOR,
            linewidths=self.EDGE_WIDTH,
            zorder=5,
        )

        if show_colorbar:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
            cbar.set_label('Node Importance', fontsize=10)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')

        if created_fig:
            fig.tight_layout()

        return fig

    # -----------------------------------------------------------------

    def visualize_comparison(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        importances: Dict[str, np.ndarray],
        suptitle: str = 'Explainability Comparison',
    ) -> Any:
        """
        Side-by-side comparison of multiple explainability methods.

        Args:
            image:        (H, W, 3) muzzle image.
            keypoints:    (N, 2) keypoint positions.
            importances:  Mapping ``{method_name: (N,) importance}``.
            suptitle:     Overall figure title.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        n = len(importances)
        cols = min(n, 4)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                                 dpi=self.DPI)
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flat if hasattr(axes, 'flat') else [axes]

        for ax, (name, imp) in zip(axes, importances.items()):
            self.visualize_single(
                image=image,
                keypoints=keypoints,
                importance=imp,
                title=name,
                show_colorbar=True,
                ax=ax,
            )

        # Hide unused axes
        for ax in list(axes)[n:]:
            ax.set_visible(False)

        fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=1.02)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------

    @staticmethod
    def save_figure(
        fig: Any,
        path: str,
        dpi: int = 300,
        bbox_inches: str = 'tight',
    ) -> None:
        """
        Save a matplotlib figure to disk.

        Args:
            fig:         matplotlib Figure.
            path:        Output file path (e.g. ``'output.pdf'``, ``'.png'``).
            dpi:         Resolution for raster formats.
            bbox_inches: Bounding box option (``'tight'`` avoids clipping).
        """
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"[ExplainabilityVisualizer] Saved → {path}")


# =====================================================================
# Utilities
# =====================================================================

def _minmax_normalise(t: Tensor) -> Tensor:
    """Normalise a 1-D tensor to [0, 1] range."""
    t_min = t.min()
    t_max = t.max()
    if (t_max - t_min).abs() < 1e-8:
        return torch.ones_like(t)
    return (t - t_min) / (t_max - t_min)
