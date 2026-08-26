"""
PyTorch Geometric GNN for graph-level domain adaptation.

GNN
---
Configurable stack of ``torch_geometric.nn.conv`` layers with graph-level
pooling.  Accepts PyG ``Data`` / ``Batch`` objects (use
``torch_geometric.loader.DataLoader``) and exposes the same
``encode`` / ``classify`` / ``regress`` / ``forward`` interface as
:class:`~shiftkit.models.gnn.SimpleGCN` and :class:`~shiftkit.models.networks.MLP`.

Requires ``torch-geometric`` (optional dependency)::

    pip install torch-geometric
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from torch_geometric.data import Batch

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import (
        SAGEConv,
        GCNConv,
        GATConv,
        GINConv,
        GraphConv,
        global_mean_pool,
        global_max_pool,
        global_add_pool,
    )
except ImportError as e:
    raise ImportError(
        "shiftkit.models.GNN requires torch-geometric. "
        "Install it with:  pip install torch-geometric"
    ) from e

PyGData = Union[Data, "Batch"]

# ─── conv registry ───────────────────────────────────────────────────────────

_CONV_ALIASES = {
    "SAGE": "SAGE",
    "SAGECONV": "SAGE",
    "GCN": "GCN",
    "GCNCONV": "GCN",
    "GAT": "GAT",
    "GATCONV": "GAT",
    "GIN": "GIN",
    "GINCONV": "GIN",
    "GRAPHCONV": "GRAPHCONV",
    "GRAPH": "GRAPHCONV",
}

_CONV_CLASSES = {
    "SAGE": SAGEConv,
    "GCN": GCNConv,
    "GAT": GATConv,
    "GIN": GINConv,
    "GRAPHCONV": GraphConv,
}

_POOL_FUNCS = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "sum": global_add_pool,
    "add": global_add_pool,
}


def _resolve_conv(name: str) -> str:
    """Return canonical registry key for *name* (case-insensitive)."""
    key = _CONV_ALIASES.get(name.upper().replace("_", ""))
    if key is None:
        supported = sorted({k for k in _CONV_CLASSES})
        raise ValueError(
            f"Unknown GNN model '{name}'. Supported: {supported}"
        )
    return key


def _gin_mlp(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(),
        nn.Linear(out_channels, out_channels),
    )


def _build_conv(
    conv_key: str,
    in_channels: int,
    out_channels: int,
    aggr: str = "mean",
) -> nn.Module:
    """Instantiate one convolution layer for the chosen architecture."""
    if conv_key == "SAGE":
        return SAGEConv(in_channels, out_channels, aggr=aggr)
    if conv_key == "GCN":
        return GCNConv(in_channels, out_channels)
    if conv_key == "GAT":
        return GATConv(in_channels, out_channels, heads=1, concat=False)
    if conv_key == "GIN":
        return GINConv(_gin_mlp(in_channels, out_channels))
    if conv_key == "GRAPHCONV":
        return GraphConv(in_channels, out_channels, aggr=aggr)
    raise ValueError(f"Unhandled conv key: {conv_key}")


def _graph_batch_vector(data: PyGData) -> torch.Tensor:
    """Return per-node batch indices; synthesise zeros for a single graph."""
    if getattr(data, "batch", None) is not None:
        return data.batch
    return torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)


# ─── GNN model ─────────────────────────────────────────────────────────────────

class GNN(nn.Module):
    """
    Configurable PyG GNN for graph-level classification or regression.

    Parameters
    ----------
    data            : template ``Data`` object (used for ``num_node_features``)
    model_name      : conv type — ``SAGE``, ``GCN``, ``GAT``, ``GIN``, ``GraphConv``
    hidden_channels : width of conv layers and graph-level latent size
    num_layers      : number of message-passing layers (>= 1)
    num_classes     : output classes for ``classify`` (required if ``regress=False``)
    regress         : if ``True``, build regression head only; ``forward`` uses ``regress``
    pool            : graph readout — ``mean``, ``max``, ``sum``, ``add``, or ``none`` (node-level, no pool)
    use_layernorm   : apply ``LayerNorm`` after each conv
    dropout         : dropout probability between conv layers
    aggr            : aggregation for convs that support it (e.g. SAGE, GraphConv)
    predict_var     : if ``True`` (requires ``regress=True``), the regression head outputs
                      2 columns ``[mean, log_var]`` instead of a single scalar, for
                      heteroscedastic-uncertainty training (e.g. Gaussian NLL loss).
    device          : ``torch.device`` or device string (e.g. ``"cuda"``, ``"cpu"``) to
                      place the model on. If not passed (``None``), defaults to ``"cuda"``
                      when available, otherwise ``"cpu"``. If ``"cuda"`` is requested but
                      unavailable, falls back to ``"cpu"``.
    """

    def __init__(
        self,
        data: Data,
        model_name: str,
        hidden_channels: int,
        num_layers: int,
        num_classes: int = 2,
        regress: bool = False,
        pool: str = "mean",
        use_layernorm: bool = True,
        dropout: float = 0.0,
        aggr: str = "mean",
        predict_var: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        pool_key = pool.lower()
        if pool_key not in _POOL_FUNCS and pool_key != "none":
            raise ValueError(
                f"Unknown pool '{pool}'. Choose from: {list(_POOL_FUNCS)} + ['none']"
            )

        if not regress and num_classes < 1:
            raise ValueError("num_classes must be >= 1 when regress=False")

        if predict_var and not regress:
            raise ValueError("predict_var=True requires regress=True")

        conv_key = _resolve_conv(model_name)
        in_channels = data.num_node_features

        self.is_regression = regress
        self.latent_dim = hidden_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool_key
        self._pool_fn = None if pool_key == "none" else _POOL_FUNCS[pool_key]
        self._conv_key = conv_key

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(_build_conv(conv_key, in_channels, hidden_channels, aggr))
        self.norms.append(
            nn.LayerNorm(hidden_channels) if use_layernorm else nn.Identity()
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                _build_conv(conv_key, hidden_channels, hidden_channels, aggr)
            )
            self.norms.append(
                nn.LayerNorm(hidden_channels) if use_layernorm else nn.Identity()
            )

        self.predict_var = predict_var
        if regress:
            self.regressor = nn.Sequential(nn.Linear(hidden_channels, 128),
                                            nn.ReLU(), 
                                            nn.ReLU(), 
                                            nn.Linear(128, 2 if predict_var else 1))
            self.classifier = None
        else:
            self.classifier = nn.Linear(hidden_channels, num_classes)
            self.regressor = None

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
        self.device = device
        self.to(self.device)

    def encode(self, data: PyGData) -> torch.Tensor:
        """
        Message passing, then optional global pooling.

        Parameters
        ----------
        data : PyG ``Data`` or ``Batch``. Moved onto ``self.device`` automatically
               if not already there (mutates ``data`` in place, per PyG's ``.to()``).

        Returns
        -------
        z : (num_nodes, hidden_channels) if ``pool='none'``, else (num_graphs, hidden_channels)
        """
        if data.x.device != self.device:
            data = data.to(self.device)
        x, edge_index = data.x, data.edge_index

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            if self.dropout > 0.0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self._pool_fn is None:
            return h
        batch = _graph_batch_vector(data)
        return self._pool_fn(h, batch)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Linear classification head on graph-level features."""
        if self.classifier is None:
            raise RuntimeError(
                "GNN was built with regress=True; classify() is not available."
            )
        return self.classifier(z)

    def regress(self, z: torch.Tensor) -> torch.Tensor:
        """Linear regression head. Scalar output per graph, or ``[mean, log_var]``
        (2 columns) when the model was built with ``predict_var=True``."""
        if self.regressor is None:
            raise RuntimeError(
                "GNN was built with regress=False; regress() is not available. "
                "Use classify() or reconstruct with regress=True."
            )
        return self.regressor(z)

    def forward(self, data: PyGData) -> torch.Tensor:
        z = self.encode(data)
        if self.is_regression:
            return self.regress(z)
        return self.classify(z)
