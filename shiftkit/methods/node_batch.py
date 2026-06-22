"""
Shared helpers for node-level PyG batches (single graph per domain).
"""

from typing import Tuple, Union

import torch
import torch.nn as nn

from shiftkit.data.pyg_utils import NodeGraphBatch, is_node_graph_batch, move_node_graph_batch


def unpack_batch(
    x, y: torch.Tensor, device: torch.device
) -> Tuple[Union[torch.Tensor, NodeGraphBatch], torch.Tensor, bool]:
    """Return ``(x, y, is_node_batch)``; move tensors/batches to *device*."""
    if is_node_graph_batch(x):
        batch = move_node_graph_batch(x, device)
        return batch, batch.y, True
    return x.to(device), y.to(device), False


@torch.no_grad()
def batch_accuracy(model: nn.Module, x, y: torch.Tensor) -> int:
    """Return number of correct predictions for a single batch (no grad)."""
    if is_node_graph_batch(x):
        x = move_node_graph_batch(x, next(model.parameters()).device)
        return node_classification_correct(model, x)
    return (model(x).argmax(1) == y).sum().item()


def node_classify_logits(model: nn.Module, batch: NodeGraphBatch) -> torch.Tensor:
    z = model.encode(batch.graph)
    return model.classify(z[batch.node_idx])


def node_regress_preds(model: nn.Module, batch: NodeGraphBatch) -> torch.Tensor:
    z = model.encode(batch.graph)
    preds = model.regress(z[batch.node_idx])
    return preds.view(-1) if preds.dim() > 1 and preds.size(-1) == 1 else preds.squeeze(-1)


def node_latent_vectors(model: nn.Module, batch: NodeGraphBatch) -> torch.Tensor:
    z = model.encode(batch.graph)
    return z[batch.node_idx]


@torch.no_grad()
def node_classification_correct(model: nn.Module, batch: NodeGraphBatch) -> int:
    logits = node_classify_logits(model, batch)
    return (logits.argmax(1) == batch.y).sum().item()


__all__ = [
    "NodeGraphBatch",
    "is_node_graph_batch",
    "move_node_graph_batch",
    "unpack_batch",
    "batch_accuracy",
    "node_classify_logits",
    "node_regress_preds",
    "node_latent_vectors",
    "node_classification_correct",
]
