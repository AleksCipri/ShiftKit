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
        batch = x.to(device)
        return batch, batch.y, True
    return x.to(device), y.to(device), False


def latents_and_targets(
    model: nn.Module, x, y: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode one batch and return ``(z, y)`` restricted to that batch's samples.

    The single place that branches on batch type: for a ``NodeGraphBatch`` the
    graph is encoded once and both ``z`` and ``y`` are indexed by ``node_idx``,
    so downstream heads never see nodes outside the current split.  For a plain
    tensor / PyG ``Batch`` it is just ``model.encode(x)``.
    """
    if is_node_graph_batch(x):
        batch = x.to(device)
        return node_latent_vectors(model, batch), batch.y
    return model.encode(x.to(device)), y.to(device)


@torch.no_grad()
def batch_accuracy(model: nn.Module, x, y: torch.Tensor) -> int:
    """Return number of correct predictions for a single batch (no grad)."""
    if is_node_graph_batch(x):
        x = x.to(next(model.parameters()).device)
        return node_classification_correct(model, x)
    return (model(x).argmax(1) == y).sum().item()


def node_latent_vectors(model: nn.Module, batch: NodeGraphBatch) -> torch.Tensor:
    """Encode the graph once and return latents for this batch's nodes only."""
    return model.encode(batch.graph)[batch.node_idx]


def node_classify_logits(model: nn.Module, batch: NodeGraphBatch) -> torch.Tensor:
    return model.classify(node_latent_vectors(model, batch))


def node_regress_preds(model: nn.Module, batch: NodeGraphBatch) -> torch.Tensor:
    preds = model.regress(node_latent_vectors(model, batch))
    return preds.view(-1) if preds.dim() > 1 and preds.size(-1) == 1 else preds.squeeze(-1)


@torch.no_grad()
def node_classification_correct(model: nn.Module, batch: NodeGraphBatch) -> int:
    logits = node_classify_logits(model, batch)
    return (logits.argmax(1) == batch.y).sum().item()


__all__ = [
    "NodeGraphBatch",
    "is_node_graph_batch",
    "move_node_graph_batch",
    "unpack_batch",
    "latents_and_targets",
    "batch_accuracy",
    "node_classify_logits",
    "node_regress_preds",
    "node_latent_vectors",
    "node_classification_correct",
]
