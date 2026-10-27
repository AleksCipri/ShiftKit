"""
Shared helpers for node-level PyG batches (single graph per domain).
"""

from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn

from shiftkit.data.node_batch import NodeGraphBatch, is_node_graph_batch, move_node_graph_batch
from .metrics import split_target_batch


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


@contextmanager
def eval_mode(model: nn.Module):
    """Temporarily switch *model* to eval mode, restoring its previous mode.

    ``torch.no_grad()`` alone does not stop BatchNorm running-statistic updates
    or dropout, so monitoring passes must also use evaluation mode.
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)


@torch.no_grad()
def batch_accuracy(model: nn.Module, x, y: torch.Tensor) -> int:
    """Return number of correct predictions for a single batch.

    Runs without gradients and in evaluation mode, so monitoring never updates
    BatchNorm statistics or applies dropout; the previous mode is restored.
    """
    with eval_mode(model):
        if is_node_graph_batch(x):
            x = x.to(next(model.parameters()).device)
            return node_classification_correct(model, x)
        return (model(x).argmax(1) == y).sum().item()


def unpack_target_batch(batch, device: torch.device) -> Tuple[object, Optional[torch.Tensor]]:
    """Return ``(x, y_or_None)`` for a target batch, moved to *device*.

    Accepts ``x``, ``(x,)`` or ``(x, y)``.  Node batches always carry labels in
    ``batch.y``; plain tensor batches without labels give ``y = None``.
    """
    x, y = split_target_batch(batch)
    if is_node_graph_batch(x):
        x = x.to(device)
        return x, x.y
    return x.to(device), (None if y is None else y.to(device))


def _encode_one(model: nn.Module, x) -> torch.Tensor:
    if is_node_graph_batch(x):
        return node_latent_vectors(model, x)
    return model.encode(x)


def encode_domains(model: nn.Module, x_src, x_tgt=None):
    """Encode source and (optionally) target inputs; return ``(z_src, z_tgt)``.

    Plain tensor batches are encoded in **one joint forward pass**
    (``torch.cat`` then split), so in training mode BatchNorm normalises both
    domains with shared batch statistics rather than each domain with its own.
    Node batches and other batch types (e.g. PyG ``Batch``) are encoded
    separately.  ``z_tgt`` is ``None`` when *x_tgt* is ``None``.
    """
    if x_tgt is None:
        return _encode_one(model, x_src), None
    if torch.is_tensor(x_src) and torch.is_tensor(x_tgt):
        n_src = x_src.size(0)
        z = model.encode(torch.cat([x_src, x_tgt], dim=0))
        return z[:n_src], z[n_src:]
    return _encode_one(model, x_src), _encode_one(model, x_tgt)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: Iterable, device: torch.device,
                      domain: str = "source") -> dict:
    """Classification accuracy on a labelled loader, in evaluation mode.

    Handles tensor batches, PyG ``Batch`` objects and ``NodeGraphBatch``.
    Raises ``ValueError`` for unlabelled or empty loaders.
    """
    correct = total = 0
    for batch in loader:
        x, y = unpack_target_batch(batch, device)
        if y is None:
            raise ValueError("evaluate() needs a labelled loader yielding (x, y) batches")
        correct += batch_accuracy(model, x, y)
        total   += len(y)
    if total == 0:
        raise ValueError(f"cannot evaluate on an empty loader (domain={domain!r})")
    return {"domain": domain, "accuracy": correct / total, "n_samples": total}


def paired_batches(source_loader: Iterable, target_loader: Iterable) -> Tuple[Iterator, int]:
    """Pair source and target batches for one epoch.

    An epoch covers the *longer* loader; the shorter one is restarted as needed
    (each restart re-shuffles if its sampler shuffles).  Returns the iterator
    and the number of batches.  Raises a clear error for empty loaders.
    """
    n_src, n_tgt = len(source_loader), len(target_loader)
    if n_src == 0 or n_tgt == 0:
        raise ValueError(
            f"source and target loaders must be non-empty (got {n_src} and {n_tgt} batches)")
    if n_src >= n_tgt:
        pairs = zip(source_loader, _restarting(target_loader))
    else:
        pairs = zip(_restarting(source_loader), target_loader)
    return pairs, max(n_src, n_tgt)


def _restarting(loader: Iterable) -> Iterator:
    while True:
        yielded = False
        for item in loader:
            yielded = True
            yield item
        if not yielded:
            return


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
    "eval_mode",
    "paired_batches",
    "unpack_target_batch",
    "encode_domains",
    "evaluate_accuracy",
    "node_classify_logits",
    "node_regress_preds",
    "node_latent_vectors",
    "node_classification_correct",
]
