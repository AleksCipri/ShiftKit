"""Node-level batch containers shared by tensor and graph trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data


@dataclass
class NodeGraphBatch:
    """
    One full graph plus the node indices and labels for the current step.

    A node-level domain is a single graph, so a "batch" is the whole graph plus
    a mask-derived index into it. ``graph`` is shared across steps; only
    ``node_idx`` / ``y`` describe the current split.

    Implements ``.to(device)`` and ``__len__`` so it can stand in for a plain
    tensor batch. Use the node helpers in ``shiftkit.methods.node_batch`` to
    preserve the selected node indices when encoding or predicting.
    """

    graph: Data
    node_idx: torch.Tensor
    y: torch.Tensor

    def to(self, device: torch.device) -> "NodeGraphBatch":
        """Return a copy with graph, indices, and labels moved to *device*."""
        return NodeGraphBatch(
            graph=self.graph.to(device),
            node_idx=self.node_idx.to(device),
            y=self.y.to(device),
        )

    def __len__(self) -> int:
        """Number of nodes in this split, rather than in the whole graph."""
        return int(self.node_idx.numel())


def is_node_graph_batch(x) -> bool:
    return isinstance(x, NodeGraphBatch)


def move_node_graph_batch(batch: NodeGraphBatch, device: torch.device) -> NodeGraphBatch:
    """Backwards-compatible alias for :meth:`NodeGraphBatch.to`."""
    return batch.to(device)
