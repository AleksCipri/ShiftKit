"""
PyTorch Geometric utilities for ShiftKit DataManager.

Supports graph-level (many graphs per domain) and node-level (one graph per
domain) domain adaptation with stratified train/val/test splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torch_geometric.data import Data, Batch
except ImportError as e:
    raise ImportError(
        "shiftkit.data.pyg_utils requires torch-geometric. "
        "Install it with:  pip install torch-geometric"
    ) from e


# ─── batch container for node-level single-graph domains ─────────────────────

@dataclass
class NodeGraphBatch:
    """
    One full graph plus the node indices and labels for the current step.

    A node-level domain is a single graph, so a "batch" is the whole graph plus
    a mask-derived index into it.  ``graph`` is shared across steps; only
    ``node_idx`` / ``y`` describe the current split.

    Implements ``.to(device)`` and ``__len__`` so it can stand in for a plain
    tensor batch wherever a trainer writes ``x = x.to(device)``.  Reaching into
    ``.graph`` directly drops ``node_idx`` and silently widens predictions to
    every node in the graph -- go through :func:`node_latent_vectors` and friends
    instead.
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
        """Number of nodes in this split (not the number of graph nodes)."""
        return int(self.node_idx.numel())


def is_node_graph_batch(x) -> bool:
    return isinstance(x, NodeGraphBatch)


def move_node_graph_batch(batch: NodeGraphBatch, device: torch.device) -> NodeGraphBatch:
    """Backwards-compatible alias for :meth:`NodeGraphBatch.to`."""
    return batch.to(device)


# ─── mask / split helpers ────────────────────────────────────────────────────

def _has_masks(data: Data) -> bool:
    return (
        hasattr(data, "train_mask")
        and data.train_mask is not None
        and data.train_mask.any()
    )


def _labels_numpy(data: Data) -> np.ndarray:
    y = data.y
    if y is None:
        raise ValueError("PyG Data must have a 'y' attribute for stratified splitting.")
    return y.detach().cpu().numpy().reshape(-1)


def _is_discrete_labels(y: np.ndarray) -> bool:
    if y.dtype.kind in ("i", "u", "b"):
        return True
    uniq = np.unique(y)
    if len(uniq) <= 20 and np.allclose(uniq, np.round(uniq)):
        return True
    return False


def _stratified_indices(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(y)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_cls = len(idx)
        n_train = max(1, int(n_cls * train_ratio)) if n_cls > 2 else max(0, n_cls - 1)
        n_val = max(0, int(n_cls * val_ratio))
        if n_train + n_val >= n_cls:
            n_train = max(1, n_cls - 1)
            n_val = 0
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train : n_train + n_val]] = True
        test_mask[idx[n_train + n_val :]] = True

    unassigned = ~(train_mask | val_mask | test_mask)
    if unassigned.any():
        idx_rest = np.where(unassigned)[0]
        rng.shuffle(idx_rest)
        test_mask[idx_rest] = True

    return train_mask, val_mask, test_mask


def _random_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 1)
        n_val = 0
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    if not test_mask.any():
        test_mask[perm[-1]] = True
        train_mask[perm[-1]] = False
    return train_mask, val_mask, test_mask


def ensure_masks(
    data: Data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    split_mode: str = "stratified",
) -> Data:
    """
  Assign ``train_mask``, ``val_mask``, and ``test_mask`` on *data* in-place.

    Skips splitting if ``train_mask`` is already present and non-empty.
    """
    if _has_masks(data):
        return data

    y_np = _labels_numpy(data)
    n = data.num_nodes
    use_stratified = split_mode == "stratified" and _is_discrete_labels(y_np)

    if use_stratified:
        tr, va, te = _stratified_indices(y_np, train_ratio, val_ratio, seed)
    else:
        tr, va, te = _random_indices(n, train_ratio, val_ratio, seed)

    device = data.y.device if data.y is not None else "cpu"
    data.train_mask = torch.tensor(tr, dtype=torch.bool, device=device)
    data.val_mask = torch.tensor(va, dtype=torch.bool, device=device)
    data.test_mask = torch.tensor(te, dtype=torch.bool, device=device)
    return data


def _normalize_graph_list(
    graphs: Union[Data, Sequence[Data], Dataset],
) -> List[Data]:
    if isinstance(graphs, Data):
        return [graphs]
    if isinstance(graphs, Dataset):
        return [graphs[i] for i in range(len(graphs))]
    return list(graphs)


def split_graph_list(
    graphs: Union[Sequence[Data], Data, Dataset],
    train: bool,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    split_mode: str = "stratified",
) -> List[Data]:
    """
    Split a list of graphs into train+val (``train=True``) or test (``train=False``).
    """
    graph_list = _normalize_graph_list(graphs)
    n = len(graph_list)
    if n == 0:
        raise ValueError("Graph list is empty.")

    labels = []
    for g in graph_list:
        if g.y is None:
            labels.append(0)
        elif g.y.numel() == 1:
            labels.append(int(g.y.item()) if g.y.dtype in (torch.int64, torch.int32) else float(g.y.item()))
        else:
            labels.append(int(g.y.view(-1)[0].item()))
    y_np = np.array(labels)

    use_stratified = split_mode == "stratified" and _is_discrete_labels(
        y_np.astype(float) if y_np.dtype.kind == "f" else y_np
    )

    if use_stratified:
        tr, va, te = _stratified_indices(y_np, train_ratio, val_ratio, seed)
    else:
        tr, va, te = _random_indices(n, train_ratio, val_ratio, seed)

    fit_mask = tr | va
    chosen = fit_mask if train else te
    return [g for g, m in zip(graph_list, chosen) if m]


# ─── datasets / loaders ──────────────────────────────────────────────────────

class _GraphListDataset(Dataset):
    def __init__(self, graphs: List[Data]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        y = g.y
        if y is None:
            raise ValueError("Graph-level task requires graph label g.y")
        if y.dim() > 0:
            y = y.view(-1)[0]
        return g, y.long() if y.dtype in (torch.int64, torch.int32) else y.float()


def _graph_collate(items):
    graphs = [item[0] for item in items]
    ys = torch.stack([item[1] for item in items])
    return Batch.from_data_list(graphs), ys


def build_graph_loaders(
    source_graphs: List[Data],
    target_graphs: List[Data],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> Tuple[DataLoader, DataLoader]:
    src_loader = DataLoader(
        _GraphListDataset(source_graphs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_graph_collate,
    )
    tgt_loader = DataLoader(
        _GraphListDataset(target_graphs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_graph_collate,
    )
    return src_loader, tgt_loader


class _SingleGraphNodeDataset(Dataset):
    """One batch per epoch: full graph + train or test node indices."""

    def __init__(self, data: Data, eval_split: bool = False):
        self.data = data
        self.eval_split = eval_split

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        mask = self.data.test_mask if self.eval_split else self.data.train_mask
        node_idx = mask.nonzero(as_tuple=False).view(-1)
        y = self.data.y[node_idx]
        if y.dtype in (torch.int64, torch.int32):
            y = y.long().view(-1)
        else:
            y = y.float().view(-1)
        batch = NodeGraphBatch(graph=self.data, node_idx=node_idx, y=y)
        # y is returned twice so the loader yields the same ``(x, y)`` shape as
        # the tensor path; ``batch.y`` is the authoritative copy.
        return batch, y


def _node_collate(items):
    return items[0]


def build_node_loaders(
    source: Data,
    target: Data,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> Tuple[DataLoader, DataLoader]:
    eval_split = not train
    src_loader = DataLoader(
        _SingleGraphNodeDataset(source, eval_split=eval_split),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_node_collate,
    )
    tgt_loader = DataLoader(
        _SingleGraphNodeDataset(target, eval_split=eval_split),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_node_collate,
    )
    return src_loader, tgt_loader


def build_pyg_domain_loaders(
    task_level: str,
    source,
    target,
    train: bool,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
    split_mode: str,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build (source_loader, target_loader) for ``pyg_domains`` factory.
    """
    task_level = task_level.lower()
    if task_level not in ("node", "graph"):
        raise ValueError("task_level must be 'node' or 'graph'")

    if task_level == "node":
        if isinstance(source, list) or isinstance(target, list):
            raise ValueError(
                "task_level='node' expects a single PyG Data object per domain, not a list."
            )
        source = ensure_masks(source, train_ratio, val_ratio, split_seed, split_mode)
        target = ensure_masks(target, train_ratio, val_ratio, split_seed + 1, split_mode)
        return build_node_loaders(source, target, batch_size, num_workers, train)

    src_graphs = split_graph_list(
        source, train, train_ratio, val_ratio, split_seed, split_mode
    )
    tgt_graphs = split_graph_list(
        target, train, train_ratio, val_ratio, split_seed + 1, split_mode
    )
    return build_graph_loaders(
        src_graphs, tgt_graphs, batch_size, num_workers, shuffle=train
    )
