"""
PyTorch Geometric utilities for ShiftKit DataManager.

Supports graph-level (many graphs per domain) and node-level (one graph per
domain) domain adaptation with stratified train/val/test splits.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .node_batch import NodeGraphBatch, is_node_graph_batch, move_node_graph_batch

try:
    from torch_geometric.data import Data, Batch
except ImportError as e:
    raise ImportError(
        "shiftkit.data.pyg_utils requires torch-geometric. "
        "Install it with:  pip install torch-geometric"
    ) from e


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
    small_classes = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_cls = len(idx)
        # Every class with at least one node contributes at least one training
        # node. Classes with 1–2 nodes have no room for validation; a class
        # with a single node has no test node either.
        if n_cls <= 2:
            n_train, n_val = 1, 0
        else:
            n_train = max(1, int(n_cls * train_ratio))
            n_val = max(0, int(n_cls * val_ratio))
            if n_train + n_val >= n_cls:
                n_train = max(1, n_cls - 1)
                n_val = 0
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train : n_train + n_val]] = True
        test_mask[idx[n_train + n_val :]] = True
        if n_cls < 3:
            small_classes.append((cls, n_cls))

    unassigned = ~(train_mask | val_mask | test_mask)
    if unassigned.any():
        idx_rest = np.where(unassigned)[0]
        rng.shuffle(idx_rest)
        test_mask[idx_rest] = True

    if small_classes:
        warnings.warn(
            "Stratified split: classes with fewer than 3 samples "
            f"{[(c.item() if hasattr(c, 'item') else c, k) for c, k in small_classes]} "
            "(label, count) get one training sample and no validation sample; "
            "a single-sample class has no test sample.",
            stacklevel=3,
        )
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


_SPLIT_PARAMS_ATTR = "_shiftkit_split_params"


def ensure_masks(
    data: Data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    split_mode: str = "stratified",
) -> Data:
    """
      Assign ``train_mask``, ``val_mask``, and ``test_mask`` on *data* in-place.

    - Masks created by this function record the split parameters used. If it
      is called again with different ``train_ratio`` / ``val_ratio`` /
      ``seed`` / ``split_mode`` (or on a graph whose size changed), the masks
      are recomputed instead of silently reusing the stale split.
    - User-supplied masks (a non-empty ``train_mask`` not created by
      ShiftKit) are respected and never overwritten.
    - The same ``Data`` object passed as both source and target is split
      with the target seed on the second call; pass separate objects (e.g.
      ``data.clone()``) if both domains share one graph.
    """
    params = (float(train_ratio), float(val_ratio), int(seed), str(split_mode),
              int(data.num_nodes))
    if _has_masks(data):
        previous = getattr(data, _SPLIT_PARAMS_ATTR, None)
        if previous is None or previous == params:
            # User-supplied masks, or masks already made with these params.
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
    # Plain Python attribute (not a graph attribute): survives copy/pickle/.to()
    # and is not collated into batches.
    setattr(data, _SPLIT_PARAMS_ATTR, params)
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

    Note: for graph-level tasks the training selection is ``train | val``;
    there is no separate validation loader. Node-level splitting differs: the
    training loader uses ``train_mask`` only and ``val_mask`` is left unused
    for the caller (e.g. for model selection).
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


_NODE_SPLIT_MASKS = {"train": "train_mask", "val": "val_mask", "test": "test_mask"}


class _SingleGraphNodeDataset(Dataset):
    """One batch per epoch: full graph + train, val or test node indices."""

    def __init__(self, data: Data, eval_split: bool = False, split: Optional[str] = None):
        self.data = data
        if split is None:
            split = "test" if eval_split else "train"
        if split not in _NODE_SPLIT_MASKS:
            raise ValueError(f"split must be one of {list(_NODE_SPLIT_MASKS)}, got {split!r}")
        self.split = split
        self.eval_split = split != "train"

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        mask = getattr(self.data, _NODE_SPLIT_MASKS[self.split])
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
    split: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Node-level loaders. ``split`` (``"train"``, ``"val"`` or ``"test"``)
    overrides ``train``; by default ``train=True`` → train nodes and
    ``train=False`` → test nodes.
    """
    if split is None:
        split = "train" if train else "test"
    src_loader = DataLoader(
        _SingleGraphNodeDataset(source, split=split),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_node_collate,
    )
    tgt_loader = DataLoader(
        _SingleGraphNodeDataset(target, split=split),
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
    split: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build (source_loader, target_loader) for ``pyg_domains`` factory.

    ``split="val"`` (node-level only) returns loaders over ``val_mask`` nodes.
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
        return build_node_loaders(source, target, batch_size, num_workers, train, split)

    if split is not None:
        raise ValueError(
            "split= is only supported for task_level='node'; graph-level "
            "splitting returns train+val (train=True) or test (train=False)."
        )

    src_graphs = split_graph_list(
        source, train, train_ratio, val_ratio, split_seed, split_mode
    )
    tgt_graphs = split_graph_list(
        target, train, train_ratio, val_ratio, split_seed + 1, split_mode
    )
    return build_graph_loaders(
        src_graphs, tgt_graphs, batch_size, num_workers, shuffle=train
    )
