"""
Example: node-level domain adaptation on two PyG graphs (one per domain).

Uses DataManager.load("pyg_domains") with stratified node masks and
shiftkit.models.GNN with pool="none".

Run from repo root:
    python examples/pyg_node_mmd.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch_geometric.data import Data

from shiftkit.data import DataManager
from shiftkit.models import GNN
from shiftkit.methods import MMDTrainer, SourceOnlyTrainer


def make_domain_graph(n_nodes: int, feat_dim: int, n_classes: int, seed: int, shift: float = 0.0) -> Data:
    torch.manual_seed(seed)
    x = torch.randn(n_nodes, feat_dim) + shift
    row = torch.arange(n_nodes - 1)
    edge_index = torch.stack([row, row + 1], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    y = torch.randint(0, n_classes, (n_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)


if __name__ == "__main__":
    N_NODES = 300
    FEAT = 8
    NUM_CLASSES = 3
    EPOCHS = 30

    source_graph = make_domain_graph(N_NODES, FEAT, NUM_CLASSES, seed=0, shift=0.0)
    target_graph = make_domain_graph(N_NODES, FEAT, NUM_CLASSES, seed=1, shift=1.5)

    dm = DataManager(batch_size=1, num_workers=0)
    train_src, train_tgt = dm.load(
        "pyg_domains",
        train=True,
        task_level="node",
        source=source_graph,
        target=target_graph,
        train_ratio=0.6,
        val_ratio=0.2,
        split_seed=42,
        split_mode="stratified",
    )
    test_src, test_tgt = dm.load(
        "pyg_domains",
        train=False,
        task_level="node",
        source=source_graph,
        target=target_graph,
        train_ratio=0.6,
        val_ratio=0.2,
        split_seed=42,
        split_mode="stratified",
    )

    model_so = GNN(
        source_graph, "SAGE", hidden_channels=32, num_layers=2,
        num_classes=NUM_CLASSES, pool="none",
    )
    model_mmd = GNN(
        source_graph, "SAGE", hidden_channels=32, num_layers=2,
        num_classes=NUM_CLASSES, pool="none",
    )

    print("Training Source-Only...")
    so = SourceOnlyTrainer(model_so, train_src, train_tgt, lr=1e-3, device="cpu")
    so.fit(epochs=EPOCHS)

    print("Training MMD...")
    mmd = MMDTrainer(model_mmd, train_src, train_tgt, mmd_weight=0.5, lr=1e-3, device="cpu")
    mmd.fit(epochs=EPOCHS)

    for name, trainer in [("Source-Only", so), ("MMD", mmd)]:
        r_src = trainer.evaluate(test_src, domain="source-test")
        r_tgt = trainer.evaluate(test_tgt, domain="target-test")
        print(
            f"{name:12s}  src acc={r_src['accuracy']*100:.1f}%  "
            f"tgt acc={r_tgt['accuracy']*100:.1f}%"
        )

    print("Done.")
