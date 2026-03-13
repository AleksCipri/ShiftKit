"""
Graph Neural Network architectures for domain adaptation.

SimpleGCN
---------
A 2-layer Graph Convolutional Network (Kipf & Welling, 2017) for graph-level
binary classification.  Designed to pair with SyntheticGraphDataset without
requiring PyTorch Geometric or any external graph library.

Input format
------------
Each sample x has shape (n_nodes, n_nodes + feat_dim):
    adj   = x[:, :n_nodes]          -- adjacency matrix  (N × N)
    feats = x[:, n_nodes:]          -- node features      (N × feat_dim)

A batch from DataLoader has shape (B, n_nodes, n_nodes + feat_dim).

Architecture
------------
    (B, N, N+d)  ──► split adj / feats
    feats        ──► GCN layer 1 (feat_dim  → hidden_dim)
    feats        ──► GCN layer 2 (hidden_dim → latent_dim)
    latent nodes ──► mean-pool over nodes  →  (B, latent_dim)   [encode]
    z            ──► Linear → num_classes                        [classify]

Reference
---------
Kipf, T. N., & Welling, M. (2017).
Semi-Supervised Classification with Graph Convolutional Networks.
ICLR 2017. https://arxiv.org/abs/1609.02907
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalisation with self-loops: Â = D̂^{-1/2} (A+I) D̂^{-1/2}.

    Parameters
    ----------
    adj : (B, N, N)  — raw (0/1) adjacency matrices

    Returns
    -------
    a_hat : (B, N, N)
    """
    n = adj.size(-1)
    a = adj + torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
    deg = a.sum(dim=-1)                          # (B, N)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt = d_inv_sqrt.masked_fill(deg == 0, 0.0)
    d = torch.diag_embed(d_inv_sqrt)             # (B, N, N)
    return torch.bmm(torch.bmm(d, a), d)         # D^{-1/2} (A+I) D^{-1/2}


class SimpleGCN(nn.Module):
    """
    2-layer GCN for graph-level classification.

    Works directly with the packed tensor format from SyntheticGraphDataset:
    ``x`` has shape ``(B, n_nodes, n_nodes + feat_dim)`` where the first
    ``n_nodes`` columns are the adjacency matrix and the rest are node features.

    The encode/classify interface matches all other ShiftKit models, so all
    DA methods (MMD, DANN, SIDDA, …) can be used without modification.

    Parameters
    ----------
    n_nodes    : number of nodes per graph (must match dataset)
    feat_dim   : node feature dimensionality (must match dataset)
    latent_dim : graph-level embedding size after pooling
    num_classes: number of output classes
    hidden_dim : GCN hidden layer width
    dropout    : dropout probability between GCN layers
    """

    def __init__(
        self,
        n_nodes: int = 10,
        feat_dim: int = 4,
        latent_dim: int = 64,
        num_classes: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_nodes   = n_nodes
        self.feat_dim  = feat_dim
        self.latent_dim = latent_dim

        # GCN weight matrices (applied node-wise via nn.Linear on last dim)
        self.gcn1 = nn.Linear(feat_dim,   hidden_dim, bias=False)
        self.gcn2 = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.drop  = nn.Dropout(dropout)

        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Graph encoding: two GCN layers + mean pooling over nodes.

        Parameters
        ----------
        x : (B, N, N + feat_dim)

        Returns
        -------
        z : (B, latent_dim)
        """
        adj  = x[:, :, :self.n_nodes]             # (B, N, N)
        h    = x[:, :, self.n_nodes:]             # (B, N, feat_dim)

        a_hat = _norm_adj(adj)                     # (B, N, N)

        # Layer 1: H = ReLU(A_hat @ (H @ W1))
        h = F.relu(torch.bmm(a_hat, self.gcn1(h)))  # (B, N, hidden_dim)
        h = self.drop(h)

        # Layer 2: H = ReLU(A_hat @ (H @ W2))
        h = F.relu(torch.bmm(a_hat, self.gcn2(h)))  # (B, N, latent_dim)

        # Mean pooling over nodes → graph-level embedding
        return h.mean(dim=1)                        # (B, latent_dim)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x))
