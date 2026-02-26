"""
Diagnostic visualisations for domain adaptation.

plot_latent_space
-----------------
Encodes samples from source and target loaders, reduces to 2-D with t-SNE,
and produces two side-by-side scatter plots:
  - left panel:  coloured by domain (source / target)
  - right panel: coloured by class label

plot_training_history
---------------------
Line plot of CE loss, MMD loss, total loss, and source accuracy over epochs.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from typing import Optional


# ─── helpers ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
):
    """Return (embeddings, labels) numpy arrays, capped at max_samples."""
    model.eval()
    zs, ys = [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        z = model.encode(x).cpu().numpy()
        zs.append(z)
        ys.append(y.numpy())
        n += len(y)
        if n >= max_samples:
            break
    return np.concatenate(zs)[:max_samples], np.concatenate(ys)[:max_samples]


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ─── public API ──────────────────────────────────────────────────────────────

def plot_latent_space(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    max_samples: int = 2000,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
    title: str = "Latent Space",
    save_path: Optional[str] = None,
    class_names: Optional[list] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot 2-D t-SNE projections of the latent space.

    Left panel  — coloured by domain (source = blue, target = orange)
    Right panel — coloured by class label (10 distinct colours)

    Parameters
    ----------
    model          : trained model with .encode() method
    source_loader  : DataLoader for the source domain
    target_loader  : DataLoader for the target domain
    max_samples    : max samples to collect per domain (keeps t-SNE fast)
    tsne_perplexity: t-SNE perplexity hyperparameter
    tsne_n_iter    : t-SNE number of iterations
    title          : figure suptitle
    save_path      : if given, save figure to this path
    class_names    : list of class label strings; uses integers if None
    show           : whether to call plt.show()

    Returns
    -------
    matplotlib Figure
    """
    device = _device_of(model)

    print("Collecting source embeddings …")
    z_src, y_src = _collect_embeddings(model, source_loader, device, max_samples)
    print("Collecting target embeddings …")
    z_tgt, y_tgt = _collect_embeddings(model, target_loader, device, max_samples)

    z_all = np.concatenate([z_src, z_tgt], axis=0)
    domain_labels = np.array([0] * len(z_src) + [1] * len(z_tgt))   # 0=src, 1=tgt
    class_labels  = np.concatenate([y_src, y_tgt], axis=0)

    print(f"Running t-SNE on {len(z_all)} points (dim={z_all.shape[1]}) …")
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    z2d = tsne.fit_transform(z_all)

    # ── colours ──────────────────────────────────────────────────────────
    domain_palette = ["#4C72B0", "#DD8452"]   # blue=source, orange=target
    n_classes = len(np.unique(class_labels))
    class_palette = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 10)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    scatter_kw = dict(s=8, alpha=0.6, linewidths=0)

    # ── left: by domain ──────────────────────────────────────────────────
    ax = axes[0]
    for d, (label, color) in enumerate(zip(["Source", "Target"], domain_palette)):
        mask = domain_labels == d
        ax.scatter(z2d[mask, 0], z2d[mask, 1], color=color, label=label, **scatter_kw)
    ax.set_title("By Domain")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, framealpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    # ── right: by class ──────────────────────────────────────────────────
    ax = axes[1]
    unique_classes = sorted(np.unique(class_labels))
    for i, c in enumerate(unique_classes):
        mask = class_labels == c
        label = class_names[c] if class_names else str(c)
        ax.scatter(z2d[mask, 0], z2d[mask, 1],
                   color=class_palette[i], label=label, **scatter_kw)
    ax.set_title("By Class")
    ax.set_xlabel("t-SNE 1")
    ax.legend(markerscale=3, framealpha=0.8,
              ncol=2, fontsize=8, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig


def plot_training_history(
    history: list,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot loss curves and source accuracy from MMDTrainer.history.

    Parameters
    ----------
    history   : list of dicts returned by MMDTrainer.fit()
    save_path : optional path to save the figure
    show      : whether to call plt.show()
    """
    epochs     = [h["epoch"]      for h in history]
    ce_loss    = [h["ce_loss"]    for h in history]
    mmd_loss   = [h["mmd_loss"]   for h in history]
    total_loss = [h["total_loss"] for h in history]
    src_acc    = [h["src_acc"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    ax1.plot(epochs, ce_loss,    label="Cross-Entropy",  marker="o", ms=4)
    ax1.plot(epochs, mmd_loss,   label="MMD",            marker="s", ms=4)
    ax1.plot(epochs, total_loss, label="Total",          marker="^", ms=4, ls="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, src_acc, color="green", marker="o", ms=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Source Domain Accuracy")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig
