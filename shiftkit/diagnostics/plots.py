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
Accepts one or more training histories (as a dict {label: history}).
Left panel: CE loss per model. Right panel: source & target accuracy per model.

compare_latent_spaces
---------------------
Side-by-side 2×2 grid comparing two models' latent spaces (e.g. baseline vs DA).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from typing import Optional, Union


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


def _run_tsne(z_src, z_tgt, perplexity, n_iter):
    z_all = np.concatenate([z_src, z_tgt], axis=0)
    domain_labels = np.array([0] * len(z_src) + [1] * len(z_tgt))
    class_labels  = np.concatenate([
        np.zeros(len(z_src), dtype=int),   # placeholder; caller fills in
        np.zeros(len(z_tgt), dtype=int),
    ])
    print(f"  Running t-SNE on {len(z_all)} points (dim={z_all.shape[1]}) …")
    tsne = TSNE(
        n_components=2, perplexity=perplexity, n_iter=n_iter,
        random_state=42, init="pca", learning_rate="auto",
    )
    return tsne.fit_transform(z_all), domain_labels


def _draw_domain_panel(ax, z2d, domain_labels, title):
    palette = ["#4C72B0", "#DD8452"]
    for d, (label, color) in enumerate(zip(["Source", "Target"], palette)):
        mask = domain_labels == d
        ax.scatter(z2d[mask, 0], z2d[mask, 1],
                   c=color, label=label, s=8, alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.legend(markerscale=3, framealpha=0.8, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _draw_class_panel(ax, z2d, y_src, y_tgt, title, class_names):
    class_labels  = np.concatenate([y_src, y_tgt])
    unique        = sorted(np.unique(class_labels))
    palette       = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 10)))
    for i, c in enumerate(unique):
        mask  = class_labels == c
        label = class_names[c] if class_names else str(c)
        ax.scatter(z2d[mask, 0], z2d[mask, 1],
                   c=[palette[i]], label=label, s=8, alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.legend(markerscale=3, framealpha=0.8, ncol=2, fontsize=7, loc="best")
    ax.set_xticks([]); ax.set_yticks([])


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
    Plot 2-D t-SNE projections of the latent space for a single model.

    Left panel  — coloured by domain (source = blue, target = orange)
    Right panel — coloured by class label
    """
    device = _device_of(model)
    print("Collecting source embeddings …")
    z_src, y_src = _collect_embeddings(model, source_loader, device, max_samples)
    print("Collecting target embeddings …")
    z_tgt, y_tgt = _collect_embeddings(model, target_loader, device, max_samples)

    z2d, domain_labels = _run_tsne(z_src, z_tgt, tsne_perplexity, tsne_n_iter)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _draw_domain_panel(axes[0], z2d, domain_labels, "By Domain")
    _draw_class_panel (axes[1], z2d, y_src, y_tgt,  "By Class", class_names)
    axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
    axes[1].set_xlabel("t-SNE 1")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig


def compare_latent_spaces(
    models: dict,
    source_loader: DataLoader,
    target_loader: DataLoader,
    max_samples: int = 2000,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 1000,
    save_path: Optional[str] = None,
    class_names: Optional[list] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare latent spaces of multiple models in a grid (one row per model).

    Each row has two panels: [By Domain | By Class].

    Parameters
    ----------
    models  : dict mapping label -> model, e.g.
              {"Source Only": baseline, "MMD": da_model}
              Rows appear in dict insertion order.
    """
    model_names = list(models.keys())
    n_models    = len(model_names)

    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(14, 6 * n_models),
        squeeze=False,
    )
    fig.suptitle("Latent Space Comparison", fontsize=14, fontweight="bold")

    for row, name in enumerate(model_names):
        model  = models[name]
        device = _device_of(model)

        print(f"\n[{name}]")
        print("  Collecting source embeddings …")
        z_src, y_src = _collect_embeddings(model, source_loader, device, max_samples)
        print("  Collecting target embeddings …")
        z_tgt, y_tgt = _collect_embeddings(model, target_loader, device, max_samples)

        z2d, domain_labels = _run_tsne(z_src, z_tgt, tsne_perplexity, tsne_n_iter)

        _draw_domain_panel(axes[row, 0], z2d, domain_labels, f"{name} — By Domain")
        _draw_class_panel (axes[row, 1], z2d, y_src, y_tgt,  f"{name} — By Class",
                           class_names)
        axes[row, 0].set_xlabel("t-SNE 1")
        axes[row, 0].set_ylabel("t-SNE 2")
        axes[row, 1].set_xlabel("t-SNE 1")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")
    if show:
        plt.show()
    return fig


def plot_training_history(
    histories: Union[list, dict],
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot loss curves and accuracy from one or more training histories.

    Parameters
    ----------
    histories : either
        - a single history list (backward-compatible), or
        - a dict {label: history_list} to overlay multiple runs
    Left panel  : CE loss per model
    Right panel : Source accuracy (solid) and Target accuracy (dashed) per model
    """
    # normalise to dict
    if isinstance(histories, list):
        histories = {"Model": histories}

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(histories), 2)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    for i, (label, history) in enumerate(histories.items()):
        color   = colors[i]
        epochs  = [h["epoch"]        for h in history]
        ce      = [h["ce_loss"]      for h in history]
        src_acc = [h["src_acc"] * 100 for h in history]
        tgt_acc = [h["tgt_acc"] * 100 for h in history]

        ax1.plot(epochs, ce, color=color, marker="o", ms=4, label=label)

        ax2.plot(epochs, src_acc, color=color, marker="o",  ms=4,
                 ls="-",  label=f"{label} — Source")
        ax2.plot(epochs, tgt_acc, color=color, marker="s",  ms=4,
                 ls="--", label=f"{label} — Target")

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("CE Loss")
    ax1.set_title("Cross-Entropy Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Source & Target Accuracy")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig
