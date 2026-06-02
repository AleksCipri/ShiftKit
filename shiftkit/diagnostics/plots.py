"""
Diagnostic visualisations for domain adaptation.

plot_latent_space
-----------------
Encodes samples from source and target loaders, reduces to 2-D with a chosen
projection method (t-SNE, Isomap, or UMAP), and produces two side-by-side
scatter plots:
  - left panel:  coloured by domain (source / target)
  - right panel: coloured by class label

plot_training_history
---------------------
Accepts one or more training histories (as a dict {label: history}).
Left panel: CE loss per model. Right panel: source & target accuracy per model.

compare_latent_spaces
---------------------
Side-by-side grid comparing multiple models' latent spaces (one row per model).

plot_confusion_matrix
---------------------
Compute and display a normalised confusion matrix for one or more models on a
given DataLoader.  Accepts a single model or a dict {label: model} to compare
multiple models side-by-side.

plot_roc_curve
--------------
Plot per-class ROC curves with AUC scores.  For binary tasks the curve is a
single line; for multi-class tasks one curve per class (OvR) is drawn.
Accepts a single model or a dict {label: model} for comparison.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union


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


_PROJ_AXIS_LABELS = {
    "tsne":   ("t-SNE 1",   "t-SNE 2"),
    "isomap": ("Isomap 1",  "Isomap 2"),
    "umap":   ("UMAP 1",    "UMAP 2"),
}


def _run_projection(
    z_src: np.ndarray,
    z_tgt: np.ndarray,
    method: str,
    perplexity: float,
    n_iter: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple:
    """
    Reduce concatenated [z_src; z_tgt] to 2-D using the chosen method.

    Returns (z2d, domain_labels) where domain_labels is 0=source, 1=target.
    """
    z_all = np.concatenate([z_src, z_tgt], axis=0)
    domain_labels = np.array([0] * len(z_src) + [1] * len(z_tgt))
    method = method.lower()

    if method not in _PROJ_AXIS_LABELS:
        raise ValueError(
            f"Unknown projection {method!r}. Choose from: "
            + ", ".join(f"'{m}'" for m in _PROJ_AXIS_LABELS)
        )

    print(f"  Running {method.upper()} on {len(z_all)} points "
          f"(dim={z_all.shape[1]}) …")

    if method == "tsne":
        from sklearn.manifold import TSNE
        tsne_kw = dict(
            n_components=2, perplexity=perplexity,
            random_state=42, init="pca", learning_rate="auto",
        )
        try:
            reducer = TSNE(**tsne_kw, n_iter=n_iter)
        except TypeError:
            reducer = TSNE(**tsne_kw, max_iter=n_iter)
    elif method == "isomap":
        from sklearn.manifold import Isomap
        reducer = Isomap(n_components=2, n_neighbors=n_neighbors)
    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError as e:
            raise ImportError(
                "umap-learn is required for UMAP projection. "
                "Install it with:  pip install umap-learn"
            ) from e
        reducer = UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=min_dist, random_state=42,
        )

    return reducer.fit_transform(z_all), domain_labels


def _draw_domain_panel(
    ax, z2d, domain_labels, title,
    domain_names: Optional[tuple] = None,
):
    palette = ["#4C72B0", "#DD8452"]
    names = domain_names if domain_names is not None else ("Source", "Target")
    for d, (label, color) in enumerate(zip(names, palette)):
        mask = domain_labels == d
        ax.scatter(z2d[mask, 0], z2d[mask, 1],
                   c=color, label=label, s=8, alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.legend(markerscale=3, framealpha=0.8, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


@torch.no_grad()
def _collect_node_embeddings(
    model: nn.Module,
    data,
    mask_attr: str,
    device: torch.device,
    max_samples: int,
) -> np.ndarray:
    """
    Encode a single PyG graph and return latent vectors for masked nodes.

    For node-level models (``pool='none'``), ``encode`` returns shape ``(N, D)``.
    """
    mask = getattr(data, mask_attr, None)
    if mask is None:
        raise AttributeError(f"Graph has no mask attribute '{mask_attr}'")
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        raise ValueError(f"No nodes selected by {mask_attr}")

    model.eval()
    z = model.encode(data.to(device))
    if idx.numel() > max_samples:
        perm = torch.randperm(idx.numel(), device=idx.device)[:max_samples]
        idx = idx[perm]
    return z[idx].cpu().numpy()


def _draw_class_panel(ax, z2d, y_src, y_tgt, title, class_names):
    class_labels = np.concatenate([y_src, y_tgt])
    unique       = sorted(np.unique(class_labels))
    palette      = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 10)))
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
    projection: str = "tsne",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "Latent Space",
    save_path: Optional[str] = None,
    class_names: Optional[list] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot 2-D projections of the latent space for a single model.

    Left panel  — coloured by domain (source = blue, target = orange)
    Right panel — coloured by class label

    Parameters
    ----------
    projection  : dimensionality reduction method — 'tsne', 'isomap', or 'umap'
    perplexity  : t-SNE perplexity (t-SNE only)
    n_iter      : t-SNE iterations (t-SNE only)
    n_neighbors : neighbourhood size (Isomap and UMAP)
    min_dist    : minimum distance between embedded points (UMAP only)
    """
    device = _device_of(model)
    print("Collecting source embeddings …")
    z_src, y_src = _collect_embeddings(model, source_loader, device, max_samples)
    print("Collecting target embeddings …")
    z_tgt, y_tgt = _collect_embeddings(model, target_loader, device, max_samples)

    z2d, domain_labels = _run_projection(
        z_src, z_tgt, projection, perplexity, n_iter, n_neighbors, min_dist
    )
    xlabel, ylabel = _PROJ_AXIS_LABELS[projection.lower()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _draw_domain_panel(axes[0], z2d, domain_labels, "By Domain")
    _draw_class_panel (axes[1], z2d, y_src, y_tgt,  "By Class", class_names)
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
    axes[1].set_xlabel(xlabel)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig


def plot_latent_space_domains(
    model: nn.Module,
    source_graph,
    target_graph,
    max_samples_per_domain: int = 2000,
    node_mask: str = "test_mask",
    projection: str = "tsne",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    domain_names: tuple = ("Source", "Target"),
    title: str = "Latent space by domain",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a single 2-D latent projection coloured by domain only.

    Intended for node-level PyG graphs (one graph per domain), e.g. FIREbox vs
    TNG300.  Pass ``domain_names`` to label the legend (default Source / Target).

    Parameters
    ----------
    source_graph, target_graph : PyG ``Data`` objects (one graph per domain)
    max_samples_per_domain     : cap on nodes sampled per graph
    node_mask                  : mask attribute on ``Data`` (e.g. ``test_mask``)
    projection                 : ``tsne``, ``isomap``, or ``umap``
    domain_names               : legend labels for domain 0 and 1
    """
    device = _device_of(model)
    print(f"Collecting {domain_names[0]} node embeddings …")
    z_src = _collect_node_embeddings(
        model, source_graph, node_mask, device, max_samples_per_domain
    )
    print(f"Collecting {domain_names[1]} node embeddings …")
    z_tgt = _collect_node_embeddings(
        model, target_graph, node_mask, device, max_samples_per_domain
    )

    z2d, domain_labels = _run_projection(
        z_src, z_tgt, projection, perplexity, n_iter, n_neighbors, min_dist
    )
    xlabel, ylabel = _PROJ_AXIS_LABELS[projection.lower()]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _draw_domain_panel(ax, z2d, domain_labels, "By domain", domain_names=domain_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
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
    projection: str = "tsne",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    save_path: Optional[str] = None,
    class_names: Optional[list] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare latent spaces of multiple models in a grid (one row per model).

    Each row has two panels: [By Domain | By Class].

    Parameters
    ----------
    models      : dict mapping label -> model, e.g.
                  {"Source Only": baseline, "MMD": da_model}
                  Rows appear in dict insertion order.
    projection  : dimensionality reduction method — 'tsne', 'isomap', or 'umap'
    perplexity  : t-SNE perplexity (t-SNE only)
    n_iter      : t-SNE iterations (t-SNE only)
    n_neighbors : neighbourhood size (Isomap and UMAP)
    min_dist    : minimum distance between embedded points (UMAP only)
    """
    model_names = list(models.keys())
    n_models    = len(model_names)
    xlabel, ylabel = _PROJ_AXIS_LABELS[projection.lower()]

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

        z2d, domain_labels = _run_projection(
            z_src, z_tgt, projection, perplexity, n_iter, n_neighbors, min_dist
        )

        _draw_domain_panel(axes[row, 0], z2d, domain_labels, f"{name} — By Domain")
        _draw_class_panel (axes[row, 1], z2d, y_src, y_tgt,  f"{name} — By Class",
                           class_names)
        axes[row, 0].set_xlabel(xlabel)
        axes[row, 0].set_ylabel(ylabel)
        axes[row, 1].set_xlabel(xlabel)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")
    if show:
        plt.show()
    return fig


@torch.no_grad()
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
):
    """Return (true_labels, predicted_labels, softmax_probs) numpy arrays."""
    model.eval()
    ys, y_preds, probs = [], [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        y_preds.append(logits.argmax(1).cpu().numpy())
        ys.append(y.numpy())
        n += len(y)
        if n >= max_samples:
            break
    ys      = np.concatenate(ys)[:max_samples]
    y_preds = np.concatenate(y_preds)[:max_samples]
    probs   = np.concatenate(probs)[:max_samples]
    return ys, y_preds, probs


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
        epochs  = [h["epoch"]         for h in history]
        ce      = [h["ce_loss"]       for h in history]
        src_acc = [h["src_acc"] * 100 for h in history]
        tgt_acc = [h["tgt_acc"] * 100 for h in history]

        ax1.plot(epochs, ce, color=color, marker="o", ms=4, label=label)

        ax2.plot(epochs, src_acc, color=color, marker="o", ms=4,
                 ls="-",  label=f"{label} — Source")
        ax2.plot(epochs, tgt_acc, color=color, marker="s", ms=4,
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


def plot_confusion_matrix(
    models: Union[nn.Module, Dict[str, nn.Module]],
    loader: DataLoader,
    class_names: Optional[List[str]] = None,
    max_samples: int = 5000,
    normalize: bool = True,
    domain: str = "target",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a normalised confusion matrix for one or more models.

    Parameters
    ----------
    models      : a single model, or a dict {label: model} to compare side-by-side
    loader      : labelled DataLoader (source or target test split)
    class_names : list of class label strings; uses integers if None
    max_samples : maximum number of samples to evaluate
    normalize   : if True (default) show row-normalised proportions; else raw counts
    domain      : label shown in the figure title (e.g. "target", "source")
    save_path   : if set, save figure to this path
    show        : whether to call plt.show()

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import confusion_matrix

    if isinstance(models, nn.Module):
        models = {"Model": models}

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5 * n_models, 4.5),
        squeeze=False,
    )
    fig.suptitle(f"Confusion Matrix — {domain}", fontsize=13, fontweight="bold")

    for col, (name, model) in enumerate(models.items()):
        device = _device_of(model)
        y_true, y_pred, _ = _collect_predictions(model, loader, device, max_samples)

        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.where(row_sums == 0, 0.0, cm / row_sums.astype(float))

        ax = axes[0, col]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues",
                       vmin=0.0, vmax=(1.0 if normalize else None))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        n_classes = cm.shape[0]
        ticks = np.arange(n_classes)
        labels = class_names if class_names else [str(i) for i in ticks]
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)

        # annotate cells
        thresh = cm.max() / 2.0
        fmt = ".2f" if normalize else "d"
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center", fontsize=7,
                        color="white" if val > thresh else "black")

        acc = np.diag(cm).sum() / (cm.sum() if not normalize else n_classes)
        ax.set_title(f"{name}\nacc={y_true[y_pred == y_true].size / y_true.size * 100:.1f}%",
                     fontsize=10)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig


def plot_roc_curve(
    models: Union[nn.Module, Dict[str, nn.Module]],
    loader: DataLoader,
    class_names: Optional[List[str]] = None,
    max_samples: int = 5000,
    domain: str = "target",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot per-class ROC curves with AUC scores (one-vs-rest for multi-class).

    For binary tasks a single ROC curve is drawn.
    For multi-class tasks one curve per class is drawn on the same axes per model.

    Parameters
    ----------
    models      : a single model, or a dict {label: model} to compare side-by-side
    loader      : labelled DataLoader (source or target test split)
    class_names : list of class label strings; uses integers if None
    max_samples : maximum number of samples to evaluate
    domain      : label shown in the figure title
    save_path   : if set, save figure to this path
    show        : whether to call plt.show()

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    if isinstance(models, nn.Module):
        models = {"Model": models}

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5 * n_models, 4.5),
        squeeze=False,
    )
    fig.suptitle(f"ROC Curves — {domain}", fontsize=13, fontweight="bold")

    for col, (name, model) in enumerate(models.items()):
        device = _device_of(model)
        y_true, _, probs = _collect_predictions(model, loader, device, max_samples)

        n_classes = probs.shape[1]
        labels = class_names if class_names else [str(i) for i in range(n_classes)]
        ax = axes[0, col]

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2,
                    label=f"{labels[1]}  (AUC={roc_auc:.3f})")
        else:
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
            for c in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[c], lw=1.5,
                        label=f"{labels[c]}  (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig
