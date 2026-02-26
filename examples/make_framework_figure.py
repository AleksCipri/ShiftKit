"""Generate the ShiftKit framework overview figure."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "framework_overview.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C = dict(
    data   = "#2E86AB",
    model  = "#28A864",
    method = "#C0392B",
    diag   = "#7D3C98",
    bg     = "#F4F6F8",
    dark   = "#2C3E50",
    code   = "#1E2D3D",
    arrow  = "#7F8C8D",
)

fig, ax = plt.subplots(figsize=(17, 7.8))
ax.set_xlim(0, 17)
ax.set_ylim(0, 7.8)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.axis("off")


# ── helpers ───────────────────────────────────────────────────────────────────

def draw_module(x, y, w, h, color, title, lines):
    header_h = 0.72
    # colored outer box (header + border)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white",
        linewidth=2.5, zorder=3,
    ))
    # white content area
    pad = 0.09
    ax.add_patch(FancyBboxPatch(
        (x + pad, y + pad), w - 2*pad, h - header_h - pad,
        boxstyle="round,pad=0.04",
        facecolor="white", edgecolor="none",
        linewidth=0, zorder=4,
    ))
    # header text
    ax.text(x + w/2, y + h - header_h/2, title,
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color="white", zorder=5)
    # content lines
    for i, line in enumerate(lines):
        bold   = line.startswith("●")
        mono   = line.startswith("  ") and any(c in line for c in ["(", "·", "=", "→"])
        family = "monospace" if mono else "sans-serif"
        size   = 9.0 if not bold else 9.5
        weight = "bold" if bold else "normal"
        color_txt = "#1A252F" if bold else "#2C3E50"
        ax.text(x + 0.22, y + h - header_h - 0.28 - i * 0.37,
                line, ha="left", va="top",
                fontsize=size, fontweight=weight,
                fontfamily=family, color=color_txt, zorder=5)


def draw_arrow(x1, x2, y, label=""):
    ax.annotate("",
        xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                        lw=2.0, mutation_scale=18),
        zorder=6,
    )
    if label:
        mid = (x1 + x2) / 2
        for i, part in enumerate(label.split("\n")):
            ax.text(mid, y + 0.22 - i*0.22, part,
                    ha="center", va="bottom", fontsize=8.5,
                    color="#555", fontstyle="italic", zorder=6)


# ── title ─────────────────────────────────────────────────────────────────────
ax.text(8.5, 7.45, "ShiftKit — Domain Adaptation Framework",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=C["dark"], zorder=4)

# ── module boxes ──────────────────────────────────────────────────────────────
BX = [0.20, 4.45, 8.70, 12.95]
BW, BH, BY = 3.90, 5.50, 1.40

draw_module(BX[0], BY, BW, BH, C["data"], "DATA", [
    "● DataManager",
    "",
    "  Source domain:",
    "    torchvision.MNIST",
    "",
    "  Target domain:",
    "    NoisyMNIST",
    "    (+Gaussian noise σ)",
    "",
    "  Custom datasets via",
    "  DataManager.register()",
])

draw_module(BX[1], BY, BW, BH, C["model"], "MODELS", [
    "● CNN   ● MLP",
    "",
    "  encode(x)",
    "    ↓  latent z ∈ ℝᵈ",
    "  classify(z)",
    "    ↓  logits ŷ",
    "",
    "  Configurable:",
    "    latent_dim",
    "    hidden_dims (MLP)",
    "    dropout",
])

draw_module(BX[2], BY, BW, BH, C["method"], "METHODS", [
    "● MMDTrainer",
    "  loss = CE(ŷ, y_src)",
    "       + λ · MMD²(z_s, z_t)",
    "  RBF mixture kernel",
    "",
    "● SourceOnlyTrainer",
    "  loss = CE(ŷ, y_src)",
    "  (baseline, no DA)",
    "",
    "  Tracks per epoch:",
    "    src_acc,  tgt_acc",
])

draw_module(BX[3], BY, BW, BH, C["diag"], "DIAGNOSTICS", [
    "● plot_latent_space",
    "    t-SNE: by domain",
    "           by class",
    "",
    "● compare_latent_spaces",
    "    Grid: N models × 2 views",
    "",
    "● plot_training_history",
    "    CE loss,  MMD loss",
    "    src & tgt accuracy",
    "    multi-model overlay",
])

# ── arrows ────────────────────────────────────────────────────────────────────
arrow_y = BY + BH / 2 + 0.15
draw_arrow(BX[0]+BW, BX[1],    arrow_y, "loaders")
draw_arrow(BX[1]+BW, BX[2],    arrow_y, "latent z")
draw_arrow(BX[2]+BW, BX[3],    arrow_y, "model +\nhistory")

# ── training loop banner ──────────────────────────────────────────────────────
lx, ly, lw, lh = 0.20, 0.10, 16.60, 1.15
ax.add_patch(FancyBboxPatch(
    (lx, ly), lw, lh,
    boxstyle="round,pad=0.1",
    facecolor=C["code"], edgecolor="none", zorder=3,
))
ax.text(lx + lw/2, ly + lh*0.68,
        "Training loop",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#95A5A6", fontfamily="monospace", zorder=4)
ax.text(lx + lw/2, ly + lh*0.35,
        "for (x_src, y_src), (x_tgt, _) in zip(source_loader, target_loader):\n"
        "    z_src, z_tgt = model.encode(x_src), model.encode(x_tgt)\n"
        "    loss = CrossEntropy(model.classify(z_src), y_src)  +  λ · MMD²(z_src, z_tgt)",
        ha="center", va="center", fontsize=9,
        color="#5DADE2", fontfamily="monospace",
        linespacing=1.7, zorder=4)

# ── save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=C["bg"])
print(f"Saved → {OUT_PATH}")
