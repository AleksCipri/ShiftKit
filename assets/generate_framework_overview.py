"""Generate an abstract, version-stable framework overview figure for ShiftKit docs."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── colours ──────────────────────────────────────────────────────────────────
C_DATA  = "#2E86AB"   # teal
C_MODEL = "#28A745"   # green
C_METH  = "#E85D04"   # orange-red
C_DIAG  = "#7B2D8B"   # purple
C_DARK  = "#1C1C2E"   # near-black (code strip)
C_BG    = "#F8F9FA"   # off-white background
C_PIPE  = "#4A4A6A"   # pipeline arrows / boxes

COLS = [C_DATA, C_MODEL, C_METH, C_DIAG]
HEADERS = ["DATA", "MODELS", "METHODS", "DIAGNOSTICS"]

MODULE_BULLETS = [
    [
        "DataManager",
        "─────────────────",
        "Source domain",
        "(any labelled dataset)",
        "",
        "Target domain",
        "(any unlabelled dataset)",
        "",
        "Register custom datasets",
        "via DataManager.register()",
    ],
    [
        "Pluggable backbone",
        "─────────────────",
        "Encoder  f(x) → z",
        "latent dim: configurable",
        "",
        "Task head  g(z) → ŷ",
        "cls / regression head",
        "",
        "Any PyTorch nn.Module",
        "Drop-in architecture",
    ],
    [
        "DA Trainer API",
        "─────────────────",
        "L = L_task + λ · L_align",
        "",
        "Alignment strategies:",
        "  · distribution matching",
        "  · kernel-based",
        "  · adversarial",
        "",
        "Baseline: source-only",
        "Custom via Trainer base",
    ],
    [
        "Visualisation suite",
        "─────────────────",
        "plot_latent_space()",
        "  t-SNE by domain/class",
        "",
        "compare_latent_spaces()",
        "  N models × 2 views",
        "",
        "plot_training_history()",
        "  loss, acc, multi-model",
    ],
]

CODE = (
    "for x_src, y_src, x_tgt in loader:\n"
    "    z_src, z_tgt = model.encode(x_src), model.encode(x_tgt)\n"
    "    loss = task_loss(model.classify(z_src), y_src) + λ · align_loss(z_src, z_tgt)"
)

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=C_BG)

# regions (in figure-fraction coords)
TITLE_Y   = 0.93
PIPE_Y    = 0.60   # top of pipeline band
GRID_Y    = 0.18   # top of module grid
CODE_Y    = 0.00   # top of code strip

# ── title ─────────────────────────────────────────────────────────────────────
fig.text(
    0.5, TITLE_Y + 0.04,
    "ShiftKit — Domain Adaptation Framework",
    ha="center", va="center",
    fontsize=20, fontweight="bold", color=C_PIPE,
    fontfamily="monospace",
)

# ── pipeline diagram ──────────────────────────────────────────────────────────
pipe_ax = fig.add_axes([0.03, GRID_Y + 0.30, 0.94, 0.30])
pipe_ax.set_xlim(0, 10)
pipe_ax.set_ylim(0, 3.2)
pipe_ax.axis("off")
pipe_ax.set_facecolor(C_BG)


def rbox(ax, x, y, w, h, color, text, fontsize=10, text_color="white", radius=0.25):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            zorder=4, multialignment="center")


def arrow(ax, x0, y0, x1, y1, color=C_PIPE):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
        zorder=2,
    )


# domain boxes
rbox(pipe_ax, 1.1, 2.3, 1.6, 0.65, C_DATA, "Source Domain\n(labelled)", fontsize=9)
rbox(pipe_ax, 1.1, 1.0, 1.6, 0.65, C_DATA, "Target Domain\n(unlabelled)", fontsize=9)

# encoder
rbox(pipe_ax, 3.5, 1.65, 1.7, 0.75, C_MODEL, "Shared\nEncoder", fontsize=10)

# latent z node
circle = plt.Circle((5.35, 1.65), 0.35, color=C_PIPE, zorder=3)
pipe_ax.add_patch(circle)
pipe_ax.text(5.35, 1.65, "z", ha="center", va="center",
             fontsize=13, color="white", fontweight="bold", zorder=4)

# task head
rbox(pipe_ax, 7.1, 2.45, 1.6, 0.65, C_MODEL, "Task Head\n(cls / regr.)", fontsize=9)

# predictions
rbox(pipe_ax, 9.1, 2.45, 1.4, 0.60, "#555577", "Predictions\nŷ", fontsize=9)

# DA alignment
rbox(pipe_ax, 7.1, 0.80, 1.8, 0.75, C_METH,
     "DA Alignment\nL_task + λ·L_align", fontsize=9)

# plug label on DA box
pipe_ax.text(7.1, 0.30, "⟵ pluggable method", ha="center", va="center",
             fontsize=8, color=C_METH, style="italic")

# arrows: domains → encoder
arrow(pipe_ax, 1.9, 2.30, 2.62, 1.90)
arrow(pipe_ax, 1.9, 1.00, 2.62, 1.40)

# encoder → z
arrow(pipe_ax, 4.35, 1.65, 4.98, 1.65)

# z → task head
arrow(pipe_ax, 5.70, 1.85, 6.28, 2.35)

# z → DA loss
arrow(pipe_ax, 5.70, 1.45, 6.28, 1.00)

# task head → predictions
arrow(pipe_ax, 7.90, 2.45, 8.38, 2.45)

# DA loss dashed feedback label
pipe_ax.annotate(
    "", xy=(3.50, 0.80), xytext=(6.20, 0.80),
    arrowprops=dict(arrowstyle="-|>", color=C_METH, lw=1.4,
                    linestyle="dashed"),
    zorder=2,
)
pipe_ax.text(4.85, 0.62, "gradient signal", ha="center", va="center",
             fontsize=8, color=C_METH, style="italic")

# ── module grid ───────────────────────────────────────────────────────────────
for i, (col, header, bullets) in enumerate(zip(COLS, HEADERS, MODULE_BULLETS)):
    x0 = 0.03 + i * 0.245
    ax = fig.add_axes([x0, CODE_Y + 0.09, 0.235, GRID_Y + 0.19])
    ax.set_facecolor(col)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # header bar
    hbar = FancyBboxPatch((0, 0.87), 1, 0.13,
                          boxstyle="square,pad=0", facecolor="white",
                          alpha=0.20, zorder=2)
    ax.add_patch(hbar)
    ax.text(0.5, 0.935, header, ha="center", va="center",
            fontsize=13, fontweight="bold", color="white",
            fontfamily="monospace", zorder=3)

    # bullets
    y = 0.83
    for line in bullets:
        style = "italic" if line.startswith("  ·") else "normal"
        weight = "bold" if line == bullets[0] and i > 0 else "normal"
        alpha = 1.0
        ax.text(0.08, y, line,
                ha="left", va="top",
                fontsize=8.2, color="white",
                style=style, fontweight=weight,
                alpha=alpha, fontfamily="monospace")
        y -= 0.082

# ── code strip ────────────────────────────────────────────────────────────────
code_ax = fig.add_axes([0.03, CODE_Y + 0.01, 0.94, 0.085])
code_ax.set_facecolor(C_DARK)
code_ax.set_xlim(0, 1)
code_ax.set_ylim(0, 1)
code_ax.axis("off")

code_ax.text(0.5, 0.55, CODE,
             ha="center", va="center",
             fontsize=9, color="#A8D8A8",
             fontfamily="monospace")

# ── save ──────────────────────────────────────────────────────────────────────
out = "framework_overview.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
print(f"Saved {out}")
plt.close()
