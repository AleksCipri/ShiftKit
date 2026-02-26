"""
Example: MNIST (source) -> Noisy MNIST (target) with MMD domain adaptation.

Run from the repo root:
    python examples/mnist_mmd.py

Optional flags (edit the CONFIG block below to customise the run).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shiftkit.data        import DataManager
from shiftkit.models      import CNN, MLP
from shiftkit.methods     import MMDTrainer
from shiftkit.diagnostics import plot_latent_space, plot_training_history

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_TYPE  = "cnn"       # "cnn" or "mlp"
LATENT_DIM  = 128
EPOCHS      = 10
BATCH_SIZE  = 128
LR          = 1e-3
MMD_WEIGHT  = 1.0         # λ: how strongly to penalise domain shift
NOISE_STD   = 0.3         # Gaussian noise added to target domain
SAVE_DIR    = "./outputs"  # figures saved here

# ─── SETUP ───────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Data
print("=" * 60)
print("1. Loading data")
print("=" * 60)
dm = DataManager(root="./data", batch_size=BATCH_SIZE, num_workers=2)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True,  noise_std=NOISE_STD)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False, noise_std=NOISE_STD)

print(f"   Source train batches : {len(train_src)}")
print(f"   Target train batches : {len(train_tgt)}")

# 2. Model
print("\n" + "=" * 60)
print("2. Building model")
print("=" * 60)
if MODEL_TYPE == "cnn":
    model = CNN(latent_dim=LATENT_DIM, num_classes=10, dropout=0.3)
else:
    model = MLP(latent_dim=LATENT_DIM, num_classes=10,
                hidden_dims=(512, 256), dropout=0.3)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Model     : {MODEL_TYPE.upper()}  |  latent_dim={LATENT_DIM}")
print(f"   Parameters: {n_params:,}")

# 3. DA Training
print("\n" + "=" * 60)
print("3. Training with MMD domain adaptation")
print(f"   mmd_weight={MMD_WEIGHT}  |  epochs={EPOCHS}  |  lr={LR}")
print("=" * 60)
trainer = MMDTrainer(
    model=model,
    source_loader=train_src,
    target_loader=train_tgt,
    mmd_weight=MMD_WEIGHT,
    lr=LR,
)
history = trainer.fit(epochs=EPOCHS)

# Save training curves
plot_training_history(
    history,
    save_path=os.path.join(SAVE_DIR, "training_history.png"),
    show=False,
)

# 4. Evaluate
print("\n" + "=" * 60)
print("4. Evaluation")
print("=" * 60)
src_train_stats = trainer.evaluate(train_src, domain="source-train")
src_test_stats  = trainer.evaluate(test_src,  domain="source-test")
tgt_test_stats  = trainer.evaluate(test_tgt,  domain="target-test")

for s in [src_train_stats, src_test_stats, tgt_test_stats]:
    print(f"   [{s['domain']:>16}]  acc={s['accuracy']*100:.2f}%  "
          f"(n={s['n_samples']:,})")

# 5. Latent space plot
print("\n" + "=" * 60)
print("5. Generating latent space diagnostic plot")
print("=" * 60)
plot_latent_space(
    model=model,
    source_loader=test_src,
    target_loader=test_tgt,
    max_samples=2000,
    title=f"{MODEL_TYPE.upper()} + MMD  (noise_std={NOISE_STD}, λ={MMD_WEIGHT})",
    save_path=os.path.join(SAVE_DIR, "latent_space.png"),
    show=True,
)

print(f"\nDone. Figures saved to '{SAVE_DIR}/'")
