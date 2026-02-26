"""
Example: MNIST (source) -> Noisy MNIST (target)
Compares Source-Only baseline vs MMD domain adaptation.

Run from the repo root:
    python examples/mnist_mmd.py

Edit the CONFIG block below to customise the run.
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shiftkit.data        import DataManager
from shiftkit.models      import CNN, MLP
from shiftkit.methods     import MMDTrainer, SourceOnlyTrainer
from shiftkit.diagnostics import plot_training_history, compare_latent_spaces

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_TYPE  = "cnn"       # "cnn" or "mlp"
LATENT_DIM  = 128
EPOCHS      = 10
BATCH_SIZE  = 128
LR          = 1e-3
MMD_WEIGHT  = 1.0
NOISE_STD   = 0.3
SAVE_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("1. Loading data")
    print("=" * 60)
    dm = DataManager(root="./data", batch_size=BATCH_SIZE, num_workers=0)
    train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True,  noise_std=NOISE_STD)
    test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False, noise_std=NOISE_STD)
    print(f"   Source train batches : {len(train_src)}")
    print(f"   Target train batches : {len(train_tgt)}")

    # ── 2. Build two models (same architecture, different random seeds) ───────
    print("\n" + "=" * 60)
    print("2. Building models")
    print("=" * 60)

    def make_model():
        if MODEL_TYPE == "cnn":
            return CNN(latent_dim=LATENT_DIM, num_classes=10, dropout=0.3)
        return MLP(latent_dim=LATENT_DIM, num_classes=10,
                   hidden_dims=(512, 256), dropout=0.3)

    model_baseline = make_model()
    model_mmd      = make_model()
    n_params = sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)
    print(f"   Architecture : {MODEL_TYPE.upper()}  |  latent_dim={LATENT_DIM}  |  params={n_params:,}")

    # ── 3. Train Source-Only baseline ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Training Source-Only baseline (no domain adaptation)")
    print("=" * 60)
    baseline_trainer = SourceOnlyTrainer(
        model=model_baseline,
        source_loader=train_src,
        target_loader=train_tgt,
        lr=LR,
    )
    history_baseline = baseline_trainer.fit(epochs=EPOCHS)

    # ── 4. Train MMD model ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"4. Training with MMD domain adaptation  (λ={MMD_WEIGHT})")
    print("=" * 60)
    mmd_trainer = MMDTrainer(
        model=model_mmd,
        source_loader=train_src,
        target_loader=train_tgt,
        mmd_weight=MMD_WEIGHT,
        lr=LR,
    )
    history_mmd = mmd_trainer.fit(epochs=EPOCHS)

    # ── 5. Evaluation ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. Evaluation")
    print("=" * 60)
    print(f"\n  {'Domain':<18}  {'Source-Only':>12}  {'MMD':>10}")
    print("  " + "-" * 44)
    for loader, domain in [(train_src, "source-train"),
                           (test_src,  "source-test"),
                           (test_tgt,  "target-test")]:
        b = baseline_trainer.evaluate(loader, domain)
        m = mmd_trainer.evaluate(loader, domain)
        print(f"  {domain:<18}  {b['accuracy']*100:>11.2f}%  {m['accuracy']*100:>9.2f}%")

    # ── 6. Training history comparison ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. Saving training history plot")
    print("=" * 60)
    plot_training_history(
        histories={"Source Only": history_baseline, "MMD": history_mmd},
        save_path=os.path.join(SAVE_DIR, "training_history.png"),
        show=False,
    )

    # ── 7. Latent space comparison ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. Generating latent space comparison plot")
    print("=" * 60)
    compare_latent_spaces(
        models={"Source Only": model_baseline, "MMD": model_mmd},
        source_loader=test_src,
        target_loader=test_tgt,
        max_samples=2000,
        save_path=os.path.join(SAVE_DIR, "latent_space_comparison.png"),
        show=False,
    )

    print(f"\nDone. Figures saved to '{SAVE_DIR}/'")
