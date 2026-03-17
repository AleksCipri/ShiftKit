"""
Example: MMD vs LMMD on MNIST → Noisy MNIST
============================================
Compares global MMD (aligns marginal distributions) against local/class-
conditional LMMD (aligns per-class subdomains) on the MNIST → Noisy MNIST
domain adaptation benchmark.

Run from the repo root:
    python examples/mnist_mmd_lmmd.py

Edit the CONFIG block below to customise the run.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shiftkit.data        import DataManager
from shiftkit.models      import CNN
from shiftkit.methods     import SourceOnlyTrainer, MMDTrainer, LMMDTrainer
from shiftkit.diagnostics import plot_training_history, compare_latent_spaces

# ─── CONFIG ──────────────────────────────────────────────────────────────────

LATENT_DIM    = 128
EPOCHS        = 15
BATCH_SIZE    = 128
LR            = 1e-3
MMD_WEIGHT    = 1.0
LMMD_WEIGHT   = 1.0
NUM_CLASSES   = 10
NOISE_STD     = 0.3
WARMUP        = 3       # warmup epochs (source-only CE) before DA loss kicks in
SAVE_DIR      = os.path.join(os.path.dirname(__file__), "..", "outputs")

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
    print(f"   Noise std            : {NOISE_STD}")

    # ── 2. Build independent models ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Building models (CNN, latent_dim={})".format(LATENT_DIM))
    print("=" * 60)

    def make_model():
        return CNN(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, dropout=0.3)

    model_baseline = make_model()
    model_mmd      = make_model()
    model_lmmd     = make_model()
    n_params = sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)
    print(f"   Parameters : {n_params:,}")

    # ── 3. Source-Only baseline ───────────────────────────────────────────────
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

    # ── 4. Global MMD ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"4. Training with global MMD  (λ={MMD_WEIGHT}, warmup={WARMUP})")
    print("=" * 60)
    mmd_trainer = MMDTrainer(
        model=model_mmd,
        source_loader=train_src,
        target_loader=train_tgt,
        mmd_weight=MMD_WEIGHT,
        lr=LR,
        warmup_epochs=WARMUP,
    )
    history_mmd = mmd_trainer.fit(epochs=EPOCHS)

    # ── 5. Local LMMD ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"5. Training with local LMMD  (λ={LMMD_WEIGHT}, warmup={WARMUP})")
    print("=" * 60)
    lmmd_trainer = LMMDTrainer(
        model=model_lmmd,
        source_loader=train_src,
        target_loader=train_tgt,
        num_classes=NUM_CLASSES,
        lmmd_weight=LMMD_WEIGHT,
        lr=LR,
        warmup_epochs=WARMUP,
    )
    history_lmmd = lmmd_trainer.fit(epochs=EPOCHS)

    # ── 6. Evaluation ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. Evaluation")
    print("=" * 60)
    print(f"\n  {'Domain':<18}  {'Source-Only':>12}  {'MMD':>10}  {'LMMD':>10}")
    print("  " + "-" * 56)
    for loader, domain in [(train_src, "source-train"),
                           (test_src,  "source-test"),
                           (test_tgt,  "target-test")]:
        b = baseline_trainer.evaluate(loader, domain)
        m = mmd_trainer.evaluate(loader, domain)
        l = lmmd_trainer.evaluate(loader, domain)
        print(f"  {domain:<18}  {b['accuracy']*100:>11.2f}%"
              f"  {m['accuracy']*100:>9.2f}%"
              f"  {l['accuracy']*100:>9.2f}%")

    # ── 7. Training history ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. Saving training history plot")
    print("=" * 60)
    plot_training_history(
        histories={
            "Source Only": history_baseline,
            "MMD":         history_mmd,
            "LMMD":        history_lmmd,
        },
        save_path=os.path.join(SAVE_DIR, "mmd_lmmd_history.png"),
        show=False,
    )
    print(f"   Saved → outputs/mmd_lmmd_history.png")

    # ── 8. Latent space comparison ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("8. Generating latent space comparison plot")
    print("=" * 60)
    compare_latent_spaces(
        models={
            "Source Only": model_baseline,
            "MMD":         model_mmd,
            "LMMD":        model_lmmd,
        },
        source_loader=test_src,
        target_loader=test_tgt,
        max_samples=2000,
        save_path=os.path.join(SAVE_DIR, "mmd_lmmd_latent.png"),
        show=False,
    )
    print(f"   Saved → outputs/mmd_lmmd_latent.png")

    print(f"\nDone. All figures saved to '{SAVE_DIR}/'")
