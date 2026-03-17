# Tutorial 5 — Tuning the DA weight λ

The `mmd_weight` / `lmmd_weight` / `domain_weight` parameter λ controls the
trade-off between source classification accuracy and domain alignment.

---

## Quick λ sweep

```python
from shiftkit.models  import CNN
from shiftkit.methods import MMDTrainer

results = {}
for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
    model   = CNN(latent_dim=128, num_classes=10)
    trainer = MMDTrainer(model, train_src, train_tgt,
                         mmd_weight=lam, lr=1e-3)
    trainer.fit(epochs=10)
    stats = trainer.evaluate(test_tgt, domain="target-test")
    results[lam] = stats["accuracy"]
    print(f"λ={lam:<5}  target acc={stats['accuracy']*100:.2f}%")
```

Typical output:

```
λ=0.1   target acc=95.41%
λ=0.5   target acc=96.83%
λ=1.0   target acc=97.20%
λ=2.0   target acc=96.95%
λ=5.0   target acc=94.10%
```

---

## Interpreting the results

| Symptom | Likely cause | Remedy |
|---------|-------------|--------|
| Source accuracy drops, target accuracy low | λ too large — DA loss overwhelming classifier | Reduce λ, or increase `warmup_epochs` |
| Source and target latent spaces remain separated | λ too small | Increase λ |
| Both accuracies good on source, poor on target | Fundamental distribution mismatch; λ tuning helps only if distributions are alignable | Try DANN or LMMD |

---

## Using TrainerRegistry for sweeps

Register all methods once, then loop by name:

```python
from shiftkit.models   import CNN
from shiftkit.methods  import TrainerRegistry

sweep = [
    ("mmd",  {"mmd_weight": 0.5}),
    ("mmd",  {"mmd_weight": 1.0}),
    ("lmmd", {"num_classes": 10, "lmmd_weight": 1.0}),
    ("dann", {"domain_weight": 1.0}),
]

for name, kwargs in sweep:
    model   = CNN(latent_dim=128, num_classes=10)
    trainer = TrainerRegistry.create(
        name,
        model=model,
        source_loader=train_src,
        target_loader=train_tgt,
        **kwargs,
    )
    history = trainer.fit(epochs=10)
    result  = trainer.evaluate(test_tgt, domain="target-test")
    print(f"{name} {kwargs}  →  {result['accuracy']*100:.2f}%")
```

---

## Warmup + λ interaction

Using `warmup_epochs` before DA begins means the model has a head start on
source classification. This typically allows a slightly larger λ to be used
without source accuracy collapsing:

```python
trainer = MMDTrainer(
    model=model, source_loader=train_src, target_loader=train_tgt,
    mmd_weight=2.0,      # higher λ is safe because warmup stabilises source acc
    warmup_epochs=5,
    lr=1e-3,
)
trainer.fit(epochs=20)
```

!!! tip
    Start at `λ=1.0` with `warmup_epochs=5`. Only tune λ after confirming
    that source accuracy remains stable.
