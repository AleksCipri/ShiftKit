# Tutorial 5 — Tuning the DA weight λ

The `mmd_weight` / `lmmd_weight` / `domain_weight` parameter λ controls the
trade-off between source classification accuracy and domain alignment.

Tuning λ (or choosing between methods) is **model selection**. The data used
to make that choice must not be the data used to report the final result.

---

## Which data may be used for selection?

| Setting | Target labels available? | Select on | Report on |
|---------|-------------------------|-----------|-----------|
| Unsupervised DA (the usual assumption) | none | source validation set and/or unsupervised criteria (below) | target test set, once |
| Semi-supervised / "small labelled target" | a few, explicitly allowed by your problem | a **separate labelled target validation set** | target test set, once |
| Benchmark study with a target test set | test labels only for reporting | never the target test set | target test set, once |

Use target labels for selection only if your real deployment would also have
them. Otherwise the reported target accuracy overstates what you could
achieve, because the target test set has been used for development.

---

## λ sweep with a labelled target validation set

This assumes a small labelled target set is allowed. Split it off **before**
training, select λ on it, and evaluate the chosen model on the target test
set once.

```python
import torch
from torch.utils.data import DataLoader, Subset
from shiftkit.data    import DataManager
from shiftkit.models  import CNN
from shiftkit.methods import MMDTrainer

dm = DataManager(root="./data", batch_size=128)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False)

# Hold out 1,000 target *training* images as a labelled validation set.
# They are removed from the unlabelled target training stream.
tgt_ds = train_tgt.dataset
perm   = torch.randperm(len(tgt_ds), generator=torch.Generator().manual_seed(0))
val_idx, rest_idx = perm[:1000].tolist(), perm[1000:].tolist()
val_tgt   = DataLoader(Subset(tgt_ds, val_idx), batch_size=256)
train_tgt = DataLoader(Subset(tgt_ds, rest_idx), batch_size=128, shuffle=True)

val_acc, models = {}, {}
for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
    torch.manual_seed(0)                       # same initialisation per λ
    model   = CNN(latent_dim=128, num_classes=10)
    trainer = MMDTrainer(model, train_src, train_tgt, mmd_weight=lam, lr=1e-3)
    trainer.fit(epochs=10)
    val_acc[lam] = trainer.evaluate(val_tgt, domain="target-val")["accuracy"]
    models[lam]  = trainer
    print(f"λ={lam:<5}  target-val acc={val_acc[lam]*100:.2f}%")

best = max(val_acc, key=val_acc.get)
final = models[best].evaluate(test_tgt, domain="target-test")   # report once
print(f"selected λ={best}: target-test acc={final['accuracy']*100:.2f}%")
```

No output is shown here: we have not run this sweep for the documentation,
and the numbers depend on seed, epochs and hardware. With a single seed,
differences of a few tenths of a percent between λ values are usually within
run-to-run noise; repeat over several seeds before trusting a ranking.

---

## Without target labels: unsupervised selection

When no target labels are allowed, there is no direct estimate of target
accuracy. Common proxies, and their limits:

| Criterion | What it measures | Limitation |
|-----------|------------------|------------|
| Source validation accuracy | fit on the source task | can be unrelated to target accuracy; favours small λ |
| Domain discrepancy (e.g. MMD) on held-out data | how aligned the latents are | alignment can be achieved while mismatching classes; favours large λ |
| Target prediction entropy / confidence | how confident the model is on target | confident predictions can be confidently wrong |
| Importance-weighted source validation (e.g. using [KLIEP](../methods/kliep.md) weights) | source error reweighted towards the target | assumes covariate shift with overlapping support; high-variance when weights are extreme |
| Reverse validation (train on pseudo-labelled target, test on source) | consistency across domains | expensive; pseudo-label errors propagate |

None of these is guaranteed to pick the λ with the best target accuracy.
Report which criterion you used, fix the λ grid in advance, and prefer
settings that are stable across a range of λ over a single sharp optimum.

---

## Interpreting the results

| Symptom | Likely cause | Remedy |
|---------|-------------|--------|
| Source accuracy drops, target accuracy low | λ too large — DA loss overwhelming classifier | Reduce λ, or increase `warmup_epochs` |
| Source and target latent spaces remain separated | λ too small | Increase λ |
| Both accuracies good on source, poor on target | Fundamental distribution mismatch; λ tuning helps only if distributions are alignable | Try DANN or LMMD |

---

## Using TrainerRegistry for sweeps

Register all methods once, then loop by name. Compare candidates on the
validation loader (`val_tgt` above, or a source validation loader in the
unsupervised setting), not on the test set:

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
    result  = trainer.evaluate(val_tgt, domain="target-val")
    print(f"{name} {kwargs}  →  {result['accuracy']*100:.2f}%")
```

Reporting several **pre-declared** methods on the test set is fine; using
those test results to choose among them is not.

---

## Warmup + λ interaction

Using `warmup_epochs` before DA begins means the model has a head start on
source classification. This typically allows a slightly larger λ to be used
without source accuracy collapsing:

```python
trainer = MMDTrainer(
    model=model, source_loader=train_src, target_loader=train_tgt,
    mmd_weight=2.0,      # higher λ is usually tolerable after warmup
    warmup_epochs=5,
    lr=1e-3,
)
trainer.fit(epochs=20)
```

!!! tip
    Start at `λ=1.0` with `warmup_epochs=5`. Only tune λ after confirming
    that source accuracy remains stable, and tune on validation data.
