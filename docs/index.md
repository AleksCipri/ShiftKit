<p align="center">
  <img src="logo_fox.png" alt="ShiftKit" width="340"/>
</p>

A lightweight, modular **domain adaptation** framework for PyTorch. Transfer knowledge from a labelled *source* domain to an unlabelled *target* domain using deep latent-space alignment.

![Framework Overview](assets/framework.png)

---

## Installation

```bash
git clone https://github.com/AleksCipri/ShiftKit.git
cd ShiftKit
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`, `tqdm`,
`torch-geometric` (GNN models and `"pyg_domains"` data; listed in `requirements.txt`)

**Optional:** `geomloss` (`SIDDATrainer`), `umap-learn` (UMAP projections). The code imports
and trains on tensor data even if `torch-geometric` is not installed (only the GNN features
are then unavailable); see [Credits](credits.md) for versions.

---

## Quick start

```python
from shiftkit.data        import DataManager
from shiftkit.models      import CNN
from shiftkit.methods     import MMDTrainer, SourceOnlyTrainer
from shiftkit.diagnostics import compare_latent_spaces, plot_training_history

from shiftkit.utils import set_seed
set_seed(0)   # record the seed with your results; makes runs repeatable on the same hardware

# 1. Load data (MNIST is downloaded to ./data on first use)
dm = DataManager(batch_size=128)
train_src, train_tgt = dm.load("mnist_noisy_mnist", train=True)
test_src,  test_tgt  = dm.load("mnist_noisy_mnist", train=False)

# 2. Build models
model_baseline = CNN(latent_dim=128, num_classes=10)
model_mmd      = CNN(latent_dim=128, num_classes=10)

# 3. Train — baseline vs MMD domain adaptation
#    (target images only are used for adaptation; target labels, if present,
#     are used only to monitor target accuracy)
baseline = SourceOnlyTrainer(model_baseline, train_src, train_tgt)
mmd      = MMDTrainer(model_mmd, train_src, train_tgt, mmd_weight=1.0)

history_baseline = baseline.fit(epochs=10)
history_mmd      = mmd.fit(epochs=10)

# 4. Evaluate on held-out target data and visualise
for name, trainer in [("Source Only", baseline), ("MMD", mmd)]:
    print(name, trainer.evaluate(test_tgt, domain="target-test")["accuracy"])
plot_training_history({"Source Only": history_baseline, "MMD": history_mmd})
compare_latent_spaces({"Source Only": model_baseline, "MMD": model_mmd},
                      test_src, test_tgt)
```

A single run compares two models trained with one seed; repeat over several seeds
before concluding that one method is better. Choose hyperparameters such as
`mmd_weight` on validation data, not on the target test set (see
[Tuning λ](tutorials/tuning.md)). Latent-space plots are exploratory diagnostics,
not proof of successful adaptation.
