# ShiftKit
Test repo for scientific domain adaptation solutions

## Overview

![ShiftKit Framework](outputs/framework_overview.png)

## Framework structure

```
shiftkit/
├── data/datasets.py       # DataManager + NoisyMNIST
├── models/networks.py     # MLP + CNN (encoder/classify split)
├── methods/mmd.py         # MMDLoss + MMDTrainer
└── diagnostics/plots.py   # plot_latent_space + plot_training_history
examples/
└── mnist_mmd.py           # end-to-end demo
```

## How each module works

**Data** — `shiftkit/data/datasets.py`
- `DataManager.load("mnist_noisy_mnist")` returns `(source_loader, target_loader)`
- New dataset pairs can be registered via `DataManager.register(name, factory_fn)`

**Models** — `shiftkit/models/networks.py`
- Both `MLP` and `CNN` expose `.encode(x)` → latent vector and `.classify(z)` → logits
- This split is what lets DA methods operate in the latent space

**MMD** — `shiftkit/methods/mmd.py`
- `MMDLoss`: unbiased MMD² with a mixture of RBF kernels (captures structure at multiple scales)
- `MMDTrainer.fit(epochs)`: trains with `total_loss = CrossEntropy(source) + λ·MMD²(z_src, z_tgt)`
- Returns a `history` list for plotting

**Diagnostics** — `shiftkit/diagnostics/plots.py`
- `plot_latent_space`: t-SNE reduction → two panels (by domain / by class label)
- `compare_latent_spaces`: side-by-side grid comparing multiple models
- `plot_training_history`: CE loss + source & target accuracy, supports multi-model overlay

## To run

```bash
pip install -r requirements.txt
python examples/mnist_mmd.py
```

Outputs saved to `./outputs/`:
- `training_history.png` — CE loss and source/target accuracy curves for both models
- `latent_space_comparison.png` — 2×2 t-SNE grid comparing Source-Only vs MMD latent spaces

The `CONFIG` block at the top of the example lets you toggle between CNN/MLP, adjust `mmd_weight`, noise level, epochs, etc.
