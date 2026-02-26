# ShiftKit
Test repo for scientific domain adaptation solutions

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
- `plot_training_history`: loss curves + source accuracy

## To run

```bash
pip install -r requirements.txt
python examples/mnist_mmd.py
```

Outputs saved to `./outputs/`: `latent_space.png` and `training_history.png`. The `CONFIG` block at the top of the example lets you toggle between CNN/MLP, adjust `mmd_weight`, noise level, etc.
