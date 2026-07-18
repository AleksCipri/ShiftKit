# Credits

## Author

**Aleksandra Ciprijanovic**
GitHub: [@AleksCipri](https://github.com/AleksCipri)

ShiftKit is a modular, science-focused domain adaptation framework for PyTorch, built around clean interfaces for fast implementation of different deep learning architectures, datasets, and domain adaptation methods.

---

## Other Contributors

**Karla Tame-Narvaez**
GitHub: [@karlaTame](https://github.com/karlaTame)

**Abdelrahman Helal**
GitHub: [@abdelrahman-helal](https://github.com/abdelrahman-helal)

Added PyTorch Geometric support: GNN model, node-level and graph-level domain adaptation, PyG data utilities, and the node-level MMD example.

---

## Dependencies

ShiftKit builds on the following open-source libraries:

| Library | Version | Use |
|---------|---------|-----|
| [PyTorch](https://pytorch.org) | ≥ 2.0 | Deep learning backend |
| [torchvision](https://pytorch.org/vision) | ≥ 0.15 | Built-in datasets (MNIST) |
| [NumPy](https://numpy.org) | ≥ 1.24 | Numerical operations |
| [scikit-learn](https://scikit-learn.org) | ≥ 1.2 | t-SNE for latent space visualisation |
| [matplotlib](https://matplotlib.org) | ≥ 3.7 | All plotting and diagnostics |
| [tqdm](https://tqdm.github.io) | ≥ 4.65 | Training progress bars |

**Optional dependencies** (required only for specific methods):

| Library | Version | Use |
|---------|---------|-----|
| [torch-geometric](https://pyg.org) | ≥ 2.0 | GNN models and PyG graph data (`shiftkit.models.GNN`) |
| [geomloss](https://www.kernel-operations.io/geomloss/) | ≥ 0.2 | Sinkhorn divergence for `SIDDATrainer` |

Documentation built with [MkDocs](https://www.mkdocs.org) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

---

## Licence

ShiftKit is released under the [Apache License 2.0](https://github.com/AleksCipri/ShiftKit/blob/main/LICENSE).
