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
| [scikit-learn](https://scikit-learn.org) | ≥ 1.2 | t-SNE/Isomap projections, metrics, California Housing loader |
| [matplotlib](https://matplotlib.org) | ≥ 3.7 | All plotting and diagnostics |
| [tqdm](https://tqdm.github.io) | ≥ 4.65 | Training progress bars |

**Optional dependencies** (required only for specific features; ShiftKit imports without them):

| Library | Version | Use |
|---------|---------|-----|
| [torch-geometric](https://pyg.org) | ≥ 2.4 | GNN models and PyG graph data (`shiftkit.models.GNN`, `"pyg_domains"`) |
| [geomloss](https://www.kernel-operations.io/geomloss/) | ≥ 0.2.6 | Sinkhorn divergence for `SIDDATrainer` |
| [umap-learn](https://umap-learn.readthedocs.io) | ≥ 0.5 | UMAP projection in diagnostics |

The versions above are minimums, matching `requirements.txt`; no upper bounds are
enforced. The combination most recently exercised locally (CPU) was torch 2.12,
NumPy 2.4, scikit-learn 1.9, matplotlib 3.11, torch-geometric 2.8.1 and geomloss 0.3.1;
other combinations are untested. Record exact versions (e.g. `pip freeze`) alongside
any results you publish.

Documentation built with [MkDocs](https://www.mkdocs.org) (1.x) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme (9.x); the deployment workflow pins these major versions. Equations are rendered with [MathJax 3](https://www.mathjax.org).

---

## Licence

ShiftKit is released under the [Apache License 2.0](https://github.com/AleksCipri/ShiftKit/blob/main/LICENSE).
