# Tutorials & Examples

Step-by-step guides covering common ShiftKit workflows.

!!! note "About the numbers in these tutorials"
    Unless a tutorial states otherwise, printed accuracies and figures are
    illustrative output from single runs and are not benchmarks. Where a tutorial
    reports numbers we re-ran (e.g. the node-level example), it says so and gives the
    seed. Results vary across seeds, hardware and library versions.

---

| # | Tutorial | What you'll learn |
|---|----------|-------------------|
| 1 | [MNIST → Noisy MNIST: Source-Only vs MMD vs DANN](mnist_mmd.md) | End-to-end DA experiment comparing three methods on the built-in benchmark |
| 2 | [MMD vs LMMD comparison](mmd_vs_lmmd.md) | Global vs class-conditional alignment and how to compare them |
| 3 | [Swapping the model](swap_model.md) | Using MLP instead of CNN; bring-your-own architecture |
| 4 | [Registering a custom dataset](custom_dataset.md) | Plug in any PyTorch dataset pair without modifying the library |
| 5 | [Tuning the DA weight λ](tuning.md) | Selecting λ on validation data (labelled or unsupervised) and reporting test results once |
| 6 | [Regression with Domain Adaptation](regression.md) | Sine wave phase shift and California Housing geographic split using `MMDRegressionTrainer` |
| 7 | [Node-level DA with PyTorch Geometric](pyg_node_mmd.md) | Graph neural networks, node-level domain adaptation, and PyTorch Geometric integration (optional dependency) |
| 8 | [SIDDA Regression & Latent-Space Reweighting](sidda_regression.md) | `SIDDARegressionTrainer` on 2D blobs; comparing plain SIDDA, CE-reweight, OT-reweight, and both |
| 9 | [Prediction intervals in 5 minutes](uq_intervals.md) | Add intervals to any trained regressor with conformal prediction; read coverage and width |
| 10 | [Uncertainty under domain shift](uq_shift.md) | Why intervals under-cover on the target and how importance weighting repairs it |
| 11 | [Local intervals with LVD](uq_lvd.md) | Intervals that adapt to where the model is more or less certain |
| 12 | [SIDDA on galaxy images](https://github.com/AleksCipri/ShiftKit/blob/main/examples/sidda_galaxies.ipynb) | Notebook with saved results: SIDDA against a source-only baseline on SDSS → DESI, over several seeds |
| 13 | [Calibrating a classifier under shift](https://github.com/AleksCipri/ShiftKit/blob/main/examples/uq_galaxies.ipynb) | Notebook with saved results: temperature scaling and KCal on SDSS → DESI, with and without target labels |

The two notebooks download a subset of the [SIDDA galaxy data](https://zenodo.org/records/14583107) on first use (about 240 MB, a few minutes) and cache it.
