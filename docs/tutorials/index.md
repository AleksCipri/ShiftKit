# Tutorials & Examples

Step-by-step guides covering common ShiftKit workflows.

---

| # | Tutorial | What you'll learn |
|---|----------|-------------------|
| 1 | [MNIST → Noisy MNIST: Source-Only vs MMD vs DANN](mnist_mmd.md) | End-to-end DA experiment comparing three methods on the built-in benchmark |
| 2 | [MMD vs LMMD comparison](mmd_vs_lmmd.md) | How class-conditional alignment improves over global MMD |
| 3 | [Swapping the model](swap_model.md) | Using MLP instead of CNN; bring-your-own architecture |
| 4 | [Registering a custom dataset](custom_dataset.md) | Plug in any PyTorch dataset pair without modifying the library |
| 5 | [Tuning the DA weight λ](tuning.md) | Systematic λ sweep and how to read the results |
| 6 | [Regression with Domain Adaptation](regression.md) | Sine wave phase shift and California Housing geographic split using `MMDRegressionTrainer` |
| 7 | [Node-level DA with PyTorch Geometric](pyg_node_mmd.md) | Graph neural networks, node-level domain adaptation, and PyTorch Geometric integration |
| 8 | [SIDDA Regression & Latent-Space Reweighting](sidda_regression.md) | `SIDDARegressionTrainer` on 2D blobs; comparing plain SIDDA, CE-reweight, OT-reweight, and both |
