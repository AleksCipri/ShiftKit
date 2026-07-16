# Background

**Domain shift** occurs when the statistical distribution of data changes between the context in which a model is trained (the *source domain*) and the context in which it is deployed (the *target domain*). Even a well-trained model can fail badly when its training distribution does not match what it encounters at inference time — a classifier trained on clean images may degrade on noisy ones, or a model trained on data from one detector may not transfer to another.

**Domain adaptation (DA)** is a family of machine learning techniques that address this problem by explicitly accounting for the gap between source and target distributions. Rather than simply retraining on target data (which is often scarce or unlabelled), DA methods align representations, reweight samples, or align feature statistics so that a model trained on the source generalises better to the target.

For a more detailed introduction to domain adaptation concepts and theory, see the
[Introduction to Domain Adaptation](assets/Intro to Domain Adaptation.pdf) lecture notes.
