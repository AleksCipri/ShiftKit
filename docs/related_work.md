# Related Work

ShiftKit builds on foundational methods in domain adaptation and statistical testing.
Each entry includes a plain-text citation with a link to the official paper, followed by a BibTeX entry.

---

## Maximum Mean Discrepancy (MMD)

Used by: [MMDTrainer](methods/mmd.md), [LMMDTrainer](methods/lmmd.md)

Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
[A Kernel Two-Sample Test](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html).
*Journal of Machine Learning Research*, 13, 723–773.

```bibtex
@article{gretton2012kernel,
  title   = {A Kernel Two-Sample Test},
  author  = {Gretton, Arthur and Borgwardt, Karsten M. and Rasch, Malte J.
             and Sch{\"o}lkopf, Bernhard and Smola, Alexander},
  journal = {Journal of Machine Learning Research},
  volume  = {13},
  pages   = {723--773},
  year    = {2012}
}
```

---

## Domain-Adversarial Neural Networks (DANN)

Used by: [DANNTrainer](methods/dann.md)

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016).
[Domain-Adversarial Training of Neural Networks](https://jmlr.org/papers/volume17/15-239/15-239.pdf).
*Journal of Machine Learning Research*, 17(59), 1–35.

```bibtex
@article{ganin2016dann,
  title   = {Domain-Adversarial Training of Neural Networks},
  author  = {Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and
             Germain, Pascal and Larochelle, Hugo and Laviolette, Fran{\c{c}}ois
             and Marchand, Mario and Lempitsky, Victor},
  journal = {Journal of Machine Learning Research},
  volume  = {17},
  number  = {59},
  pages   = {1--35},
  year    = {2016}
}
```

---

## Deep Subdomain Adaptation (LMMD)

Used by: [LMMDTrainer](methods/lmmd.md)

Zhu, Y., Zhuang, F., Wang, J., Ke, G., Chen, J., Bian, J., Xiong, H., & He, Q. (2021).
[Deep Subdomain Adaptation Network for Image Classification](https://ieeexplore.ieee.org/document/9085896).
*IEEE Transactions on Neural Networks and Learning Systems*, 32(4), 1713–1722.
DOI: 10.1109/TNNLS.2020.2988928

```bibtex
@article{zhu2021lmmd,
  title   = {Deep Subdomain Adaptation Network for Image Classification},
  author  = {Zhu, Yongchun and Zhuang, Fuzhen and Wang, Jindong and Ke, Guolin
             and Chen, Jingwu and Bian, Jiang and Xiong, Hui and He, Qing},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume  = {32},
  number  = {4},
  pages   = {1713--1722},
  year    = {2021},
  doi     = {10.1109/TNNLS.2020.2988928}
}
```

---

## Deep CORAL

Used by: [CORALTrainer](methods/coral.md)

Sun, B., & Saenko, K. (2016).
[Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719).
*ECCV Workshops 2016*, LNCS 9915, 443–450.

```bibtex
@inproceedings{sun2016coral,
  title     = {Deep {CORAL}: Correlation Alignment for Deep Domain Adaptation},
  author    = {Sun, Baochen and Saenko, Kate},
  booktitle = {ECCV Workshops},
  series    = {LNCS},
  volume    = {9915},
  pages     = {443--450},
  year      = {2016}
}
```

---

## SIDDA — SInkhorn Dynamic Domain Adaptation

Used by: [SIDDATrainer](methods/sidda.md)

Pandya, S., Patel, P., Nord, B. D., Walmsley, M., & Ćiprijanović, A. (2025).
[SIDDA: SInkhorn Dynamic Domain Adaptation for image classification with equivariant neural networks](https://iopscience.iop.org/article/10.1088/2632-2153/adf701).
*Machine Learning: Science and Technology*, 6(3), 035032.
DOI: 10.1088/2632-2153/adf701

```bibtex
@article{2025MLS&T...6c5032P,
  author  = {{Pandya}, Sneh and {Patel}, Purvik and {Nord}, Brian D. and
             {Walmsley}, Mike and {\'C}iprijanovi{\'c}, Aleksandra},
  title   = {{SIDDA}: {SI}nkhorn Dynamic Domain Adaptation for image classification
             with equivariant neural networks},
  journal = {Machine Learning: Science and Technology},
  year    = {2025},
  month   = sep,
  volume  = {6},
  number  = {3},
  eid     = {035032},
  pages   = {035032},
  doi     = {10.1088/2632-2153/adf701},
  eprint  = {2501.14048},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}
```

---

## KLIEP — Kullback–Leibler Importance Estimation Procedure

Used by: [KLIEPTrainer](methods/kliep.md)

Sugiyama, M., Nakajima, S., Kashima, H., Buenau, P., & Kawanabe, M. (2007).
[Direct Importance Estimation with Model Selection and Its Application to Covariate Shift Adaptation](https://proceedings.neurips.cc/paper_files/paper/2007/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper.pdf).
*Advances in Neural Information Processing Systems*, 20.

```bibtex
@inproceedings{NIPS2007_be83ab3e,
  author    = {Sugiyama, Masashi and Nakajima, Shinichi and Kashima, Hisashi and
               Buenau, Paul and Kawanabe, Motoaki},
  booktitle = {Advances in Neural Information Processing Systems},
  editor    = {J. Platt and D. Koller and Y. Singer and S. Roweis},
  publisher = {Curran Associates, Inc.},
  title     = {Direct Importance Estimation with Model Selection and Its
               Application to Covariate Shift Adaptation},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2007/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper.pdf},
  volume    = {20},
  year      = {2007}
}
```

---

## Semantic Centroid Alignment (MSTN)

Used by: [DANNTrainer](methods/dann.md) (optional `semantic_weight` parameter)

Xie, S., Zheng, Z., Chen, L., & Chen, C. (2018).
[Learning Semantic Representations for Unsupervised Domain Adaptation](https://proceedings.mlr.press/v80/xie18c.html).
*Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:5423–5432.

```bibtex
@inproceedings{xie2018mstn,
  title     = {Learning Semantic Representations for Unsupervised Domain Adaptation},
  author    = {Xie, Shaoan and Zheng, Zibin and Chen, Liang and Chen, Chuan},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {5423--5432},
  series    = {PMLR},
  volume    = {80},
  year      = {2018}
}
```
