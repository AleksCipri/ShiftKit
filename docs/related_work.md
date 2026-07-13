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

## Deep Adaptation Networks (DAN)

Used by: [MMDTrainer](methods/mmd.md)

Long, M., Cao, Y., Wang, J., & Jordan, M. (2015).
[Learning Transferable Features with Deep Adaptation Networks](https://proceedings.mlr.press/v37/long15.html).
*Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 97–105.

```bibtex
@inproceedings{long2015learning,
  title     = {Learning Transferable Features with Deep Adaptation Networks},
  author    = {Long, Mingsheng and Cao, Yue and Wang, Jianmin and Jordan, Michael},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {97--105},
  year      = {2015}
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

Zhu, Y., Zhuang, F., & Wang, D. (2020).
[Deep Subdomain Adaptation Network for Image Classification](https://arxiv.org/abs/2106.09388).
*IEEE Transactions on Neural Networks and Learning Systems*, 32(4), 1713–1722.

```bibtex
@article{zhu2020lmmd,
  title   = {Deep Subdomain Adaptation Network for Image Classification},
  author  = {Zhu, Yongchun and Zhuang, Fuzhen and Wang, Deqing},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume  = {32},
  number  = {4},
  pages   = {1713--1722},
  year    = {2020}
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

Ciprijanovic, A., Lewis, A., Pedro, K., Downey, E., Nord, B., & Stark, A. (2025).
[SIDDA: SInkhorn Dynamic Domain Adaptation for Image Classification with Equivariant Neural Networks](https://iopscience.iop.org/article/10.1088/2632-2153/adf701).
*Machine Learning: Science and Technology*, 6, 035032.

```bibtex
@article{ciprijanovic2025sidda,
  title   = {{SIDDA}: {SI}nkhorn Dynamic Domain Adaptation for Image Classification
             with Equivariant Neural Networks},
  author  = {Ciprijanovic, Aleksandra and Lewis, Ashia and Pedro, Karolina and
             Downey, Eve and Nord, Brian and Stark, Abigail},
  journal = {Machine Learning: Science and Technology},
  volume  = {6},
  pages   = {035032},
  year    = {2025}
}
```

---

## KLIEP — Kullback–Leibler Importance Estimation Procedure

Used by: [KLIEPTrainer](methods/kliep.md)

Sugiyama, M., Nakajima, S., Kashima, H., Bünau, P. V., & Kawanabe, M. (2008).
[Direct Importance Estimation with Model Selection and Its Application to Covariate Shift Adaptation](https://papers.nips.cc/paper/3248-direct-importance-estimation-with-model-selection-and-its-application-to-covariate-shift-adaptation).
*Advances in Neural Information Processing Systems 20 (NeurIPS)*, 1433–1440.

```bibtex
@inproceedings{sugiyama2008kliep,
  title     = {Direct Importance Estimation with Model Selection and Its
               Application to Covariate Shift Adaptation},
  author    = {Sugiyama, Masashi and Nakajima, Shinichi and Kashima, Hisashi and
               B{\"u}nau, Paul von and Kawanabe, Motoaki},
  booktitle = {Advances in Neural Information Processing Systems 20 (NeurIPS)},
  pages     = {1433--1440},
  year      = {2008}
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
