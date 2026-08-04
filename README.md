# FMS-KAN for Well-Logging Knowledge Discovery

**Feature-guided Multi-Scale Kolmogorov-Arnold Network for interpretable prediction of in-situ stress and reservoir parameters from well logs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

`FMS-KAN` predicts three engineering-critical logging targets — **maximum horizontal principal stress (SHMAX)**, **total organic carbon (TOC)**, and **permeability (PERM)** — directly from wireline logs, while remaining **interpretable**: unlike black-box models, a trained FMS-KAN can be symbolically simplified into explicit analytical formulas.

The method extends the standard Kolmogorov-Arnold Network (KAN) with two designs:

1. **Multi-scale B-spline adaptation** — coarse (*G*=5), medium (*G*=10) and fine (*G*=20) grids are combined on every edge, capturing formation-level trends, reservoir-level variations and thin-bed anomalies simultaneously.
2. **XGBoost-SHAP feature guidance** — grid density on each input edge is allocated by feature importance, focusing network capacity on physically dominant curves.

## Key Results

Pooled 70/15/15 train/validation/test split, test-set R²:

| Target | Linear | Poly3 | RF | MLP | Std. KAN | **FMS-KAN** |
|--------|:------:|:-----:|:--:|:---:|:--------:|:-----------:|
| SHMAX  | 0.858  | 0.922 | **0.934** | 0.925 | 0.931 | 0.933 |
| TOC    | 0.850  | 0.947 | 0.970 | 0.971 | 0.939 | **0.976** |
| PERM   | < 0    | 0.875 | 0.976 | 0.943 | 0.964 | **0.987** |
| **Mean** | — | 0.915 | 0.960 | 0.946 | 0.945 | **0.965** |

FMS-KAN **surpasses all black-box baselines on TOC and PERM**, matches the strongest black box (RF) on SHMAX, and improves over the standard KAN by **+0.020** on average — while being the only model that yields explicit formulas.

## Repository Structure

```
FMS-KAN-well-logging/
├── README.md
├── requirements.txt
├── code/
│   ├── build_dataset.py       # parse raw logs → clean feature/target tables
│   ├── train_pooled.py        # pooled train/val/test, 7-way model comparison
│   ├── finalize_pipeline.py   # train final FMS-KAN, export predictions & weights
│   ├── regenerate_v2.py       # reproduce all paper figures
│   └── ...                    # ablation / optimization / formula-extraction scripts
└── data/
    ├── WellA.csv  WellB.csv  WellC.csv  WellD.csv
    └── (12 logging features + SHMAX/TOC/PERM targets)
```

## Data Availability

> ⚠️ **The data in this repository is a de-identified, randomly sampled 50% subset**, released with the data provider's authorization. Well names are anonymized (Well A–D). The full dataset is not publicly available due to data-provider restrictions. The released subset is sufficient to reproduce the modeling pipeline and qualitative results.

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 7-way pooled comparison (Linear / Poly / RF / GBDT / MLP / KAN)
python code/train_pooled.py

# Train final FMS-KAN and export predictions + weights
python code/finalize_pipeline.py

# Reproduce paper figures
python code/regenerate_v2.py
```

## Citation

```bibtex
@article{gong_fmskan_2026,
  title   = {Feature-guided Multi-Scale Kolmogorov-Arnold Network for
             Knowledge Discovery of In-situ Stress and Reservoir Parameters},
  author  = {Gong, An and Qi, Zhenpeng and Zhang, Yongan and Li, Yizheng
             and Sun, Youzhuang and Liu, Mingyu},
  journal = {[under review]},
  year    = {2026}
}
```

## Authors

- **An Gong**¹, **Zhenpeng Qi**¹, **Yongan Zhang**²,³ (corresponding), **Yizheng Li**²,³, **Youzhuang Sun**¹, **Mingyu Liu**⁴

¹ College of Computer Science and Technology, China University of Petroleum (East China), Qingdao, China
² State Key Laboratory of Climate Resilience for Coastal Cities, Department of Building Environment and Energy Engineering, The Hong Kong Polytechnic University, Hong Kong SAR, China
³ Zhejiang Key Laboratory of Industrial Intelligence and Digital Twin, Eastern Institute of Technology, Ningbo, China
⁴ Department of Physics, Colorado State University, Fort Collins, CO, USA

**Corresponding author:** Yongan Zhang (yongan.zhang@connect.polyu.hk)

## License

Released under the MIT License — see [LICENSE](LICENSE).
