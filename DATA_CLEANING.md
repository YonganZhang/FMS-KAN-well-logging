# Data Cleaning & Released-Subset Protocol

This document records the **objective, model-independent** quality control applied to the raw wireline logs before modeling, and the de-identification of the released subset. Every rule below was fixed **prior to** modeling and does not depend on any model's residuals (i.e. no result-driven cherry-picking).

## 1. Quality Control (applied to the restricted raw logs)

1. **Sentinel removal.** Tool sentinel values (`-9999`, and values `>= 99990`) are replaced with `NaN`. Depth samples lacking valid core acoustic/density curves (DTC, DTS, DEN) are dropped.
2. **Physical-range filtering.** Each target is bounded by its physical admissible range (e.g. SHMAX `> 0`, TOC `>= 0`, PERM `>= 0`); feature outliers are removed with an inter-quartile-range (IQR) rule.
3. **Placeholder-artifact removal (SHMAX).** Isolated values far below a well's main distribution — specifically `SHMAX < 0.4 x (per-well median)` — are removed as data-provider placeholders. This affected a short deep interval in **Well C**, where the recorded maximum horizontal stress collapses from ~135 MPa to ~8.5 MPa near 4386 m, which is physically impossible for in-situ stress and continuous depth. Removing this placeholder interval also eliminated an artifact-driven sensitivity of the random train/test split.

## 2. Released Subset (this repository)

- **Random 50% subsample** per well (`random_state = 42`), **de-identified** (wells relabeled Well A–D).
- Released **solely for reproduction** under the data provider's authorization; the full, depth-continuous dataset remains restricted and is not distributed here.
- Row counts of the released subset:

  | Well | Rows |
  |------|-----:|
  | Well A | 1,346 |
  | Well B | 2,106 |
  | Well C | 1,788 |
  | Well D | 572 |
  | **Total** | **5,812** |

## 3. Scope

The released subset reproduces the modeling pipeline and the study's qualitative results and yields point estimates in the neighborhood of those reported. Because the subsample is random over depth (not contiguous), it is not the exact dataset behind every published figure.
