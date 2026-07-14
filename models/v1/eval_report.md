# Surgery-Duration Model v1 — Evaluation Report

Distributional model. **Primary metrics are distribution-aware (pinball, CRPS, coverage).** Point metrics below are a humans-need-a-number sanity check only — NOT the model's success criterion.

- Train / Val / Test: 4412 / 945 / 946
- Quantile grid (K=11): [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
- Target: log1p(duration), models expm1'd back; CDF via monotone rearrangement + linear interp.

## 1. Headline (TEST)

| metric | LightGBM (primary) | NGBoost LogNormal |
|---|---|---|
| mean pinball loss | **15.998** | 17.400 |
| CRPS (min) | **32.86** | 34.90 |
| coverage @ 50% (target 0.50) | 0.416 | 0.459 |
| coverage @ 80% (target 0.80) | 0.707 | 0.765 |
| coverage @ 90% (target 0.90) | 0.815 | 0.876 |
| coverage @ 95% (target 0.95) | 0.863 | 0.925 |
| median MAPE | 41.4% | 45.8% |

**Head-to-head:** LightGBM wins on mean pinball (Δ=1.401). NGBoost is the parametric sanity check; a small/negative gap means the LGB ensemble is not leaving obvious distributional signal on the table.

## 2. Sharpness / accuracy

Per-quantile pinball loss (TEST):

| q | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LGB | 4.72 | 8.18 | 13.54 | 17.51 | 20.18 | 21.81 | 22.53 | 22.22 | 19.93 | 14.87 | 10.50 |
| NGB | 4.76 | 8.51 | 14.73 | 19.41 | 22.81 | 24.90 | 25.57 | 24.66 | 21.21 | 14.83 | 10.01 |

## 3. Calibration

- PIT mean (test) = 0.474 (ideal 0.50), PIT std = 0.322 (ideal ~0.29 for uniform).
- Empirical vs nominal central-interval coverage (TEST):

| nominal | LGB empirical | NGB empirical |
|---|---|---|
| 50% | 0.416 | 0.459 |
| 80% | 0.707 | 0.765 |
| 90% | 0.815 | 0.876 |
| 95% | 0.863 | 0.925 |

See `reliability_plot.png` for the full curve.

## 4. Point-summary sanity vs naive baselines (informational)

| predictor | MAE (min) | MAPE |
|---|---|---|
| **LGB predicted median** | 43.6 | 41.4% |
| LGB predicted mean | 46.2 | 51.3% |
| baseline: global_mean | 79.9 | 116.9% |
| baseline: optype_mean | 63.4 | 89.0% |
| baseline: opname_mean | 49.1 | 58.2% |

## 5. Stratified errors (TEST)

Best 5 strata (lowest pinball):

| type | value | n | pinball | cov80 |
|---|---|---|---|---|
| optype | Breast | 56 | 10.11 | 0.80 |
| department | Urology | 15 | 11.66 | 0.60 |
| optype | Colorectal | 205 | 11.74 | 0.68 |
| optype | Biliary/Pancreas | 118 | 12.82 | 0.75 |
| optype | Thyroid | 47 | 13.43 | 0.70 |

Worst 5 strata (highest pinball):

| type | value | n | pinball | cov80 |
|---|---|---|---|---|
| department | Thoracic surgery | 170 | 22.25 | 0.70 |
| department | Gynecology | 36 | 22.54 | 0.69 |
| optype | Hepatic | 44 | 22.66 | 0.70 |
| optype | Others | 100 | 24.03 | 0.67 |
| asa_bucket | ASA 4-5 | 9 | 32.65 | 0.67 |

**Miscalibrated test strata (80% coverage <0.70 or >0.90):** 10.

| type | value | n | cov80 |
|---|---|---|---|
| department | Gynecology | 36 | 0.69 |
| department | Urology | 15 | 0.60 |
| optype | Colorectal | 205 | 0.68 |
| optype | Minor resection | 82 | 0.66 |
| optype | Others | 100 | 0.67 |
| optype | Stomach | 101 | 0.69 |
| optype | Transplantation | 62 | 0.65 |
| emergency | emergency | 112 | 0.69 |
| asa_bucket | ASA 3 | 100 | 0.67 |
| asa_bucket | ASA 4-5 | 9 | 0.67 |

Full per-slice table: `stratified_errors.csv`.

## 6. Interpretation

- The model emits a full CDF per case; the scheduler selects a quantile from live OR state (newsvendor). Coverage near nominal at the 80/90% levels is what makes those quantile picks trustworthy.
- **Key calibration finding:** the 80% central interval empirically covers only **0.707** (target 0.80) — the predicted intervals are systematically TOO NARROW. Per-quantile diagnostics show the lower quantiles are biased high (e.g. the nominal-P10 line sits above ~15% of actuals) and the upper tail slightly low, pulling both interval edges inward. Pinball-optimal quantiles do not guarantee marginal coverage, especially with a skewed target, a log transform, and sparse tails on ~4.4k rows. **v2 fix:** a post-hoc recalibration layer (split-conformal or isotonic quantile recalibration on val) — cheap and interface-preserving.
- Point MAPE beating the per-optype / per-opname baselines confirms the features add signal beyond 'procedure-average duration'.
- NGBoost (LogNormal) is slightly *better calibrated* at 80% but worse on pinball/CRPS — its parametric shape spreads mass more honestly while the LGB ensemble is sharper. This is the expected sharpness/calibration trade; v2 recalibration should let LGB keep its sharpness AND fix coverage.
- Flagged strata are the calibration hotspots to revisit in v2 (likely small-n or heavy-tail slices, e.g. ASA 4-5 and emergencies).
