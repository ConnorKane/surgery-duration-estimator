# Surgery-Duration Model v1 — Decisions Log

_Distributional model: output is a full duration CDF, not a point estimate. The scheduler picks a per-case quantile downstream (newsvendor). This log records every non-trivial modeling choice._

### Feature set for v1
- **Choice:** Use all 42 non-placeholder canonical features; drop the 4 all-null v2 placeholders (op_hour, op_dow, surgeon_id, surgeon_case_count_for_this_procedure).
- **Alternatives considered:** Feed placeholders as all-NaN columns; or one-hot the categoricals.
- **Why:** The placeholders (time-of-day, surgeon) are 100% null in this anonymized release — zero signal and NGBoost rejects all-NaN columns. They stay in the schema/serving contract for v2 but are excluded from the fitted matrix. Categoricals stay as integer codes for LightGBM's native handling (one-hot would explode the 68-level opname and hurt trees).

- Train 4412 | Val 945 | Test 946 | 42 features.

### Target scale for the quantile models
- **Choice:** Train each LightGBM quantile model on log1p(duration); expm1 predictions back to minutes.
- **Alternatives considered:** Train on raw minutes.
- **Why:** Duration is strongly right-skewed (median 110, max 955 min). Quantiles are equivariant under the monotone log1p map, so expm1 of a log-scale quantile IS the duration quantile — no bias introduced. Log scale balances the split gains across magnitudes so short cases aren't drowned out by the long tail, and keeps relative (multiplicative) errors sensible, which matches how OR durations actually vary.

### Quantile grid
- **Choice:** K=11 levels: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95].
- **Alternatives considered:** Denser grid (e.g. every 0.05); or fewer (5 levels).
- **Why:** 11 levels give the scheduler fine control across the body AND the tails (0.05/0.95 anchor interval coverage at the 90% level exactly on grid points) while keeping 11 independent fits cheap and low-variance on ~4.4k train rows. A denser grid raises crossing/variance risk with little scheduling benefit at v1; 5 levels is too coarse to reconstruct a smooth CDF.

### LightGBM hyperparameters + early stopping
- **Choice:** learning_rate=0.05, num_leaves=31, min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8/freq=1, max n_estimators=2000 with early stopping (100 rounds) on VAL pinball loss, per-quantile.
- **Alternatives considered:** Heavier per-quantile grid search; or library defaults with no early stop.
- **Why:** This is v1: sensible regularization for a small (~4.4k) tabular set — shallow leaves, min_child_samples to avoid overfitting sparse strata, subsampling for variance reduction. Early stopping on each quantile's own val pinball loss is the principled per-model selector and costs nothing. A full grid search is deferred to v2 (the brief says don't tune to death).

- LightGBM best iterations per quantile: {0.05: 123, 0.1: 150, 0.2: 455, 0.3: 256, 0.4: 295, 0.5: 183, 0.6: 120, 0.7: 139, 0.8: 122, 0.9: 94, 0.95: 98}

### CDF reconstruction from K quantiles
- **Choice:** Monotone rearrangement (sort the K predicted quantiles per row) + piecewise-linear interpolation of the quantile function; linear tail extrapolation clipped to [1, 1440] min.
- **Alternatives considered:** Isotonic regression on quantiles; spline fit; parametric refit.
- **Why:** Independent quantile models can cross; sorting (Chernozhukov et al. rearrangement) is the minimal, distribution-free fix that provably reduces quantile loss and guarantees a valid non-decreasing CDF. Piecewise-linear interpolation is transparent and cheap for a scheduler that mostly reads quantiles. Isotonic gives the same ordering here with more machinery; a parametric refit would re-impose shape assumptions we deliberately avoided by going nonparametric. Implemented once in predict.CDF and reused in eval.

### NGBoost comparison configuration
- **Choice:** NGBRegressor(Dist=LogNormal, Score=LogScore, n_estimators up to 800, lr=0.03) with val-based early stopping; integer category codes treated as numeric ordinals.
- **Alternatives considered:** Normal head; one-hot categoricals for NGBoost; skip the comparison.
- **Why:** A LogNormal head is the natural parametric match for a positive, right-skewed duration and gives a clean sanity baseline for whether the LGB quantile ensemble leaves obvious distributional signal on the table. NGBoost has no native categorical support; for a v1 sanity check we pass the integer codes as ordinals (a known limitation, not the production model) rather than expanding a 68-level one-hot. Early stopping on val avoids over-boosting.

---

## Design-decision index (short form)
1. Features: 42 cols; drop 4 all-null v2 placeholders.
2. Target: train on log1p(duration), expm1 back (quantile-equivariant).
3. Quantile grid: K=11 [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95].
4. LGB: lr .05, leaves 31, min_child 40, subsample .8, early-stop 100 on val pinball.
5. CDF: monotone rearrangement (sort) + piecewise-linear interp; reused from predict.py.
6. NGBoost: LogNormal head, lr .03, val early-stopping; codes as ordinals (sanity check).
