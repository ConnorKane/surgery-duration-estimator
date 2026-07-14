#!/usr/bin/env python3
"""
Serving module for VitalDB surgery-duration model v1.

FROZEN INTERFACE — v2 will re-implement this without breaking downstream code:

    class CDF:
        quantile(q) -> float        # duration in minutes at probability q
        cdf(t) -> float             # P(duration <= t minutes)
        median() -> float
        mean() -> float
        samples(n, rng=None) -> np.ndarray

    def predict(case_features: dict) -> CDF

The CDF is backed by the K LightGBM quantile models (model_lgb.pkl) with a
monotone-rearranged, piecewise-linear quantile function. `encoders.pkl` (from the
preprocessing stage) supplies the exact winsor caps / train medians / categorical
maps so serving reproduces training-time transforms.

Design choices (full reasoning in decisions_log.md):
  * CDF reconstruction  : monotone rearrangement (sort) of the K predicted
                          quantiles, then piecewise-linear interpolation of the
                          quantile function Q(u); linear tail extrapolation
                          clipped to [MIN_DUR, MAX_DUR].
  * Target scale        : models are trained on log1p(duration); predictions are
                          expm1'd back to minutes before rearrangement.
  * mean()              : trapezoidal integral of Q(u) over u in [0,1].
  * cdf(t)              : numerical inverse of Q (interp on the value axis).
  * Idempotent          : all transforms deterministic; same dict -> same CDF.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

# numpy 2.0 renamed trapz -> trapezoid; prefer the new name, fall back for <2.0.
_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
_MODEL_PATH = _HERE / "model_lgb.pkl"
_ENCODERS_PATH = Path.home() / "Desktop" / "vitaldb_work" / "processed" / "encoders.pkl"

# Physical bounds for a single OR case duration (minutes). Used to clip
# extrapolated tails so the CDF never returns nonsense.
MIN_DUR = 1.0
MAX_DUR = 1440.0  # 24h — same ceiling used to define the label in preprocessing

_BUNDLE = None      # lazy-loaded {models, quantiles, feature_cols, categorical_cols, log_target}
_ENCODERS = None    # lazy-loaded encoders.pkl


# --------------------------------------------------------------------------- #
# CDF object — the frozen serving surface
# --------------------------------------------------------------------------- #
class CDF:
    """Continuous duration distribution reconstructed from K quantile levels.

    Parameters
    ----------
    q_levels : 1-D array of probabilities in (0,1), strictly increasing.
    q_values : 1-D array of duration-minute quantiles, same length. Values are
               monotone-rearranged (sorted) and made strictly increasing on
               construction so all inversions are well-defined.
    """

    __slots__ = ("q_levels", "q_values")

    def __init__(self, q_levels, q_values):
        q_levels = np.asarray(q_levels, dtype=float)
        q_values = np.asarray(q_values, dtype=float)
        # Monotone rearrangement (Chernozhukov et al.): independent quantile
        # models can cross; sorting the predicted values restores a valid,
        # non-decreasing quantile function without changing the marginal set.
        q_values = np.sort(q_values)
        # Guarantee STRICT monotonicity for stable inversion (add tiny ramp).
        q_values = np.maximum.accumulate(q_values)
        eps = 1e-6 * (1.0 + np.arange(len(q_values)))
        q_values = q_values + eps
        self.q_levels = q_levels
        self.q_values = np.clip(q_values, MIN_DUR, MAX_DUR)

    # -- quantile function Q(u) -------------------------------------------- #
    def _quantile_vec(self, u):
        u = np.asarray(u, dtype=float)
        lo_l, hi_l = self.q_levels[0], self.q_levels[-1]
        lo_v, hi_v = self.q_values[0], self.q_values[-1]
        out = np.interp(u, self.q_levels, self.q_values)
        # Linear extrapolation in the tails using the end segments.
        left = u < lo_l
        if left.any():
            s = (self.q_values[1] - self.q_values[0]) / (self.q_levels[1] - self.q_levels[0])
            out = np.where(left, lo_v + s * (u - lo_l), out)
        right = u > hi_l
        if right.any():
            s = (self.q_values[-1] - self.q_values[-2]) / (self.q_levels[-1] - self.q_levels[-2])
            out = np.where(right, hi_v + s * (u - hi_l), out)
        return np.clip(out, MIN_DUR, MAX_DUR)

    def quantile(self, q: float) -> float:
        """Duration (minutes) at probability q in (0,1)."""
        return float(self._quantile_vec(np.array([q]))[0])

    def cdf(self, t: float) -> float:
        """P(duration <= t minutes)."""
        t = float(t)
        # Invert Q: interp on the value axis. q_values is strictly increasing.
        p = float(np.interp(t, self.q_values, self.q_levels,
                            left=np.nan, right=np.nan))
        if np.isnan(p):
            if t <= self.q_values[0]:
                # linear toward 0 below the lowest known quantile
                s = (self.q_levels[1] - self.q_levels[0]) / (self.q_values[1] - self.q_values[0])
                p = self.q_levels[0] + s * (t - self.q_values[0])
            else:
                s = (self.q_levels[-1] - self.q_levels[-2]) / (self.q_values[-1] - self.q_values[-2])
                p = self.q_levels[-1] + s * (t - self.q_values[-1])
        return float(np.clip(p, 0.0, 1.0))

    def median(self) -> float:
        return self.quantile(0.5)

    def mean(self) -> float:
        """E[T] = integral_0^1 Q(u) du (trapezoid on a fine grid).

        Tail-sensitive: the [0,0.05] and [0.95,1] regions rely on linear
        extrapolation of the outermost quantile models, so mean() is a rougher
        summary than median(). Reported for completeness; the scheduler should
        prefer quantiles.
        """
        u = np.linspace(0.0, 1.0, 2001)
        return float(_TRAPZ(self._quantile_vec(u), u))

    def samples(self, n: int, rng=None) -> np.ndarray:
        """Inverse-transform sampling: n draws from the distribution."""
        rng = np.random.default_rng() if rng is None else rng
        u = rng.uniform(size=int(n))
        return self._quantile_vec(u)


# --------------------------------------------------------------------------- #
# Loading (lazy + cached, so repeated predict() calls are cheap & idempotent)
# --------------------------------------------------------------------------- #
def _load():
    global _BUNDLE, _ENCODERS
    if _BUNDLE is None:
        with open(_MODEL_PATH, "rb") as f:
            _BUNDLE = pickle.load(f)
    if _ENCODERS is None:
        with open(_ENCODERS_PATH, "rb") as f:
            _ENCODERS = pickle.load(f)
    return _BUNDLE, _ENCODERS


# --------------------------------------------------------------------------- #
# Feature preparation — reproduce training-time transforms from a raw-ish dict
# --------------------------------------------------------------------------- #
def prepare_row(case_features: dict, bundle, encoders) -> np.ndarray:
    """Map a canonical feature dict to the exact model input row.

    Contract for `case_features`:
      * Categoricals given as RAW STRINGS (e.g. department='General surgery').
        Unseen or missing -> reserved code -1.
      * Numerics given as numbers; missing (absent or None) -> train-median
        imputation, and the companion `<col>_missing` indicator is set to 1.
      * Winsor caps (train 0.5/99.5 pctile) are applied to numerics that had them.
    Anything not in the model's feature_cols is ignored.
    """
    feat_cols = bundle["feature_cols"]
    cat_cols = set(bundle["categorical_cols"])
    cat_maps = encoders["categorical_maps"]
    unk = encoders["categorical_unknown_code"]
    caps = encoders["winsor_caps"]
    med = encoders["train_medians"]

    def provided(name):
        return name in case_features and case_features[name] is not None

    row = np.empty(len(feat_cols), dtype=float)
    for i, col in enumerate(feat_cols):
        if col.endswith("_missing"):
            base = col[:-len("_missing")]
            row[i] = 0.0 if provided(base) else 1.0
        elif col in cat_cols:
            v = case_features.get(col, None)
            if v is None:
                row[i] = unk
            elif isinstance(v, (int, np.integer)):
                # already an integer code — trust it
                row[i] = int(v)
            else:
                row[i] = cat_maps.get(col, {}).get(v, unk)
        else:  # numeric
            if provided(col):
                x = float(case_features[col])
            else:
                x = med.get(col, np.nan)
                x = 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else x
            if col in caps:  # winsorize like training
                lo, hi = caps[col]
                x = min(max(x, lo), hi)
            row[i] = x
    return row


# --------------------------------------------------------------------------- #
# predict() — the frozen entry point
# --------------------------------------------------------------------------- #
def predict(case_features: dict) -> CDF:
    """Return a CDF for one case. Idempotent for a fixed features dict."""
    bundle, encoders = _load()
    x = prepare_row(case_features, bundle, encoders).reshape(1, -1)

    quantiles = bundle["quantiles"]
    log_target = bundle.get("log_target", False)
    preds = []
    for q in quantiles:
        booster = bundle["models"][q]
        yhat = booster.predict(x)[0]
        if log_target:
            yhat = np.expm1(yhat)
        preds.append(yhat)
    return CDF(quantiles, np.array(preds))


def predict_matrix(X: np.ndarray) -> list:
    """Vectorized helper: many pre-encoded rows -> list[CDF]. Used by eval."""
    bundle, _ = _load()
    quantiles = bundle["quantiles"]
    log_target = bundle.get("log_target", False)
    cols = np.column_stack([
        (np.expm1(bundle["models"][q].predict(X)) if log_target
         else bundle["models"][q].predict(X))
        for q in quantiles
    ])
    return [CDF(quantiles, cols[i]) for i in range(cols.shape[0])]


# --------------------------------------------------------------------------- #
# __main__ smoke test — run predict() on the first 5 TEST rows
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json
    import pandas as pd

    proc = Path.home() / "Desktop" / "vitaldb_work" / "processed"
    df = pd.read_parquet(proc / "features.parquet")
    splits = json.load(open(proc / "splits.json"))
    bundle, encoders = _load()

    # Build inverse categorical maps so we can hand predict() RAW STRINGS,
    # exercising the string->code path (and -1 for anything unseen).
    inv = {c: {v: k for k, v in m.items()} for c, m in encoders["categorical_maps"].items()}
    cat_cols = set(bundle["categorical_cols"])

    test = df[df["caseid"].isin(splits["test"])].head(5)
    print("=" * 70)
    print("predict.py smoke test — first 5 TEST cases")
    print("=" * 70)
    for _, r in test.iterrows():
        # Reconstruct a raw-ish feature dict from the processed row:
        #  - categoricals -> back to strings via inverse maps
        #  - numerics that were imputed (missing flag == 1) are OMITTED so
        #    predict() re-imputes them and re-sets the missing indicator,
        #    faithfully reproducing the model's actual input row.
        fd = {}
        for col in bundle["feature_cols"]:
            if col.endswith("_missing"):
                continue
            if col in cat_cols:
                code = int(r[col])
                fd[col] = inv.get(col, {}).get(code, None)  # None -> -1 path
            else:
                mflag = f"{col}_missing"
                if mflag in df.columns and int(r[mflag]) == 1:
                    continue  # was missing -> omit -> re-imputed
                fd[col] = float(r[col])
        cdf = predict(fd)
        actual = float(r["duration_minutes"])
        print(f"\ncaseid {int(r['caseid'])}  (actual = {actual:.0f} min)")
        print(f"  median         : {cdf.median():7.1f} min")
        print(f"  mean           : {cdf.mean():7.1f} min")
        print(f"  P10 / P50 / P90: {cdf.quantile(0.1):6.1f} / "
              f"{cdf.quantile(0.5):6.1f} / {cdf.quantile(0.9):6.1f}")
        print(f"  P(<= actual)   : {cdf.cdf(actual):.3f}")
        s = cdf.samples(1000, rng=np.random.default_rng(0))
        print(f"  1000 samples   : mean={s.mean():.1f}, p90={np.percentile(s,90):.1f}")

    # Idempotency check
    d0 = predict(fd); d1 = predict(fd)
    same = np.allclose(d0.q_values, d1.q_values)
    print(f"\nIdempotency (same dict -> identical CDF): {same}")
