# Surgery Duration Estimator

**A machine-learning model that predicts how long a surgery will take — not as a single number, but as a full probability distribution — so operating rooms can be scheduled more intelligently.**

This is a working v1 proof-of-concept: an end-to-end pipeline from raw surgical records to a served `predict(case) → distribution` interface, built to prove the approach is viable and to make the path to a production system concrete.

### At a glance

| | |
|---|---|
| **Input** | Pre-operative case data — patient demographics, procedure type, ASA class, pre-op labs |
| **Output** | A full duration **CDF** per case (any quantile, median, mean, overrun probability, samples) |
| **Core model** | Ensemble of 11 LightGBM quantile regressors → monotone-rearranged CDF |
| **Sanity model** | NGBoost with a log-normal head |
| **Data** | Public, anonymized [VitalDB](https://vitaldb.net/) dataset (~6,300 cases → ~6,300 after cleaning) |
| **Status** | v1 PoC — beats every naive baseline; not yet production-calibrated (see [How v1 is doing](#how-v1-is-doing)) |

---

## The problem

Operating rooms are one of the most expensive resources in a hospital, and how they get scheduled today is essentially a guess. A surgeon estimates "about 90 minutes," the case actually takes 130, and the rest of the day cascades: patients wait longer in preop, staff run over, the last case of the day gets bumped. Underestimate and the schedule collapses; overestimate and rooms sit empty. A better duration estimate — one that also tells you *how confident* it is — has direct dollar and patient-experience value.

The catch is that surgery duration is genuinely uncertain. Two patients scheduled for the "same" cholecystectomy can differ by an hour based on anatomy, comorbidities, or complications nobody could have foreseen. A single-number prediction throws away exactly the information a scheduler needs: the *shape* of that uncertainty.

## What this model does differently

Instead of predicting *one* number ("this case will take 112 minutes"), the model predicts the entire distribution of plausible durations for each case. Concretely, for every new surgery it outputs a **cumulative distribution function (CDF)** — a curve that answers "what's the probability this surgery finishes in under X minutes?" for every X. From that a scheduler can pull whatever they need:

- The **median** (the 50/50 point) for a default estimate.
- The **90th percentile** ("we're 90% sure it finishes by this time") for booking the room.
- A **probability of overrun** past any given deadline.

This is what "distributional" or "probabilistic" forecasting means, and it's the right frame for a scheduling problem — because scheduling is fundamentally a decision under uncertainty, and different cases (the last one of the day vs. the first one on a Monday) call for different confidence levels.

## How it works

The system is two scripts and a serving module, run in sequence:

```
  raw VitalDB CSVs
        │
        ▼   preprocess.py
  ┌─────────────────────────────────────────────────────┐
  │ cohort filtering → leakage-safe features → preop-lab │
  │ time-joins → impute/encode → chronological split     │
  └─────────────────────────────────────────────────────┘
        │   processed/  (features.parquet, encoders.pkl, schema.yaml, splits.json)
        ▼   models/v1/train_eval.py
  ┌─────────────────────────────────────────────────────┐
  │ 11 LightGBM quantile models  +  NGBoost (sanity)     │
  │ → monotone-rearranged CDF → distribution-aware eval  │
  └─────────────────────────────────────────────────────┘
        │   models/v1/  (model_lgb.pkl, metrics.json, eval_report.md, …)
        ▼   models/v1/predict.py
  predict(case_features) → CDF   ← the frozen serving interface
```

### Quantile regression, briefly

To build a full CDF we train a small ensemble of **quantile regression** models. A quantile regression model, unlike ordinary regression, doesn't learn "the average outcome" — it learns *a specific percentile* of the outcome distribution. Train one model that always predicts the 10th percentile, another for the 20th, another for the median, and so on up to the 95th, and each one becomes an honest estimate of *that level* of the outcome. Line all K of those percentile estimates up together and you've traced out the shape of the full distribution for that case.

We train eleven such models (at the 5th, 10th, 20th, …, 90th, 95th percentiles) on gradient-boosted decision trees (LightGBM), on a log-transformed target so the heavy right tail of surgery durations doesn't dominate. Because the eleven models are trained independently, their predictions can occasionally cross out of order — the 60th-percentile estimate coming in below the 50th, for instance. We fix that with a well-known technique called **monotone rearrangement** (sorting the predictions), which provably produces a valid, non-decreasing CDF and reduces error in the process. Between the eleven anchor points we interpolate linearly to get a smooth curve.

We also fit a second model (NGBoost with a log-normal distribution) as a sanity check, to make sure the main model isn't leaving obvious distributional signal on the table. It isn't.

### Leakage discipline (why you can trust the numbers)

A duration model is trivially easy to cheat: anything measured *during or after* the operation (blood loss, drugs given, the actual end time) is a near-perfect predictor of duration — and completely useless at scheduling time, because you don't have it yet. The pipeline enforces a hard rule: **a feature may only use information knowable before `opstart`.** Post-op labs, intra-op measurements, and outcomes are structurally excluded, and the train/val/test split is chronological (not random) so the model is always evaluated on "future" cases. Every one of these decisions is written down as it's made — see the `decisions_log.md` files.

## Repository layout

```
vitaldb_work/
├── clinical_data.csv, lab_data.csv, *_parameters.csv   raw VitalDB inputs (read-only, never mutated)
├── preprocess.py               Stage 1: raw → clean, leakage-safe feature table
├── processed/                  preprocessing outputs (model inputs)
│   ├── features.parquet          canonical feature table + label
│   ├── encoders.pkl              fitted winsor caps, medians, category maps, scaler (train-only)
│   ├── schema.yaml               column contract (name, dtype, source, role)
│   ├── splits.json               train/val/test case-id lists (chronological)
│   └── decisions_log.md          every preprocessing decision + why
└── models/v1/                   Stage 2: model + evaluation + serving
    ├── train_eval.py             trains both models, runs the full evaluation
    ├── predict.py                the serving interface (CDF object + predict())
    ├── model_lgb.pkl             the 11-quantile LightGBM ensemble
    ├── model_ngb.pkl             the NGBoost sanity model
    ├── metrics.json              all numbers, machine-readable
    ├── eval_report.md            the honest human-readable results
    ├── reliability_plot.png      calibration diagnostic
    ├── stratified_errors.csv     per-slice metrics (by department, ASA, emergency, …)
    └── decisions_log.md          every modeling decision + why
```

The serving contract in `predict.py` is deliberately **frozen**: `predict(case_features) → CDF`, where the `CDF` exposes `.quantile(q)`, `.cdf(t)`, `.median()`, `.mean()`, and `.samples(n)`. v2 can re-implement everything behind it — a different model, a calibration layer, a new dataset — without breaking any downstream code that consumes it.

## What it needs

**Python:** 3.9+ (developed on 3.9).

**Python packages:**

```bash
pip install pandas numpy scikit-learn lightgbm ngboost pyyaml pyarrow matplotlib
```

**System dependency — OpenMP (LightGBM's one gotcha):** LightGBM needs the OpenMP runtime, which isn't installed by default on macOS. If `import lightgbm` fails with a `libomp.dylib` error:

```bash
brew install libomp        # macOS
# Debian/Ubuntu: sudo apt-get install libgomp1   (usually already present)
```

**Data:** the four VitalDB CSVs (`clinical_data.csv`, `lab_data.csv`, and their two `*_parameters.csv` dictionaries). They're already in this repo; the source is the open [VitalDB](https://vitaldb.net/) dataset. The raw files are treated as immutable — the pipeline reads them and writes only to `processed/` and `models/`.

**Compute:** trivial. The whole pipeline (preprocess + train 11 LightGBM models + NGBoost + full evaluation) runs in well under a minute on a laptop CPU. No GPU required.

## Running it

```bash
python3 preprocess.py            # builds processed/features.parquet + friends
python3 models/v1/train_eval.py  # trains, evaluates, writes the report + plots
python3 models/v1/predict.py     # smoke-test: prints CDF summaries for 5 held-out cases
```

Each script prints a summary to stdout and writes its artifacts; start with `models/v1/eval_report.md` to read the results.

## How v1 is doing

On the held-out test set (946 surgeries the model has never seen):

- **Distributional accuracy (the metrics that matter):** mean pinball loss 15.998, CRPS 32.86 minutes — both improvements over the NGBoost log-normal baseline, meaning the model is genuinely learning distribution shape and not just parametric assumptions.
- **Point accuracy (a sanity check, not the goal):** median absolute error 43.6 minutes, MAPE 41.4%. Every naive baseline is beaten cleanly — predicting the average duration for the specific procedure name (a strong baseline) gives 58.2% MAPE; the surgical-department average gives 89%; a single global average gives 117%.
- **Where it's weakest — and we're upfront about it:** the predicted uncertainty intervals are currently a bit too *narrow* — the 80% interval covers only 71% of actual cases in practice. This is a known, well-understood failure mode when you train quantiles independently on a skewed target with limited data, and the fix (a post-hoc recalibration layer using split-conformal / conformalized quantile regression) is standard, cheap, and preserves the serving interface. It's the first thing on the v2 list.
- **Best-performing case types:** breast, colorectal, biliary/pancreatic, thyroid. **Weakest:** thoracic, hepatic, transplantation, and ASA-4/5 (very sick) patients — all small-sample, high-variance slices, exactly the ones a larger dataset most directly helps.

## Where it could go

The v1 model does the hard technical scaffolding — a valid CDF, leakage-safe features, honest evaluation, a frozen interface. The interesting problems are still ahead, and the architecture was built to make them additive rather than rewrites.

**Near-term, high-leverage:**

- **A scheduling-aware objective.** Today the model minimizes generic quantile loss, which treats a 10-minute underestimate the same as a 10-minute overestimate. In an OR those errors have very different costs — an underestimate cascades into the whole afternoon; an overestimate leaves a room idle. The next step is to evaluate (and eventually train) against a **newsvendor-style overage/underage cost**, so the model is optimized for the decision it actually feeds, not for symmetric statistical error. This is the immediate next milestone.
- **Calibration.** The conformal recalibration layer described above — turns "sharp but overconfident" into "sharp *and* honest," which is the property a scheduler needs before it can trust a quantile.
- **The data unlock.** The biggest limiter on v1 isn't the modeling — it's that VitalDB is anonymized. Timestamps are stripped (so "day of week" and "time of day," which matter enormously, aren't recoverable), surgeon identifiers are gone (so surgeon-level speed and experience effects are invisible), and the working cohort is only ~4,400 training cases. The schema already carries placeholder columns for `op_hour`, `op_dow`, `surgeon_id`, and a surgeon-procedure experience count — they slot in with **zero pipeline changes** the moment a non-anonymized source comes online. This is a data question, not a re-architecture.

**Bigger, cooler bets:**

- **Predict-then-optimize.** Wire the CDFs into an actual OR scheduler and optimize the *day*, not just each case — pick booking quantiles per slot from live hospital state, and measure the real prize: OR utilization, overtime, and case-bump reduction. This is the step that turns "good model" into a dollar figure.
- **Live intra-operative re-forecasting.** VitalDB's real richness is high-resolution intra-operative *waveform* data (the "vital" signals) that this tabular model doesn't touch yet. A second model could **update the duration estimate mid-case** as the surgery progresses — telling the charge nurse at 11:40 that the noon case is now likely to start at 12:25 — a live companion to the pre-op forecast.
- **Beyond duration.** The same distributional machinery generalizes to adjacent scheduling pain points: probability of same-day cancellation, likely ICU-bed need, turnover-time prediction, and staffing-demand forecasts.
- **Cross-hospital transfer.** Train a base model on a large multi-center corpus, then fine-tune per site — so a new hospital gets a useful model on day one instead of after collecting years of its own data.

## Design notes

A few principles baked into the repo, in case you extend it:

- **Raw data is immutable.** Nothing ever writes back to the source CSVs; the pipeline is reproducible from scratch.
- **Every non-trivial decision is logged.** The `decisions_log.md` files record what was chosen, what the alternatives were, and why — so the reasoning survives past the person who wrote it.
- **Fit on train, only ever on train.** Encoders, imputers, winsor caps, and scalers are all fit on the training split alone and applied outward, so evaluation numbers aren't quietly inflated by leakage.
- **The serving interface is a contract.** Downstream code depends on `predict() → CDF`, nothing more — which is what makes every "where it could go" item above a swap-in rather than a teardown.
