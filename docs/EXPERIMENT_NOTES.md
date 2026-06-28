# Experiment Notes

## 6/25/26 Raw return baseline

### Current setup

* Target is future raw return using row shifts, so `10` means 10 trading days.
* Switched default horizon from 5 trading days to 10 trading days.
* Added `random_state=42` so XGBoost runs are easier to compare.

### Findings

* The basic classifier is still weak and often predicts mostly up.
* The more useful signal is in ranked / thresholded predictions, not plain accuracy.
* 10-day predictions looked better than 5-day predictions in the small 10-ticker test.
* Looser regressor settings did not clearly beat the 10-day baseline.
* Best raw-return baseline was the reproducible 10-day version.

### Possible next experiments

* Test on a larger ticker universe.
* Compare predicted returns against SPY-relative returns.
* Evaluate rules that combine classifier probability with regressor ranking.

### Combined signal / reporting

* Added a combined signal report using classifier probability `>= 0.60` and top 20% regressor rank.
* Added a readable experiment summary so raw training logs are easier to review.
* In the small test, regressor top 20% looked stronger than the combined signal.
* `core_25` had some classifier-threshold edge.
* `core_50` was basically baseline/noise.
* `broad_200` had a tiny high-confidence classifier pocket but weak regressor edge.
* Combined signal underperformed.
* Raw forward return is probably not the final target.

## 6/26/26 SPY-relative / horizon research

### Current setup

* Switched main research target from raw forward return to SPY-relative excess return.
* Added ranked Top-N evaluation.
* Added equal-weight basket backtest summaries.
* Added random, universe, SPY benchmark, and simple momentum baselines.
* Added configurable horizons and `--all-horizons` for 5, 10, 20, and 50 trading days.
* Updated CLI wording so predictions are labeled as SPY-relative excess return, not raw expected price.

### Findings

* The model usually beats random and the equal-weight universe.
* The model often loses to the simple momentum baseline.
* 10-day top-5 currently looks like the most promising horizon/bucket.
* The classifier remains weak by accuracy.
* Regressor ranking and basket results are more useful than plain R² or accuracy.
* The research pipeline is now more useful than the original raw-return baseline, but the model signal is still uncertain.

### Hypothesis

* The model may have weak stock-selection signal, but current features are not strong enough to consistently beat momentum.
* Simple `dailyReturn` momentum is probably too crude, but it is still a useful sanity-check baseline.
* The next likely path is to improve momentum / relative-strength features and baselines, then rerun horizon comparisons.
* A fuller trading simulator should wait until the model looks better against stronger baselines.

### Baseline update

- Added horizon-matched momentum baselines.
- Added SPY-relative momentum baselines.
- SPY-relative momentum ranks the same as raw momentum within each prediction date because the same SPY momentum value is subtracted from every stock on that date.
- Current read: stronger momentum baselines usually beat the model, so the next experiment should add momentum / relative-strength signals as model features.

### Momentum feature experiment

- Added raw momentum and SPY-relative momentum features to XGBoost.
- Feature count increased from 31 to 39.
- Clean all-horizons run did not improve model performance against momentum baselines.
- Current read: the issue is probably not just missing momentum inputs; the regression objective may be poorly aligned with per-date ranking.

### 10-year history experiment

* Added configurable raw yfinance OHLCV caching and a `--period` trainer option.
* Ran a clean cached all-horizons experiment with:

```bash
python -m src.train.trainer --all-horizons --period 10y
```

* Cache behavior looked good: all 150 requested tickers loaded from cache, and 150/150 produced valid prepared data.
* The data loading phase was fast after cache population; the remaining runtime was mostly model training/evaluation across four horizons.
* Compared with the prior 5-year run, the 10-year history improved the model’s ranked basket results.
* Current read:

  * 5d: model beat momentum in top-5, top-10, and top-20.
  * 10d: model narrowly beat momentum in top-5 and top-10, and roughly tied/slightly trailed in top-20.
  * 20d: model improved but still trailed momentum.
  * 50d: model trailed momentum in top-5 but beat momentum in top-10 and top-20.
* Interpretation: more history appears useful. The model is now more competitive with simple momentum, especially at 5d and 10d horizons. However, longer history should be tested carefully because older data can introduce ticker coverage gaps, market-regime differences, survivorship effects, and corporate-action/data-quality issues.

### Max-history experiment

* Ran:

```bash
python -m src.train.trainer --all-horizons --period max
```

* The max-history run completed successfully with 150/150 valid tickers.
* Prepared data was much larger than the 5-year and 10-year runs; for example, the 5d horizon used roughly 544k train rows, 167k validation rows, and 248k test rows.
* Results were mixed compared with the 10-year run:

  * 5d: model beat random/universe/momentum, but was slightly weaker than the 10-year run.
  * 10d: model beat random/universe but trailed momentum; weaker than the 10-year run.
  * 20d: model beat random/universe/momentum across top-5, top-10, and top-20; this was the strongest max-history result.
  * 50d: model beat random/universe but trailed momentum; weaker than the 10-year run.
* Current read: `period=max` is not clearly better than `period=10y`. More history adds useful signal in some places, especially 20d, but also adds older regimes, uneven ticker histories, survivorship bias, and possible stale market structure effects.
* Working default for now: use `--period 10y` for main experiments, and treat `--period max` as a secondary robustness check.
* Performance note: the max-history run suggests the slowest part is not XGBoost fitting. Model fitting completed quickly relative to the full horizon runtime; most runtime appears to be spent in post-fit evaluation/reporting.

## Runtime/workflow note

Added faster experiment controls:

```bash
# Fast iteration
python -m src.train.trainer --prediction-days 10 --period 10y --random-trials 20

# Full check
python -m src.train.trainer --all-horizons --period max --random-trial-workers 4
```

Notes:
- `--random-trials` defaults to `100`; use `20` for quick iteration.
- `--random-trial-workers` defaults to `4`; use `1` for sequential random-baseline execution.
- Max all-horizons runtime improved from about `8m42s` to about `3m36s`; post-fit report time dropped from about `122–127s`/horizon to about `45–47s`/horizon.

## 6/28/26 SPY-relative research plan

### Research memo takeaway

* Reviewed the SPY-relative excess-return pipeline research memo.
* Current conclusion: do not abandon the SPY-relative target yet.
* The model still needs stronger falsification before changing targets or adding major complexity.
* The main question is whether the model has real stock-selection edge beyond simple momentum.

### Current interpretation

* The 10-year 10d run is encouraging because the model beats random/universe and narrowly beats momentum in top-5/top-10.
* The max-history 10d run is more skeptical because the model still beats random/universe but trails momentum.
* This suggests the current model may be learning a momentum-like signal, or that performance is regime-dependent.
* The next experiments should compare ranking signals and expose instability across time before changing the target.

### Next experiment

Test classifier-probability-ranked Top-N against the current regressor-ranked Top-N.

Current ranking score:

```text
predicted_excess_return
```

Comparison ranking score:

```text
predicted_beat_benchmark_probability
```

Questions to answer:

* Does classifier probability rank candidates better than predicted excess return?
* Does classifier ranking improve top-5 or top-10 selection?
* Does classifier ranking beat random, universe, and momentum baselines?
* Is any improvement stable across `--period 10y` and `--period max`?
* Does classifier ranking work better at the 10d or 20d horizon?

Expected readout:

* If classifier-ranked Top-N beats or complements regressor-ranked Top-N, test a blended ranking score later.
* If classifier-ranked Top-N does not help, prioritize by-year reports, concentration diagnostics, and significance testing before adding new model complexity.

Implementation note:

* Add classifier-probability-ranked Top-N as a parallel evaluation report.
* Keep the existing regressor-ranked Top-N report unchanged.
* Do not change the SPY-relative target, model parameters, feature columns, or existing report keys.

