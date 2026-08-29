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

### Classifier-probability-ranked Top-N result

* Added classifier-probability-ranked Top-N evaluation alongside the existing regressor-ranked report.
* Ran the 10d / 10-year experiment.
* Result: classifier probability ranking performed poorly.
* Regressor-ranked Top-N remained much stronger:
  * top-5 excess: 0.99%
  * top-10 excess: 0.80%
  * top-20 excess: 0.60%
* Classifier-probability-ranked Top-N was near zero or negative:
  * top-5 excess: 0.02%
  * top-10 excess: 0.02%
  * top-20 excess: -0.01%
* Current read: classifier probability is not useful as a standalone ranking signal in the current setup.
* Next step: keep regressor-ranked Top-N as the primary ranking report and add by-year diagnostics.

### Validation-based regressor config selection

* Added validation-based XGBoost regressor config selection.
* Candidate 0 is the current/default regressor config; two small conservative variants are tested against it.
* Each candidate is trained on the train split and scored on validation Top-N basket excess return.
* Selection metric is mean validation excess across top-5, top-10, and top-20.
* Validation scoring is model-only, so random/momentum/universe baselines are not recomputed for every candidate.
* Final test reporting still runs once using the selected regressor.
* Saved metadata records the selected candidate and params.
* Tests passed: `146 passed`.
* Current read: this is infrastructure, not proof of improvement yet. Next step is to run the 10d / 10-year experiment and see which candidate is selected.

### Validation-selected config checks

* 10d / 10y improved versus the prior fixed-config run and beat momentum across top-5, top-10, and top-20.
* 10d / max still beat random/universe but trailed momentum slightly.
* 20d / 10y beat random/universe strongly but still trailed momentum, especially in top-5 and top-10.
* Current read: validation-based selection is useful infrastructure, but it does not yet prove a durable model edge. The model’s advantage is still horizon- and period-sensitive.
* Next step: inspect which candidates were selected and add stability/significance diagnostics before expanding the config search.

### Bootstrap confidence intervals

* Added date-level bootstrap confidence intervals to Top-N basket diagnostics.
* Bootstrap resamples by `prediction_date`, preserving daily basket structure.
* 10d / 10y still shows positive model excess, but model-minus-momentum CIs cross zero.
* Current read: the model clearly beats random/universe in this run, but the edge over momentum is not statistically convincing yet.

### Walk-forward Top-N evaluation

* Added expanding-window yearly walk-forward evaluation for SPY-relative Top-N diagnostics.
* Default setup: 5 years minimum training history, 1 validation year, 1 test year, 1-year step.
* Each fold selects the XGBoost regressor config on validation Top-N performance, then evaluates the selected model on the next test year.
* Initial 10d / 10y run produced 5 folds. The model beat the universe baseline in 80% of folds, but did not reliably beat momentum.
* Aggregate model-minus-momentum was negative across Top-N buckets: top-5 -0.47%, top-10 -0.15%, top-20 -0.15%.
* Current read: the model has stock-selection signal versus the universe, but not enough evidence of independent edge versus momentum.

### Cross-sectional Rank-NDCG walk-forward result

* Added grouped `cross_sectional_rank_ndcg` as an experimental target mode using `XGBRanker` with stocks grouped by `prediction_date`.
* Rank-NDCG trains on graded same-date realized excess-return rank relevance rather than predicting exact future excess returns.
* Extended the existing expanding-window walk-forward framework to support Rank-NDCG without changing the feature set, Rank-NDCG parameters, folds, embargo logic, training universe, or Top-N metric definitions.
* Evaluation used the `large_mega_cap_stocks` universe, 10 years of history, 5-year minimum training history, 1 validation year, 1 test year, and 5 walk-forward folds.
* The existing excess-return regressor remains the comparison model.

#### 10d Rank-NDCG walk-forward

Aggregate results:

* top-5:
  * model excess: 0.81%
  * model minus momentum: +0.19%
  * model minus universe: +0.74%
  * fold win rate vs momentum: 60%
  * fold win rate vs universe: 80%
* top-10:
  * model excess: 0.60%
  * model minus momentum: +0.03%
  * model minus universe: +0.52%
  * fold win rate vs momentum: 60%
  * fold win rate vs universe: 80%
* top-20:
  * model excess: 0.38%
  * model minus momentum: +0.02%
  * model minus universe: +0.30%
  * fold win rate vs momentum: 60%
  * fold win rate vs universe: 80%

Compared with the prior 10d regressor walk-forward run:

* regressor top-5 model minus momentum: -0.32%
* Rank-NDCG top-5 model minus momentum: +0.19%
* regressor top-10 model minus momentum: -0.39%
* Rank-NDCG top-10 model minus momentum: +0.03%

Rank-NDCG therefore materially improved the primary model-minus-momentum comparison at 10d, although performance remains regime-dependent and the partial 2026 fold strongly favored momentum.

#### 20d Rank-NDCG walk-forward

Aggregate results:

* top-5:
  * model excess: 1.83%
  * model minus momentum: -0.11%
  * model minus universe: +1.68%
  * fold win rate vs momentum: 40%
  * fold win rate vs universe: 80%
* top-10:
  * model excess: 1.47%
  * model minus momentum: +0.33%
  * model minus universe: +1.32%
  * fold win rate vs momentum: 60%
  * fold win rate vs universe: 80%
* top-20:
  * model excess: 0.72%
  * model minus momentum: +0.18%
  * model minus universe: +0.57%
  * fold win rate vs momentum: 60%
  * fold win rate vs universe: 80%

Compared with the prior 20d regressor walk-forward run:

* regressor top-5 model minus momentum: -0.74%
* Rank-NDCG top-5 model minus momentum: -0.11%
* regressor top-10 model minus momentum: -0.24%
* Rank-NDCG top-10 model minus momentum: +0.33%

Rank-NDCG again materially improved model-minus-momentum performance. Top-10 became positive overall, while Top-5 improved substantially but remained slightly below momentum.

#### Current interpretation

* Rank-NDCG is now the leading target/model formulation for the project's Top-N stock-selection objective.
* The result supports the hypothesis that directly optimizing cross-sectional ranking is better aligned with the intended watchlist task than predicting exact SPY-relative forward returns.
* The predefined adoption rule is met narrowly through Top-10: Rank-NDCG improved model-minus-momentum performance in both the 10d and 20d walk-forward tests.
* The evidence does not establish durable trading alpha. Only five walk-forward folds are available, results vary meaningfully by year, and momentum remains especially difficult to beat in the partial 2026 fold.
* Do not tune against the existing single holdout based on these results.
* Next research step: make Rank-NDCG feature-group ablation use the walk-forward evaluation framework before running the existing ablation grid.

### Rank-NDCG walk-forward feature ablation

After moving the Rank-NDCG feature-ablation runner from the single holdout to the expanding walk-forward framework, the existing 10 feature subsets were evaluated on the `large_mega_cap_stocks` universe using 10 years of history and 5 walk-forward folds.

The feature subset was the only experimental variable. Fold construction, embargo logic, Rank-NDCG parameters, target design, baselines, and Top-N evaluation remained unchanged.

#### 10d feature ablation

Selected results:

| Feature set | Features | Top-5 model minus momentum | Top-10 model minus momentum | Top-5 win rate vs momentum | Top-10 win rate vs momentum |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_features | 39 | +0.19% | +0.03% | 60% | 60% |
| drop_raw_ohlc | 35 | -0.02% | -0.04% | 60% | 60% |
| drop_price_level_trend | 24 | -0.26% | -0.12% | 40% | 40% |
| drop_volume_scale | 35 | -0.07% | -0.22% | 60% | 40% |
| drop_ichimoku | 32 | +0.45% | +0.08% | 60% | 60% |
| drop_relative_momentum | 35 | +0.16% | -0.05% | 60% | 60% |
| drop_macd_redundant | 37 | -0.07% | -0.08% | 60% | 60% |
| minimal_momentum_risk_oscillator | 16 | -0.13% | -0.01% | 40% | 60% |
| momentum_only | 4 | +0.04% | -0.10% | 60% | 60% |
| momentum_plus_volatility | 12 | -0.10% | -0.08% | 40% | 60% |

At 10d, removing the Ichimoku feature group produced the strongest result, improving both Top-5 and Top-10 model-minus-momentum performance versus the full feature set.

#### 20d feature ablation

Selected results:

| Feature set | Features | Top-5 model minus momentum | Top-10 model minus momentum | Top-5 win rate vs momentum | Top-10 win rate vs momentum |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_features | 39 | -0.11% | +0.33% | 40% | 60% |
| drop_raw_ohlc | 35 | -0.58% | -0.08% | 40% | 40% |
| drop_price_level_trend | 24 | -0.58% | -0.35% | 20% | 20% |
| drop_volume_scale | 35 | -0.87% | -0.28% | 60% | 60% |
| drop_ichimoku | 32 | -0.29% | -0.03% | 60% | 40% |
| drop_relative_momentum | 35 | +0.05% | +0.21% | 60% | 40% |
| drop_macd_redundant | 37 | +0.40% | +0.39% | 60% | 60% |
| minimal_momentum_risk_oscillator | 16 | -1.19% | -0.29% | 60% | 60% |
| momentum_only | 4 | -0.82% | -0.01% | 60% | 60% |
| momentum_plus_volatility | 12 | -0.84% | -0.11% | 60% | 60% |

At 20d, removing Ichimoku was worse than the full feature set, while removing `macd` and `signalLine` produced the strongest result. That MACD removal was worse at 10d.

#### Current interpretation

* No tested feature-group removal improved the model consistently across both the 10d and 20d horizons.
* The aggressively reduced 4-, 12-, and 16-feature subsets generally underperformed the full model.
* Removing raw OHLC, price/trend, or volume-scale information generally degraded performance.
* Ichimoku and MACD-related features appear to have horizon-dependent value rather than being consistently harmful.
* Do not combine individually favorable ablations into additional feature subsets based on these same folds; that would increase the risk of tuning feature decisions to the walk-forward sample.
* Keep the existing 39-feature model contract for the current Rank-NDCG research model.
* Freeze the feature set before the next robustness checks.


### Rank-NDCG max-history walk-forward robustness check

After freezing the Rank-NDCG target/model formulation and retaining the full 39-feature contract, the primary `large_mega_cap_stocks` universe was evaluated using `period=max`.

Before running the model, the raw Yahoo histories were audited against SPY trading sessions.

Data-coverage audit:

* 65 of 65 requested tickers fetched successfully.
* Maximum missing SPY-aligned trading days across all tickers: 0.
* Maximum internal missing-session gap: 0 trading days.
* Maximum stale ending versus SPY: 0 trading days.
* Historical candidate coverage increases gradually as newer companies begin trading:
  * 1993: 44 of 64 candidates (68.8%)
  * 2000: 53 of 64 (82.8%)
  * 2006: 58 of 64 (90.6%)
  * 2010: 62 of 64 (96.9%)
  * 2013 onward: 64 of 64 (100%)

The historical raw data therefore does not show evidence of missing-session or stale-history problems. The major remaining limitation is survivorship/selection bias because the universe consists of modern large/mega-cap companies projected backward through time rather than point-in-time historical constituents.

#### 10d max-history Rank-NDCG

28 walk-forward folds, with test years from 1999 through partial 2026.

Aggregate results:

* Top-5:
  * model excess: 1.19%
  * model minus momentum: +0.67%
  * model minus universe: +0.89%
  * fold win rate vs momentum: 75.00%
  * fold win rate vs universe: 75.00%
* Top-10:
  * model excess: 0.90%
  * model minus momentum: +0.53%
  * model minus universe: +0.60%
  * fold win rate vs momentum: 85.71%
  * fold win rate vs universe: 78.57%
* Top-20:
  * model excess: 0.64%
  * model minus momentum: +0.34%
  * model minus universe: +0.34%
  * fold win rate vs momentum: 85.71%
  * fold win rate vs universe: 89.29%

The result is materially stronger than the earlier 10-year / 5-fold Rank-NDCG test and remains positive across many different market regimes. Weak individual periods remain, including 2004, 2012, 2021-2022, and partial 2026.

#### 20d max-history Rank-NDCG

28 walk-forward folds, again covering test years from 1999 through partial 2026.

Aggregate results:

* Top-5:
  * model excess: 2.28%
  * model minus momentum: +0.80%
  * model minus universe: +1.68%
  * fold win rate vs momentum: 71.43%
  * fold win rate vs universe: 82.14%
* Top-10:
  * model excess: 1.72%
  * model minus momentum: +0.70%
  * model minus universe: +1.12%
  * fold win rate vs momentum: 78.57%
  * fold win rate vs universe: 78.57%
* Top-20:
  * model excess: 1.23%
  * model minus momentum: +0.53%
  * model minus universe: +0.63%
  * fold win rate vs momentum: 82.14%
  * fold win rate vs universe: 85.71%

The 20d result also strengthened materially relative to the earlier 10-year / 5-fold test. Performance remains regime-dependent, with notably weak periods including 2004, 2011-2012, 2020, 2022, and partial 2026.

#### Current interpretation

* Rank-NDCG remains the leading model formulation after substantially expanding the walk-forward test history.
* Both 10d and 20d results beat the horizon-matched momentum baseline on average across 28 folds.
* The advantage is not driven by a single favorable year; fold win rates versus momentum are above 70% for Top-5 and substantially higher for Top-10/Top-20 in several comparisons.
* The raw max-history data passed the new SPY-session coverage audit, reducing concern that missing or stale Yahoo histories explain the result.
* These results strengthen the evidence that the Rank-NDCG formulation captures a cross-sectional ranking signal across multiple market regimes.
* They do not establish a historically tradable alpha estimate because the modern large/mega-cap universe is projected backward through time and therefore contains survivorship/selection bias.
* Do not tune features or Rank-NDCG parameters against these robustness results.

### Rank-NDCG alternate-universe robustness check

After freezing the Rank-NDCG formulation and 39-feature contract, the model was evaluated on the `broad_sector_etfs` universe as a preselected alternate-universe robustness check.

This universe contains 18 ranked ETF candidates after excluding SPY, so Top-5 is the most meaningful comparison. Top-10 represents more than half of the universe, and Top-20 is effectively the full universe and is therefore not useful for judging ranking quality.

#### 10d broad-sector ETF walk-forward

5 walk-forward folds.

Aggregate results:

* Top-5:
  * model excess: -0.12%
  * model minus momentum: -0.28%
  * model minus universe: -0.08%
  * fold win rate vs momentum: 40%
  * fold win rate vs universe: 40%
* Top-10:
  * model excess: -0.11%
  * model minus momentum: -0.12%
  * model minus universe: -0.08%
  * fold win rate vs momentum: 20%
  * fold win rate vs universe: 0%

#### 20d broad-sector ETF walk-forward

5 walk-forward folds.

Aggregate results:

* Top-5:
  * model excess: -0.11%
  * model minus momentum: -0.34%
  * model minus universe: approximately 0.00%
  * fold win rate vs momentum: 60%
  * fold win rate vs universe: 60%
* Top-10:
  * model excess: -0.21%
  * model minus momentum: -0.28%
  * model minus universe: -0.10%
  * fold win rate vs momentum: 40%
  * fold win rate vs universe: 40%

#### Current interpretation

* The Rank-NDCG advantage observed in the `large_mega_cap_stocks` universe does not transfer cleanly to the `broad_sector_etfs` universe.
* Both 10d and 20d ETF aggregate model-minus-momentum results are negative.
* The 20d Top-5 fold win rate is 60%, but performance is unstable and the average result remains negative.
* The ETF experiment is not directly comparable to the stock-ranking task because the candidate universe is much smaller and the assets represent diversified sector/style baskets rather than individual companies.
* This negative result does not invalidate the large/mega-cap stock results. It instead suggests that the current model's useful domain may be specifically cross-sectional ranking among individual large/mega-cap stocks rather than arbitrary tradable assets.
* Do not tune the model to improve ETF performance based on these results.
* The planned robustness stage is now complete. Next step: structural refactor without changing research behavior.

### August 2026 price-adjustment audit

An audit identified a point-in-time data-validity issue in the canonical Rank-NDCG research. The project currently relies on Yahoo/yfinance historical OHLC data with retrospective corporate-action adjustment semantics. Of the 39 model features, 20 are absolute price-level or price-difference features whose historical values can therefore depend on later splits or dividends. Return-, ratio-, momentum-, volatility-, and most SPY-relative features do not have the same future scale-factor problem, and the audit did not identify corresponding issues in target alignment, chronological splitting, embargoes, SPY matching, or Rank-NDCG query construction.

The previously documented 10d and 20d Rank-NDCG walk-forward results should therefore be treated as **provisional historical results**, not leakage-clean canonical evidence, until the affected feature contract is corrected and the experiments are rerun. The planned remediation is to make Yahoo adjustment semantics explicit, remove or normalize corporate-action-scale-sensitive features, improve data provenance metadata, retrain the serving bundles, and rerun the canonical 10d/20d max-history walk-forward evaluations. Simply switching to `auto_adjust=False` is not sufficient because Yahoo also retrospectively restates historical prices for stock splits.
