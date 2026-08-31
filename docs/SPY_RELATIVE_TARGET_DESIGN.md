# SPY-Relative Target Design

> **Status — historical design document.** This note documents the earlier
> SPY-relative regression/classification design and should not be read as the
> current serving contract. Current serving uses Rank-NDCG with explicit 10d
> and 20d horizons.

## Purpose

The current project predicts raw forward returns. That baseline is useful, but it appears to capture broad market drift more than stock-specific edge.

This branch changes the main research target to SPY-relative stock selection. The model should learn which stocks are likely to outperform or underperform SPY over the configured forward horizon. The default horizon is 10 trading days.

The app’s eventual trading-facing output should be a ranked watchlist, not automatic trade execution.

## Current Branch Status

This branch now implements SPY-relative target creation using clearer benchmark naming:

```text
raw_forward_return
benchmark_forward_return
excess_forward_return
beat_benchmark_target
```

`targetReturns` remains a compatibility alias for `excess_forward_return`.

The default horizon is still 10 trading days, but the trainer now supports configurable horizons and `--all-horizons` for 5d, 10d, 20d, and 50d.

Evaluation now includes per-date top-N ranked basket reports.

Baselines now include random, same-date universe, SPY benchmark raw return, horizon-matched momentum, and SPY-relative momentum.

Momentum and SPY-relative momentum features were added to the XGBoost model feature contract.

Clean all-horizons testing showed the model often beats random/universe baselines but does not beat horizon-matched momentum.

Current conclusion: the branch successfully reframes the project as SPY-relative ranked stock selection, but the current regression/classification setup is not yet a winning stock-selection model.

## Primary Regression Target

Default horizon: 10 trading days. The trainer supports configurable horizons.

For each ticker row:

```text
stock_forward_return = (stock_close_t_plus_10 - stock_close_t) / stock_close_t
spy_forward_return = (spy_close_t_plus_10 - spy_close_t) / spy_close_t

target = stock_forward_return - spy_forward_return
```

The primary model output is:

```text
predicted_excess_return_vs_spy
```

Meaning:

```text
expected stock return over the configured forward horizon minus expected SPY return over the same period
```

A positive value means the model expects the stock to beat SPY.
A negative value means the model expects the stock to lag SPY.

## Secondary Classifier Target

The classifier should predict whether a stock beats SPY over the configured forward horizon. The default horizon is 10 trading days.

```text
beat_spy_target = 1 if stock_forward_return > spy_forward_return else 0
```

This replaces the old raw direction target:

```text
old: 1 if stock_forward_return > 0 else 0
new: 1 if stock_forward_return > spy_forward_return else 0
```

The classifier output should be interpreted as:

```text
probability_of_beating_spy
```

## SPY Alignment

SPY forward returns must be aligned by prediction date, not by row position across the full concatenated dataset.

Required approach:

1. Fetch or include SPY historical OHLCV data.
2. Build SPY’s own forward return using the same `prediction_days` horizon.
3. Create a date-indexed mapping from SPY date to SPY forward return.
4. For each stock row, join the matching SPY forward return by date.
5. Drop stock rows where the matching SPY forward return is unavailable.

The join key should be the prediction date. In the current data pipeline, this likely means preserving the original price date through target creation and splits.

SPY should not be included as a normal candidate stock in ranked stock-selection evaluation unless explicitly intended. It is the benchmark.

## Columns To Preserve

Target creation should preserve enough columns for later evaluation:

```text
Ticker
prediction_date or source index/date
stock_forward_return
spy_forward_return
targetReturns or targetExcessReturns
beat_spy_target
```

Recommended naming:

```text
forward_return
spy_forward_return
targetReturns
beat_spy_target
```

Where `targetReturns` becomes the SPY-relative excess return for compatibility with the existing trainer.

Alternative clearer naming:

```text
raw_forward_return
benchmark_forward_return
excess_forward_return
beat_benchmark_target
```

This branch currently uses the clearer benchmark naming while keeping `targetReturns`
as a compatibility alias for `excess_forward_return`.

## Leakage Risks

Main leakage risks:

1. Computing indicators after concatenating tickers.

   * Current pipeline avoids this by calculating indicators per ticker before concatenation. Keep that behavior.

2. Letting target shifts cross ticker boundaries.

   * Current `DataPreparator.create_target` uses ticker-aware shifts. Keep that behavior.

3. Misaligning SPY by row number instead of date.

   * SPY returns must be joined by date, not by global row position.

4. Training on future SPY information as a feature.

   * SPY forward return is allowed only as a label/baseline field, not as a model feature.

5. Tuning rules on the test set.

   * Any threshold or top-N choice used for final reporting should either be fixed in advance or selected on validation data only.

6. Evaluating top-N rows globally instead of per prediction date.

   * Ranked watchlist evaluation must rank stocks within the same date. A global top-N bucket can accidentally overrepresent certain market periods.

## Splitting Strategy

Use global date-based chronological train/validation/test splits with embargo gaps.

The previous baseline split each ticker independently:

```text
AAPL train -> embargo -> AAPL validation -> embargo -> AAPL test
MSFT train -> embargo -> MSFT validation -> embargo -> MSFT test
...
```

That was acceptable for the raw-return baseline, but it is not the best fit for SPY-relative ranked stock selection. The new goal is to evaluate the model as if it were building a same-day watchlist across the available stock universe.

For this branch, rows should be assigned to train, validation, or test based on their prediction date, not their row position within each ticker.

Preferred split structure:

```text
global train dates -> embargo dates -> global validation dates -> embargo dates -> global test dates
```

Target creation should still be ticker-aware. The stock forward return must be computed within each ticker so one ticker’s future close never crosses into another ticker’s rows.

Correct separation of responsibilities:

```text
target creation: ticker-aware forward shifts
SPY alignment: date-based join
splitting: global date-based chronological split
evaluation: per-date ranking
```

The embargo should remove approximately `prediction_days` trading dates between adjacent splits. For the default 10-trading-day target, this prevents rows near a split boundary from using future target information that overlaps the next evaluation window; other horizons should use a matching embargo.

Recommended process:

1. Preserve each row’s prediction date.
2. Create stock forward returns with ticker-aware shifts.
3. Create SPY forward returns separately.
4. Join SPY forward returns to stock rows by prediction date.
5. Drop rows without usable stock or SPY forward-return targets.
6. Build sorted unique prediction dates.
7. Assign dates to train, validation, embargo, and test windows.
8. Drop embargo-window rows.
9. Fit the scaler on training features only.
10. Transform validation and test features using the fitted training scaler.

Split metadata should include enough information for later ranked evaluation:

```text
_source_index
Ticker
prediction_date
raw_forward_return
benchmark_forward_return
excess_forward_return
beat_benchmark_target
```

Important: SPY forward return, raw forward return, and excess forward return are label/evaluation fields only. They must not be included as model features.


## Model Behavior

Linear Regression:

* Keep as a simple scaled baseline.
* It can predict SPY-relative excess return instead of raw return.

XGBoost Regressor:

* Primary model.
* Train on SPY-relative excess forward return.

XGBoost Classifier:

* Secondary model.
* Train on `beat_spy_target`.

Saved metadata should make the target type explicit:

```text
target_type = "spy_relative_forward_return"
benchmark_ticker = "SPY"
prediction_days = 10
```

`prediction_days = 10` is the default, not the only supported horizon.

## Primary Evaluation

The project should evaluate ranked stock-selection usefulness, not only isolated prediction error.

For each prediction date in the test split:

1. Score all available candidate tickers.
2. Rank by predicted SPY-relative return.
3. Select top N stocks.
4. Compare selected stocks against:

   * SPY forward return over the same date
   * same-date equal-weight universe forward return
   * full test-set average return

Recommended top-N buckets:

```text
top_5
top_10
top_20
```

If a date has fewer than N available tickers, select the smaller available count or skip that date depending on the report’s purpose. The first implementation should use:

```text
selected_count = min(N, available_tickers_that_date)
```

## Evaluation Metrics

For selected top-N groups, report:

```text
date_count
average_selected_raw_forward_return
average_selected_spy_forward_return
average_selected_excess_return_vs_spy
beat_spy_rate
average_equal_weight_universe_forward_return
average_selected_return_minus_universe_return
average_predicted_excess_return
```

Also useful:

```text
median_selected_excess_return_vs_spy
positive_raw_return_rate
average_number_of_candidates_per_date
```

Regression diagnostics should still exist:

```text
MSE
MAE
R2
Pearson correlation
Spearman rank correlation
```

But these are secondary. Ranking quality matters more than exact return magnitude.

Classifier diagnostics should change from raw up/down direction to beat-SPY classification:

```text
accuracy
precision
recall
f1
always_beat_spy_baseline_accuracy
always_lag_spy_baseline_accuracy
probability threshold reports
```

## Desired Prediction Output

Eventually, prediction output should include:

```text
ticker
predicted_raw_return
predicted_excess_return_vs_spy
probability_of_beating_spy
rank_in_universe
label
```

Possible labels:

```text
candidate
watch
avoid
```

Simple first-pass labeling:

```text
candidate: high predicted excess return and high probability of beating SPY
watch: modest or uncertain predicted excess return
avoid: negative predicted excess return or low probability of beating SPY
```

Do not implement complex trading rules yet.

## Files Likely To Change

### `src/data/data_prep.py`

Likely changes:

* Add SPY-relative target creation.
* Preserve raw stock forward return.
* Join SPY forward return by date.
* Create beat-SPY classifier target.
* Expand split metadata.

### `src/train/trainer.py`

Likely changes:

* Train regressor on excess return.
* Train classifier on beat-SPY target.
* Pass raw returns, SPY returns, dates, tickers, and predictions into evaluator.
* Save model metadata describing benchmark-relative target.

### `src/train/evaluator.py`

Likely changes:

* Add per-date top-N ranking evaluation.
* Add SPY-relative reports.
* Add equal-weight universe comparison.
* Possibly rename old up/down language so reports are not misleading.

### `src/inference/predictor.py`

Likely later changes:

* Return predicted excess return vs SPY.
* Return probability of beating SPY.
* Produce ranked watchlist output.

### Tests

Likely tests:

* `tests/test_data_prep.py`

  * verifies stock forward return
  * verifies SPY forward return alignment by date
  * verifies excess return target
  * verifies beat-SPY classifier target
  * verifies no ticker-boundary shift leakage

* `tests/test_evaluator.py`

  * verifies top-N per-date selection
  * verifies selected returns vs SPY
  * verifies selected returns vs equal-weight universe
  * verifies dates are evaluated independently

* `tests/test_trainer.py`

  * verifies trainer uses beat-SPY labels for classifier
  * verifies metadata target type

* `tests/test_predictor.py`

  * later, verifies prediction output names and feature alignment

## Staged Implementation Plan

### Stage 1: Design and tests

1. Commit this design doc.
2. Add unit tests for SPY-relative target creation.
3. Add unit tests for top-N per-date evaluator behavior.

No model behavior should be changed before the tests define the expected target and evaluation behavior.

### Stage 2: Target creation

1. Modify `DataPreparator.create_target` or add a new target method.
2. Preserve raw stock forward return.
3. Compute or join SPY forward return by date.
4. Create excess return target.
5. Create beat-SPY classifier target.
6. Expand split metadata.

### Stage 3: Trainer wiring

1. Train Linear Regression on excess return.
2. Train XGBoost regressor on excess return.
3. Train XGBoost classifier on beat-SPY target.
4. Update logging labels so reports clearly say SPY-relative / beat-SPY.

### Stage 4: Ranked evaluation

1. Add top-N per-date portfolio report.
2. Compare selected stocks against SPY and equal-weight universe.
3. Log top 5, top 10, and top 20 candidate buckets.
4. Keep old global bucket reports only as secondary diagnostics.

### Stage 5: Inference/watchlist output

1. Update predictor output names.
2. Add ranked watchlist structure.
3. Add candidate/watch/avoid labels.
4. Keep this as advisory output only, not trading automation.

## Explicit Non-Goals For This Branch

Do not add:

* live trading
* automated execution
* sector adjustment
* beta adjustment
* ticker-list optimization to chase backtest results
* large parameter fishing
* major infrastructure rewrite

This branch should answer one question:

Can a simple technical-indicator model rank stocks that outperform SPY over the configured forward horizon better than naive baselines?
