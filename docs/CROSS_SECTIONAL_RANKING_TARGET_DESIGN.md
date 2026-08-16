# Cross-Sectional Ranking Target Design

## Purpose

The project has moved from raw return prediction toward SPY-relative stock selection.

Current target:

```text
excess_forward_return = stock_forward_return - spy_forward_return
```

That is a better target than raw price or raw return prediction, but there is still a mismatch between the model’s training objective and the intended product.

The current XGBoost regressor tries to predict exact future SPY-relative excess return.

The actual useful output should be a ranked watchlist:

```text
Which stocks look most likely to outperform SPY over the next N trading days?
```

This document proposes a near-term experiment: add a cross-sectional ranking/classification target mode and compare it against the current excess-return regression target.

This is not a final decision to remove regression. The first goal is to test whether a ranking-aligned target improves Top-N watchlist performance.

## Prediction Timing Assumption

Current model features include same-day OHLCV-derived values such as `High`,
`Low`, `Close`, and `Volume`. Ranking experiments therefore assume predictions
are made after the market close for the `prediction_date`. A pre-close workflow
would need a separate feature contract that excludes not-yet-known same-day
fields.

## Research Question

Primary question:

```text
Can a cross-sectional ranking/classification target improve Top-N SPY-relative stock selection compared with the current excess-return regression target?
```

The model should be evaluated against strong simple baselines, especially horizon-matched trailing momentum:

```text
5d horizon  -> momentum_5d
10d horizon -> momentum_10d
20d horizon -> momentum_20d
50d horizon -> momentum_50d
```

A useful model should not only beat random selection or the average universe. It should improve on, or at least compete strongly with, momentum.

## Adoption Rule

Before running the experiment, define what would count as a meaningful improvement.

The cross-sectional ranking target should only become the primary target if it clearly improves practical Top-N evaluation.

Initial adoption rule:

```text
Adopt the ranking target only if it improves model_minus_momentum_excess_return
for Top-5 or Top-10 in both 10d and 20d walk-forward tests, without relying on
one isolated lucky fold.
```

Supporting evidence should include:

```text
- positive or improved average model_minus_momentum_excess_return
- better Top-5 or Top-10 selected basket excess return
- reasonable consistency across folds
- no major degradation versus the equal-weight universe baseline
```

If the ranking target improves one horizon but fails another, treat it as experimental rather than replacing the current regression target.

## Why Test A Ranking Target?

The current regression target asks:

```text
Can the model predict the exact future SPY-relative excess return?
```

That target is noisy. A model can reduce regression error by making conservative near-zero predictions while still ranking the best stocks poorly.

The actual task is cross-sectional:

```text
On a given prediction date:
  score all available candidate stocks
  rank them
  select the top 5 / 10 / 20
```

A cross-sectional target asks a more relevant question:

```text
On this date, which stocks ended up among the better future SPY-relative performers?
```

This better matches the intended ranked-watchlist output.

## Definition Of Cross-Sectional

Cross-sectional means comparing stocks to each other at the same point in time.

Example:

```text
prediction_date  ticker  excess_forward_return
2024-01-02       AAPL    +1.2%
2024-01-02       MSFT    -0.3%
2024-01-02       NVDA    +2.1%
2024-01-02       JPM     -0.8%
```

A time-series question is:

```text
How does AAPL today compare with AAPL's own past?
```

A cross-sectional question is:

```text
On 2024-01-02, how does AAPL compare with MSFT, NVDA, JPM, and the rest of the universe?
```

The project’s ranking output is cross-sectional because it selects the best candidates from the same date’s universe.

## Proposed Target Mode

Keep the current SPY-relative return calculation:

```text
raw_forward_return = stock_forward_return
benchmark_forward_return = spy_forward_return
excess_forward_return = raw_forward_return - benchmark_forward_return
```

Then, for each `prediction_date`, rank candidate stocks by realized future `excess_forward_return`.

Recommended helper column:

```text
excess_return_rank_pct_by_date
```

Definition:

```text
excess_return_rank_pct_by_date =
  percentile rank of excess_forward_return within the same prediction_date
```

Example:

```text
prediction_date  ticker  excess_forward_return  rank_percentile
2024-01-02       NVDA    +2.1%                  1.00
2024-01-02       AAPL    +1.2%                  0.67
2024-01-02       MSFT    -0.3%                  0.33
2024-01-02       JPM     -0.8%                  0.00
```

Ranks must be computed separately within each `prediction_date`, never globally across all dates.

Implementation convention:

```text
excess_return_rank_pct_by_date =
  (within_date_rank_ascending - 1) / (candidate_count_for_date - 1)
```

Higher values are better.

This produces a 0.00-to-1.00 percentile scale where the worst same-date performer is 0.00 and the best same-date performer is 1.00.

For dates with fewer than two candidate rows, ranking labels should be omitted or treated as unavailable.

## First Ranking Target

Preferred first implementation:

```text
top 20% by excess_forward_return    -> label 1
bottom 20% by excess_forward_return -> label 0
middle 60%                          -> excluded from classifier training
```

Recommended target columns:

```text
top_quintile_target
ranking_train_sample
```

Possible definitions:

```text
top_quintile_target = 1 if excess_return_rank_pct_by_date >= 0.80
top_quintile_target = 0 if excess_return_rank_pct_by_date <= 0.20
top_quintile_target = NaN otherwise

ranking_train_sample = True for top/bottom quintile rows
ranking_train_sample = False for middle-quintile rows
```

This creates a cleaner first classification problem:

```text
Can the model distinguish strong future outperformers from strong future underperformers?
```

The middle 60% is likely noisier. It should be preserved for validation/test evaluation, but excluded from classifier fitting.

## Top/Bottom Quintile Tradeoff

Training only on the top and bottom quintiles is intentional, but it has a tradeoff.

Benefit:

```text
Cleaner labels and less noisy training.
```

Risk:

```text
The model may learn to separate extremes, but be weaker at distinguishing middle-ranked candidates.
```

This matters because Top-N evaluation scores all candidate stocks, not only the top and bottom realized quintiles.

Recommended diagnostic:

```text
For selected Top-N picks, report their realized same-date rank-percentile distribution.
```

Useful summary buckets:

```text
selected picks that landed in top 20%
selected picks that landed in middle 60%
selected picks that landed in bottom 20%
average realized rank percentile of selected picks
```

This checks whether the classifier is actually selecting future top-ranked stocks or merely producing noisy scores.

## Alternative Targets For Later

A later experiment can use all rows:

```text
top_quintile_target = 1 if stock is in top 20% by same-date excess_forward_return
top_quintile_target = 0 otherwise
```

This is simpler, but it creates an imbalanced classification problem and forces the model to learn from many middling cases.

Another later experiment can use direct learning-to-rank methods, such as pairwise or listwise ranking objectives grouped by `prediction_date`.

Do not implement direct learning-to-rank first. It requires more pipeline changes. The top-vs-bottom classifier is a simpler first test of whether a ranking-aligned target helps.

## Model Output

The ranking classifier should output:

```text
probability_of_top_quintile_outperformance
```

Meaning:

```text
estimated probability that this stock belongs in the top 20%
of same-date SPY-relative performers over the selected horizon
```

For ranked selection, candidates should be sorted by this score within each prediction date.

Important interpretation:

```text
The probability is primarily a ranking score, not a fully calibrated probability.
```

Because evaluation ranks candidates within each date, exact probability calibration across different dates is less important than same-date ordering.

## Relationship To Current Regression Model

The current excess-return regressor should remain as the baseline comparison.

Do not remove regression.

Target modes should be compared:

```text
target_mode = "excess_return"
  train XGBoost regressor on numeric excess_forward_return

target_mode = "cross_sectional_top_bottom"
  train XGBoost classifier on top-vs-bottom same-date quintile labels
```

The ranking target should only become primary if it improves practical Top-N evaluation under the adoption rule above.

## SPY Handling

SPY remains the benchmark.

Required behavior:

1. Compute each ticker’s forward return using ticker-aware shifts.
2. Compute SPY’s forward return over the same horizon.
3. Join SPY forward return to stock rows by `prediction_date`.
4. Compute `excess_forward_return`.
5. Remove SPY from candidate rows.
6. Create same-date cross-sectional ranking labels from candidate rows.

SPY must be fetched for benchmark construction, but it must not be treated as a normal ranked candidate unless explicitly testing an ETF/benchmark universe design.

## Horizons

Supported horizons should remain:

```text
5 trading days
10 trading days
20 trading days
50 trading days
```

Primary research focus:

```text
10 trading days
20 trading days
```

Reason:

```text
5d  -> may be noisy and short-term
10d / 20d -> closest to the intended swing-trading watchlist use case
50d -> useful later, but more of a longer-horizon regime/factor test
```

The all-horizons runner can remain, but conclusions should focus on 10d and 20d first.

## Universe

Initial target-mode comparison should use:

```text
large_mega_cap_stocks
```

Reason:

```text
- cleaner data
- stronger liquidity
- longer histories
- better match to SPY-relative benchmarking
- easier to interpret than the old mixed universe
```

The old `current_mixed` universe should remain available for comparison only.

The `broad_sector_etfs` universe is a separate research question. It is more about sector/style rotation than individual stock selection.

## Features

The first ranking-target experiment should use the current feature contract.

Do not add new technical indicators in the first implementation.

Reason:

```text
If target mode and features change at the same time, it becomes harder to know which change caused the result.
```

Current feature groups include:

```text
raw OHLCV fields
moving averages
daily return
volatility
RSI
MACD
OBV
volume moving averages
Ichimoku-style fields
Bollinger Bands
ATR
stochastic oscillator
absolute trailing momentum
SPY-relative trailing momentum
```

Feature/indicator research should come after the target-mode comparison.

## Leakage Rules

The ranking target introduces leakage risks.

Future-return fields must never become model features:

```text
raw_forward_return
benchmark_forward_return
excess_forward_return
targetReturns
beat_benchmark_target
excess_return_rank_pct_by_date
top_quintile_target
ranking_train_sample
```

Rules:

1. Forward returns are labels/evaluation metadata only.
2. Cross-sectional ranking labels are labels, not features.
3. Ranking labels must be computed within each `prediction_date`.
4. Train/validation/test splits must remain chronological.
5. Embargo gaps must remain between train, validation, and test periods.
6. Scalers must be fit on training rows only.
7. SPY forward return must not be used as a model feature.
8. Past-looking SPY-relative momentum is allowed as a feature.

Allowed example:

```text
relative_momentum_10d =
  stock_trailing_momentum_10d - spy_trailing_momentum_10d
```

This is allowed because it uses past-looking same-date information.

Labels may be computed before splitting if they are strictly label/metadata fields and never used as features. Training must only fit on rows from the training date range.

## Splitting Strategy

Use global chronological splits by `prediction_date`.

Responsibilities:

```text
target creation: ticker-aware forward shifts
SPY alignment: date-based join
ranking label creation: same-date cross-sectional ranking
splitting: global chronological date split with embargo
evaluation: same-date ranked selection
```

Preferred split structure:

```text
train dates -> embargo -> validation dates -> embargo -> test dates
```

The model must not train on rows whose prediction dates belong to validation, test, or embargo windows.

## Training Dataset For Ranking Classifier

For the first implementation, classifier fitting should use only:

```text
ranking_train_sample == True
```

Classifier target:

```text
top_quintile_target
```

Training:

```text
top/bottom quintile rows only
```

Validation/test evaluation:

```text
all candidate rows per date
```

This distinction is important. The classifier trains on cleaner extreme labels, but evaluation still checks whether it ranks the full candidate universe well.

## Primary Evaluation

Evaluation should match the intended watchlist behavior.

For each prediction date in the validation/test split:

1. Score every available candidate stock.
2. Rank candidates by model score.
3. Select Top-N.
4. Compare selected baskets against baselines.

Recommended buckets:

```text
top_5
top_10
top_20
```

If fewer than N candidates exist:

```text
selected_count = min(N, available_candidates_that_date)
```

## Baselines

Compare against:

```text
random selection
equal-weight same-date universe
SPY benchmark raw return
horizon-matched raw momentum
horizon-matched SPY-relative momentum
current excess-return regression target mode
```

Important note:

Within a single prediction date, raw momentum and SPY-relative momentum rank stocks the same way if SPY momentum is subtracted as the same constant from every stock.

SPY-relative momentum is still useful for semantic clarity, but it is not a separate same-date ranking signal from raw momentum.

## Primary Metrics

For each Top-N bucket, report:

```text
date_count
average_selected_raw_forward_return
average_selected_benchmark_forward_return
average_selected_excess_forward_return
beat_benchmark_rate
average_equal_weight_universe_excess_return
model_minus_random_excess_return
model_minus_momentum_excess_return
average_candidate_count_per_date
```

Most important metrics:

```text
model_minus_momentum_excess_return
average_selected_excess_forward_return
beat_benchmark_rate
```

A model should not be considered successful just because it beats random. The key test is whether it improves on horizon-matched momentum and the current regression target.

## Ranking Diagnostics

Add rank-specific diagnostics if they are simple to implement.

Recommended:

```text
daily_spearman_rank_ic
mean_rank_ic
median_rank_ic
rank_ic_positive_rate
selected_pick_realized_rank_distribution
```

Rank IC definition:

```text
For each prediction date:
  compute Spearman correlation between model score and realized excess_forward_return
```

This directly measures whether higher model scores correspond to better realized future relative performance.

These diagnostics are useful, but they should not distract from the main Top-N basket results.

## Classifier Diagnostics

Standard classifier metrics are secondary.

Useful diagnostics:

```text
accuracy
precision
recall
f1
confusion_matrix
positive_class_rate
predicted_positive_rate
probability_summary
```

Do not overemphasize accuracy. The model’s job is not to classify every row correctly. The model’s job is to rank the strongest candidates near the top.

## Desired Console Summary

Console output should stay concise.

Example:

```text
Target Mode Comparison:
target_mode=excess_return
  top_5 model excess=..., model minus momentum=...
  top_10 model excess=..., model minus momentum=...

target_mode=cross_sectional_top_bottom
  top_5 model excess=..., model minus momentum=...
  top_10 model excess=..., model minus momentum=...
```

Verbose diagnostics should remain in `training.log`.

## Desired Prediction Output

Later, prediction output should support ranked watchlist fields:

```text
ticker
rank_in_universe
probability_of_top_quintile_outperformance
predicted_excess_return_vs_spy optional
raw_momentum_score
relative_momentum_score
label
```

Possible labels:

```text
candidate
watch
avoid
```

Do not implement live trading or automated order execution.

## Likely Files To Change

### `src/data/data_prep.py`

Likely changes:

```text
add same-date excess-return percentile ranks
add top_quintile_target
add ranking_train_sample
preserve ranking metadata through splits
keep future-return/ranking label fields out of model features
```

### `src/features/feature_contract.py`

Likely changes:

```text
add ranking target/metadata columns to non-feature metadata lists
ensure ranking labels cannot overlap model features
```

### `src/train/trainer.py`

Likely changes:

```text
add --target-mode
support current excess-return regression target
support cross-sectional top/bottom classifier target
train ranking classifier on ranking_train_sample rows
score all validation/test candidates
preserve existing regression path for comparison
save target-mode metadata
```

### `src/train/evaluation/*`

Likely changes:

```text
rank candidates by classifier probability
reuse Top-N basket reports where possible
add rank-IC diagnostics if practical
compare against momentum baselines
```

### `src/inference/predictor.py`

Likely later changes:

```text
return top-quintile probability
support ranked watchlist output
keep predicted excess return optional/secondary
```

## Tests

Add focused tests before or alongside implementation.

### Data preparation tests

Verify:

```text
ranking labels are computed within each prediction_date
top quintile rows receive label 1
bottom quintile rows receive label 0
middle rows are excluded from classifier training
SPY is not included as a candidate row
ranking labels are not model features
```

### Trainer tests

Verify:

```text
--target-mode excess_return keeps current behavior
--target-mode cross_sectional_top_bottom trains classifier on ranking labels
classifier fitting uses only ranking_train_sample rows
validation/test scoring uses all candidate rows
metadata records target mode
```

### Evaluation tests

Verify:

```text
candidate rows are ranked by classifier probability
Top-N basket returns are calculated correctly
model-minus-momentum is calculated correctly
rank IC calculation works if implemented
selected pick realized rank distribution works if implemented
```

## Metadata

Saved model metadata should make the target mode explicit:

```text
target_mode = "cross_sectional_top_bottom"
benchmark_ticker = "SPY"
prediction_days = N
ranking_target = "top_20_vs_bottom_20_by_excess_forward_return"
ranking_group_key = "prediction_date"
primary_model_score = "probability_of_top_quintile_outperformance"
```

For the current baseline path:

```text
target_mode = "excess_return"
primary_model_score = "predicted_excess_return"
```

## Staged Implementation Plan

### Stage 1: Target labels and tests

1. Add tests for same-date ranking labels.
2. Add `excess_return_rank_pct_by_date`.
3. Add `top_quintile_target`.
4. Add `ranking_train_sample`.
5. Ensure ranking labels are metadata/targets, not model features.

### Stage 2: Target-mode CLI

1. Add `--target-mode`.
2. Keep `excess_return` as a baseline mode.
3. Add `cross_sectional_top_bottom` as experimental mode.
4. Record target mode in metadata/logs.

### Stage 3: Ranking classifier training

1. Train ranking classifier on top/bottom quintile rows.
2. Score all validation/test candidate rows.
3. Rank by top-quintile probability.
4. Keep the current regressor available for comparison.

### Stage 4: Evaluation comparison

1. Compare target modes on `large_mega_cap_stocks`.
2. Focus on 10d and 20d.
3. Focus on Top-5 and Top-10.
4. Compare against horizon-matched momentum.
5. Apply the adoption rule before declaring the ranking target better.

### Stage 5: Later feature work

If the ranking target helps, improve features for that target mode.

Potential later feature work:

```text
remove raw price-level features
add cross-sectional feature ranks/z-scores
add volatility-adjusted momentum
add moving-average distance features
add drawdown features
add rolling beta/correlation to SPY
add relative volume features
```

Do not mix this feature work into the first target-mode experiment.

## Explicit Non-Goals

Do not add:

```text
live trading
automated execution
options strategies
sector adjustment
beta adjustment
new technical indicators
large hyperparameter sweeps
ticker-list optimization to chase backtest results
major infrastructure rewrite
```

The near-term goal is narrow:

```text
Compare the current SPY-relative excess-return regression target against
a cross-sectional top/bottom quintile ranking classifier target.
```

Success means the ranking target improves practical Top-N evaluation, especially versus horizon-matched momentum, on the cleaner `large_mega_cap_stocks` universe.
