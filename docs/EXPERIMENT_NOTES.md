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
