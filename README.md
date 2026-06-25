# PyStockProj

PyStockProj is a research-oriented stock return prediction project. The current
baseline predicts raw future returns over a default 10-trading-day horizon using
technical indicators built from daily OHLCV data.

## Current Pipeline

- Fetches historical price data with `yfinance`.
- Builds technical indicators per ticker before concatenating ticker data.
- Creates future-return targets with ticker-aware shifts.
- Uses chronological train/validation/test splits with an embargo gap.
- Fits preprocessing scalers on training data only.
- Trains three artifacts:
  - Linear Regression baseline
  - XGBoost classifier for return direction
  - XGBoost regressor for return magnitude

## Evaluation

Training logs test-set diagnostics to `training.log`, including:

- baseline actual return and positive-return rate
- classifier probability threshold reports
- validation-selected threshold report
- regressor ranking/quantile report
- combined classifier/regressor signal report

The project is still a research baseline, not a trading system.

## Scripts

- `scripts/run_medium_signal_check.py` runs a fixed 10-ticker smoke check and prints
  an AAPL prediction after training.
- `scripts/run_current_model_experiments.py` runs named experiments against named
  ticker universes such as `small_10`, `core_25`, `core_50`, and `broad_200`.

## Artifacts

Training writes model and preprocessing artifacts under `models/`, including:

- model files
- linear-model scaler
- fitted `DataPreparator`
- feature names
- model metadata used by prediction-time feature alignment

Generated artifacts and logs are intentionally ignored by Git.
