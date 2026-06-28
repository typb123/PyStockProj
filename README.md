# PyStockProj

PyStockProj is a research-oriented stock selection project. The current branch
frames the model around SPY-relative ranked watchlists.

## Current Pipeline

- Fetches historical daily OHLCV data with `yfinance`.
- Builds technical indicators per ticker before concatenating ticker data.
- Creates ticker-aware forward returns over a configurable trading-day horizon.
- Aligns each stock row to SPY forward return by prediction date.
- Trains on stock forward return minus SPY forward return.
- Uses global chronological train/validation/test splits with embargo gaps.
- Fits preprocessing scalers on training data only.
- Trains three artifacts:
  - Linear Regression baseline for excess return
  - XGBoost classifier for whether a stock beats SPY
  - XGBoost regressor for SPY-relative excess return

The default horizon is 10 trading days. The trainer also supports explicit
horizons and an all-horizons research run.

## Evaluation

Training logs test-set diagnostics to `training.log`, including:

- beat-benchmark classification reports
- excess-return regression reports
- probability threshold and validation-selected threshold reports
- regressor ranking, quantile, and correlation reports
- per-date Top-N ranked-selection reports
- equal-weight Top-N basket backtest summaries
- random, same-date universe, SPY benchmark, momentum, and relative-momentum baselines

Top-N reports rank candidates within each prediction date. SPY is the benchmark,
not a normal candidate in ranked stock-selection evaluation.

## Usage

Run the default 10-trading-day training workflow:

```bash
python -m src.train.trainer
```

Run all configured research horizons:

```bash
python -m src.train.trainer --all-horizons
```

Useful runtime controls:

```bash
python -m src.train.trainer --prediction-days 10 --period 10y --random-trials 20
python -m src.train.trainer --all-horizons --period max --parallel-random-trials --random-trial-workers 4
```

## Scripts

- `scripts/run_medium_signal_check.py` runs a fixed 10-ticker smoke check and
  prints an AAPL prediction after training.
- `scripts/run_current_model_experiments.py` is kept for historical reference;
  the trainer CLI is the authoritative current workflow.

## Artifacts

Training writes model and preprocessing artifacts under `models/`, including:

- model files
- linear-model scaler
- fitted `DataPreparator`
- feature names
- model metadata used by prediction-time feature alignment and target semantics

Generated artifacts, caches, and logs are intentionally ignored by Git.
