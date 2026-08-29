# PyStockProj

PyStockProj is a cross-sectional stock-ranking ML research project built around
XGBoost learning-to-rank (`rank:ndcg`). It pairs chronological walk-forward
evaluation with a tested, versioned artifact and same-date inference pipeline.

## What it does

- Trains an XGBoost learning-to-rank model with the `rank:ndcg` objective.
- Scores stocks **relative to one another on a shared prediction date** to
  produce a ranked watchlist; a score is not a predicted return or probability.
- Uses SPY as the benchmark when constructing forward excess-return targets.
- Serves explicit **10- and 20-trading-day** ranked-watchlist horizons.
- Supports the configured `large_mega_cap_stocks` research universe and custom
  ticker lists (with SPY excluded from candidates).
- Enforces one shared SPY-anchored date across all ranked candidates at
  inference time.

The training pipeline includes chronological, embargoed walk-forward
evaluation. Model artifacts are published as isolated, versioned bundles with
feature contracts and checksum validation.

## Research snapshot

The strongest canonical evidence is the max-history expanding walk-forward
evaluation: 28 annual folds, with test years from 1999 through partial 2026,
on the configured large/mega-cap universe. Top-5 aggregate results:

| Horizon | Model excess vs. SPY | Model minus momentum |
| --- | ---: | ---: |
| 10 trading days | +1.19% | +0.67% |
| 20 trading days | +2.28% | +0.80% |

These are historical walk-forward research results, not promises of trading
profitability. The primary limitation is survivorship/selection bias: the
historical universe projects a contemporary large/mega-cap stock list backward,
rather than using point-in-time historical membership.

## Run

Requires Python 3.10+.

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements-dev.txt

# Train and publish a 10-day Rank-NDCG bundle. Replace 10 with 20 to train the
# 20-day serving bundle; the app requires a matching bundle for its selection.
./venv/bin/python -m src.train.trainer \
  --prediction-days 10 \
  --period max \
  --target-mode cross_sectional_rank_ndcg \
  --universe large_mega_cap_stocks

# Run the interactive ranked-watchlist application (choose 10d or 20d).
./venv/bin/python -m src.app

# Run the max-history 10-day walk-forward evaluation; replace 10 with 20 for
# the 20-day evaluation.
./venv/bin/python -m src.train.trainer \
  --prediction-days 10 \
  --period max \
  --universe large_mega_cap_stocks \
  --target-mode cross_sectional_rank_ndcg \
  --walk-forward

# Test.
./venv/bin/python -m pytest -q
```

Generated model artifacts live under `models/` and are intentionally not
committed. Train the relevant horizon before using the application.

## Further reading

- [Experiment notes](docs/EXPERIMENT_NOTES.md) — detailed results and history.
- [Data audit](docs/DATA_AUDIT.md) — data sources, coverage, and limitations.
- [Ranking target design](docs/CROSS_SECTIONAL_RANKING_TARGET_DESIGN.md) —
  target and evaluation rationale.
