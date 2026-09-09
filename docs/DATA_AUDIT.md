# Data Audit

## Current data source

The project currently fetches historical daily OHLCV data through `yfinance`
with explicit adjusted-price semantics:

    yf.Ticker(ticker).history(period=period, auto_adjust=True)

## Current price field

Forward returns are currently calculated from the adjusted `Close` column:

    future_close = df.groupby("Ticker", sort=False)["Close"].shift(-prediction_days)
    raw_forward_return = (future_close - Close) / Close

The historical `raw_forward_return` field name is retained for compatibility,
but it represents a provider-adjusted, total-return-like forward return rather
than a raw price return.

## Current universes

Training universes are manually defined in named config lists. The default is
`large_mega_cap_stocks`.

Known universe choices:

- `large_mega_cap_stocks`: the new default stock universe for testing stock selection.
- `broad_sector_etfs`: an ETF universe for testing sector/style rotation.
- `current_mixed`: the old random mixed universe retained for reference and comparison only.

SPY is included in fetched universes for benchmark construction, then excluded from candidate training rows and ranked outputs.

Mid-cap, small-cap, diverse-cap, and sector-specific stock universes are intentionally not implemented yet. They need cleaner construction rules before being added.

Main concern:

- The universes are modern hand-picked lists, not historically point-in-time investable universes.
- This creates survivorship/selection-bias risk.
- Results should not be interpreted as a clean historical test of a strategy that could have been run exactly the same way in the past.

## Benchmark handling

SPY-relative target construction:

- SPY rows are used to create date-aligned benchmark forward returns.
- Candidate stock rows are merged with same-date SPY forward returns.
- Candidate targets are calculated as stock forward return minus SPY forward return.
- SPY is excluded from candidate training rows after benchmark labels are built.

## Missing data handling

Rows missing required model features or target fields are dropped.

This is practical, but it may introduce bias if missingness is systematic across:

- newer tickers,
- delisted/problematic tickers,
- illiquid names,
- tickers with shorter histories,
- assets with irregular trading histories.

## Liquidity and tradability

Volume data is available in the fetched OHLCV frame when Yahoo provides it, but liquidity is not currently used to define or filter the training universe.

Open questions:

- Are low-liquidity names included?
- Are very new tickers included with short histories?
- Are ETFs and individual stocks being compared in a way that makes the ranked universe hard to interpret?
- Should future diagnostics use minimum average dollar volume filters?

## Fetch and preparation path

Current training data flow:

1. each ticker is fetched independently,
2. adjusted OHLCV data is normalized and optionally cached to CSV,
3. technical indicators are calculated per ticker before concatenation,
4. `prediction_date` and `Ticker` are added,
5. valid ticker frames are concatenated into one training frame.

Positive finding:

- Indicators are calculated before ticker-level concatenation, so rolling indicators should not leak across tickers.

Open concerns:

- Failed or empty ticker fetches are skipped.
- Skipped tickers are logged, but there is no persisted ticker coverage report.
- The model does not currently summarize per-ticker row counts, date ranges, missing rows, or dropped rows.
- No liquidity filter is applied before training.

Future audit check:

- Add a ticker coverage report showing fetched tickers, skipped tickers, date ranges, row counts, and post-feature/pre-target usable row counts.

## Technical indicator generation

Current indicator generation appears to be ticker-local and mostly past-looking.

Positive findings:

- Technical indicators are calculated separately for each ticker before concatenation.
- Rolling averages, volatility, RSI, MACD, Bollinger Bands, ATR, stochastic oscillator, and trailing momentum use current or past rows.
- The Ichimoku-style features are implemented to avoid using future rows.
- `Volume` survives feature generation and is included in the feature contract.

Open concerns:

- The model uses price-relative OHLC and normalized technical features rather
  than nominal price levels, so a common retrospective corporate-action scale
  factor cancels from model inputs.
- `Volume`, volume moving averages, and the 20-session rolling signed-volume
  feature remain in the contract; their liquidity and split-unit interpretation
  remains a separate research choice.
- Cumulative OBV was removed because its value depended on the first row of the
  supplied history: canonical `period=max` training and five-year serving could
  assign different values to the same ticker/date. The replacement sums
  `sign(Close.diff()) * Volume` across a trailing 20-session window and requires
  a complete window, so it is invariant to an earlier history start once that
  warmup is available.

Completed audit check:

- The corrected 10d and 20d walk-forward results must be rerun under the new
  rolling signed-volume contract before they can be canonical again. See the
  [experiment notes](EXPERIMENT_NOTES.md#volume-pressure-feature-parity-remediation).

## Current conclusion

The data is usable for research and engineering development, but it has not yet been audited enough to support strong trading conclusions.

The existing corrected 10d and 20d figures are provisional until the canonical
walk-forward evaluation is rerun with the rolling signed-volume feature.
Before treating model results as meaningful trading evidence, the project should
still resolve:

1. whether the universe should be stocks-only, ETFs-only, or explicitly mixed,
2. whether liquidity filters are needed,
3. whether ticker survivorship/selection bias is acceptable for the project goal,
4. how many rows are dropped per ticker and why.
