# Data Audit

## Current data source

The project currently fetches historical daily OHLCV data through `yfinance`:

    yf.Ticker(ticker).history(period=period)

## Current price field

Forward returns are currently calculated from the `Close` column:

    future_close = df.groupby("Ticker", sort=False)["Close"].shift(-prediction_days)
    raw_forward_return = (future_close - Close) / Close

Audit question: confirm whether `Close` from `yfinance.Ticker.history()` is adjusted or raw under the current yfinance defaults. The code currently does not make this explicit.

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
2. raw OHLCV data is normalized and optionally cached to CSV,
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

- Raw price-level columns (`Open`, `High`, `Low`, `Close`, `Volume`) are included as model features.
- This may let the model learn price level, liquidity, or ticker-style proxies rather than only return dynamics.
- This is not necessarily wrong, but it should be tested later with feature ablations.
- Price adjustment behavior still needs to be made explicit because targets and indicators use `Close`.

Future audit check:

- Compare results with raw OHLCV price-level features removed or transformed into normalized return/liquidity features.
- Confirm yfinance adjustment behavior and decide whether to explicitly request adjusted or unadjusted prices.

## Current conclusion

The data is usable for research and engineering development, but it has not yet been audited enough to support strong trading conclusions.

Before treating model results as meaningful trading evidence, the project should verify:

1. whether `Close` is adjusted or raw,
2. whether the universe should be stocks-only, ETFs-only, or explicitly mixed,
3. whether liquidity filters are needed,
4. whether ticker survivorship/selection bias is acceptable for the project goal,
5. how many rows are dropped per ticker and why.
