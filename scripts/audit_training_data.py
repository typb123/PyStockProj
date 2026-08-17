"""Generate a ticker-level data coverage audit for the training universe."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_TRAINING_UNIVERSE,
    TRAINING_UNIVERSES,
    get_training_tickers,
)
from src.train.trainer import PROJECT_ROOT, fetch_raw_ticker_data


REPORT_DIR = PROJECT_ROOT / "reports" / "data_audit"


def _normalized_trading_dates(data: pd.DataFrame) -> pd.DatetimeIndex:
    """Return unique normalized trading dates from one raw OHLCV frame."""
    if data.empty:
        return pd.DatetimeIndex([])

    dates = pd.DatetimeIndex(pd.to_datetime(data.index, errors="coerce"))
    dates = dates.dropna()
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    return pd.DatetimeIndex(dates.normalize().unique()).sort_values()


def _largest_missing_run(expected_dates, missing_dates) -> int:
    """Return the longest consecutive run of missing reference sessions."""
    missing_set = set(missing_dates)
    largest_run = 0
    current_run = 0
    for date in expected_dates:
        if date in missing_set:
            current_run += 1
            largest_run = max(largest_run, current_run)
        else:
            current_run = 0
    return largest_run


def _spy_aligned_coverage_metrics(
    data: pd.DataFrame,
    spy_trading_dates: pd.DatetimeIndex | None,
) -> dict:
    """Measure internal raw-data coverage against SPY trading sessions."""
    metrics = {
        "expected_trading_days": None,
        "actual_trading_days": None,
        "missing_trading_days": None,
        "missing_trading_day_rate": None,
        "largest_internal_gap_trading_days": None,
        "stale_end_days_vs_spy": None,
    }
    if data.empty or spy_trading_dates is None or spy_trading_dates.empty:
        return metrics

    ticker_dates = _normalized_trading_dates(data)
    if ticker_dates.empty:
        return metrics

    first_date = ticker_dates.min()
    last_date = ticker_dates.max()
    expected_dates = spy_trading_dates[
        (spy_trading_dates >= first_date) & (spy_trading_dates <= last_date)
    ]
    actual_dates = ticker_dates.intersection(expected_dates)
    missing_dates = expected_dates.difference(actual_dates)
    expected_count = int(len(expected_dates))
    missing_count = int(len(missing_dates))

    return {
        "expected_trading_days": expected_count,
        "actual_trading_days": int(len(actual_dates)),
        "missing_trading_days": missing_count,
        "missing_trading_day_rate": (
            float(missing_count / expected_count) if expected_count else None
        ),
        "largest_internal_gap_trading_days": _largest_missing_run(
            expected_dates,
            missing_dates,
        ),
        "stale_end_days_vs_spy": int((spy_trading_dates > last_date).sum()),
    }


def summarize_ticker(
    ticker: str,
    period: str,
    use_cache: bool = True,
    reference_trading_dates: pd.DatetimeIndex | None = None,
    data: pd.DataFrame | None = None,
) -> dict:
    """Fetch one ticker and summarize raw OHLCV and SPY-aligned coverage."""
    if data is None:
        data = fetch_raw_ticker_data(ticker, period=period, use_cache=use_cache)

    summary = {
        "ticker": ticker,
        "period": period,
        "fetched": not data.empty,
        "row_count": int(len(data)),
        "start_date": None,
        "end_date": None,
        "missing_close_rows": None,
        "missing_volume_rows": None,
        "zero_volume_rows": None,
        "average_close": None,
        "average_volume": None,
        "average_dollar_volume": None,
        "columns": "",
        **_spy_aligned_coverage_metrics(data, reference_trading_dates),
    }

    if data.empty:
        return summary

    summary["start_date"] = data.index.min().strftime("%Y-%m-%d")
    summary["end_date"] = data.index.max().strftime("%Y-%m-%d")
    summary["columns"] = ",".join(str(column) for column in data.columns)

    if "Close" in data.columns:
        close = pd.to_numeric(data["Close"], errors="coerce")
        summary["missing_close_rows"] = int(close.isna().sum())
        summary["average_close"] = float(close.mean()) if close.notna().any() else None
    else:
        close = pd.Series(dtype=float)
        summary["missing_close_rows"] = None

    if "Volume" in data.columns:
        volume = pd.to_numeric(data["Volume"], errors="coerce")
        summary["missing_volume_rows"] = int(volume.isna().sum())
        summary["zero_volume_rows"] = int((volume == 0).sum())
        summary["average_volume"] = float(volume.mean()) if volume.notna().any() else None
    else:
        volume = pd.Series(dtype=float)
        summary["missing_volume_rows"] = None
        summary["zero_volume_rows"] = None

    if not close.empty and not volume.empty:
        dollar_volume = close * volume
        summary["average_dollar_volume"] = (
            float(dollar_volume.mean()) if dollar_volume.notna().any() else None
        )

    return summary


def build_yearly_candidate_coverage_report(
    ticker_data: dict[str, pd.DataFrame],
    requested_tickers: list[str],
    spy_trading_dates: pd.DatetimeIndex | None,
) -> pd.DataFrame:
    """Summarize non-SPY requested tickers represented on SPY dates by year."""
    columns = [
        "year",
        "requested_candidate_tickers",
        "candidate_tickers_with_usable_raw_data",
        "candidate_universe_coverage_fraction",
    ]
    candidate_tickers = [ticker for ticker in requested_tickers if ticker != "SPY"]
    if spy_trading_dates is None or spy_trading_dates.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for year in sorted(spy_trading_dates.year.unique()):
        reference_dates = spy_trading_dates[spy_trading_dates.year == year]
        represented_ticker_count = 0
        for ticker in candidate_tickers:
            ticker_dates = _normalized_trading_dates(
                ticker_data.get(ticker, pd.DataFrame())
            )
            if not ticker_dates.intersection(reference_dates).empty:
                represented_ticker_count += 1
        requested_candidate_count = len(candidate_tickers)
        rows.append(
            {
                "year": int(year),
                "requested_candidate_tickers": requested_candidate_count,
                "candidate_tickers_with_usable_raw_data": represented_ticker_count,
                "candidate_universe_coverage_fraction": (
                    float(represented_ticker_count / requested_candidate_count)
                    if requested_candidate_count
                    else None
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_coverage_reports(
    tickers: list[str],
    period: str,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ticker-level and yearly raw-data coverage reports."""
    ticker_data = {
        ticker: fetch_raw_ticker_data(ticker, period=period, use_cache=use_cache)
        for ticker in tickers
    }
    spy_data = ticker_data.get("SPY")
    if spy_data is None:
        spy_data = fetch_raw_ticker_data("SPY", period=period, use_cache=use_cache)
    spy_trading_dates = _normalized_trading_dates(spy_data)
    reference_dates = spy_trading_dates if not spy_trading_dates.empty else None

    ticker_report = pd.DataFrame(
        [
            summarize_ticker(
                ticker,
                period=period,
                use_cache=use_cache,
                reference_trading_dates=reference_dates,
                data=ticker_data[ticker],
            )
            for ticker in tickers
        ]
    )
    yearly_report = build_yearly_candidate_coverage_report(
        ticker_data,
        tickers,
        reference_dates,
    )
    return ticker_report, yearly_report


def build_coverage_report(
    tickers: list[str],
    period: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build a ticker-level raw data coverage report."""
    ticker_report, _ = build_coverage_reports(tickers, period, use_cache)
    return ticker_report


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="10y", help="YFinance period to audit.")
    parser.add_argument(
        "--universe",
        choices=sorted(TRAINING_UNIVERSES),
        default=DEFAULT_TRAINING_UNIVERSE,
        help=f"Training universe to audit. Defaults to {DEFAULT_TRAINING_UNIVERSE}.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass local yfinance cache and refetch raw data.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path. Defaults to reports/data_audit.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    """Generate and save the training data coverage report."""
    args = parse_args(argv)
    tickers = get_training_tickers(args.universe)
    report, yearly_report = build_coverage_reports(
        tickers,
        period=args.period,
        use_cache=not args.no_cache,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORT_DIR / f"ticker_coverage_{args.universe}_{args.period}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    yearly_output_path = output_path.parent / (
        f"yearly_candidate_coverage_{args.universe}_{args.period}.csv"
    )
    yearly_report.to_csv(yearly_output_path, index=False)

    fetched_count = int(report["fetched"].sum())
    skipped_count = int((~report["fetched"]).sum())
    print(f"Wrote {output_path}")
    print(f"Wrote {yearly_output_path}")
    print(f"Universe {args.universe}: requested {len(tickers)} tickers.")
    print(f"Fetched {fetched_count} tickers; skipped {skipped_count} tickers.")

    if skipped_count:
        skipped = report.loc[~report["fetched"], "ticker"].tolist()
        print(f"Skipped tickers: {skipped}")

    return output_path


if __name__ == "__main__":
    main()
