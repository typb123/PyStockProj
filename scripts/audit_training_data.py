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


def summarize_ticker(ticker: str, period: str, use_cache: bool = True) -> dict:
    """Fetch one ticker and summarize raw OHLCV coverage."""
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


def build_coverage_report(
    tickers: list[str],
    period: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build a ticker-level raw data coverage report."""
    return pd.DataFrame(
        [
            summarize_ticker(ticker, period=period, use_cache=use_cache)
            for ticker in tickers
        ]
    )


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
    report = build_coverage_report(
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

    fetched_count = int(report["fetched"].sum())
    skipped_count = int((~report["fetched"]).sum())
    print(f"Wrote {output_path}")
    print(f"Universe {args.universe}: requested {len(tickers)} tickers.")
    print(f"Fetched {fetched_count} tickers; skipped {skipped_count} tickers.")

    if skipped_count:
        skipped = report.loc[~report["fetched"], "ticker"].tolist()
        print(f"Skipped tickers: {skipped}")

    return output_path


if __name__ == "__main__":
    main()
