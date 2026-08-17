"""Tests for training-universe data audit CLI behavior."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src.config import DEFAULT_TRAINING_UNIVERSE, get_training_tickers

AUDIT_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/audit_training_data.py"
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_training_data",
    AUDIT_SCRIPT_PATH,
)
audit = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit)


def _raw_ohlcv(dates):
    frame = pd.DataFrame(
        {
            "Close": range(100, 100 + len(dates)),
            "Volume": range(1000, 1000 + len(dates)),
        },
        index=pd.to_datetime(dates),
    )
    frame.index.name = "Date"
    return frame


def _build_reports_from_frames(monkeypatch, frames):
    monkeypatch.setattr(
        audit,
        "fetch_raw_ticker_data",
        lambda ticker, period="10y", use_cache=True: frames[ticker].copy(),
    )
    return audit.build_coverage_reports(list(frames), period="max")


def test_audit_parse_args_defaults_to_default_training_universe():
    args = audit.parse_args([])

    assert args.period == "10y"
    assert args.universe == DEFAULT_TRAINING_UNIVERSE


def test_audit_parse_args_rejects_unknown_training_universe():
    with pytest.raises(SystemExit):
        audit.parse_args(["--universe", "bad_name"])


def test_audit_main_uses_selected_universe_and_universe_output_filename(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_fetch_raw_ticker_data(ticker, period="10y", use_cache=True):
        calls.append((ticker, period, use_cache))
        data = pd.DataFrame(
            {"Close": [100.0, 101.0], "Volume": [1000, 1100]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        data.index.name = "Date"
        return data

    monkeypatch.setattr(audit, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(audit, "fetch_raw_ticker_data", fake_fetch_raw_ticker_data)

    output_path = audit.main(
        ["--period", "1y", "--universe", "broad_sector_etfs"]
    )

    expected_tickers = get_training_tickers("broad_sector_etfs")
    assert output_path == tmp_path / "ticker_coverage_broad_sector_etfs_1y.csv"
    assert calls == [(ticker, "1y", True) for ticker in expected_tickers]

    report = pd.read_csv(output_path)
    assert report["ticker"].tolist() == expected_tickers
    assert report["fetched"].all()
    yearly_output_path = tmp_path / "yearly_candidate_coverage_broad_sector_etfs_1y.csv"
    assert yearly_output_path.exists()


def test_audit_main_honors_no_cache_and_explicit_output_path(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_fetch_raw_ticker_data(ticker, period="10y", use_cache=True):
        calls.append((ticker, period, use_cache))
        data = pd.DataFrame(
            {"Close": [100.0], "Volume": [1000]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        data.index.name = "Date"
        return data

    output_path = tmp_path / "custom.csv"
    monkeypatch.setattr(audit, "fetch_raw_ticker_data", fake_fetch_raw_ticker_data)

    result = audit.main(
        [
            "--period",
            "6mo",
            "--universe",
            "current_mixed",
            "--no-cache",
            "--output",
            str(output_path),
        ]
    )

    assert result == output_path
    assert calls == [
        (ticker, "6mo", False) for ticker in get_training_tickers("current_mixed")
    ]
    assert output_path.exists()


def test_audit_reports_complete_spy_aligned_trading_dates(monkeypatch):
    frames = {
        "SPY": _raw_ohlcv(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "AAA": _raw_ohlcv(["2024-01-02", "2024-01-03", "2024-01-04"]),
    }

    report, _ = _build_reports_from_frames(monkeypatch, frames)
    row = report.set_index("ticker").loc["AAA"]

    assert row["expected_trading_days"] == 3
    assert row["actual_trading_days"] == 3
    assert row["missing_trading_days"] == 0
    assert row["missing_trading_day_rate"] == 0.0
    assert row["largest_internal_gap_trading_days"] == 0
    assert row["stale_end_days_vs_spy"] == 0


def test_audit_reports_internal_missing_spy_trading_dates(monkeypatch):
    frames = {
        "SPY": _raw_ohlcv(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ]
        ),
        "AAA": _raw_ohlcv(["2024-01-02", "2024-01-05", "2024-01-08"]),
    }

    report, _ = _build_reports_from_frames(monkeypatch, frames)
    row = report.set_index("ticker").loc["AAA"]

    assert row["expected_trading_days"] == 5
    assert row["actual_trading_days"] == 3
    assert row["missing_trading_days"] == 2
    assert row["missing_trading_day_rate"] == 0.4
    assert row["largest_internal_gap_trading_days"] == 2


def test_audit_does_not_penalize_dates_before_ticker_history(monkeypatch):
    frames = {
        "SPY": _raw_ohlcv(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        ),
        "AAA": _raw_ohlcv(["2024-01-04", "2024-01-05"]),
    }

    report, _ = _build_reports_from_frames(monkeypatch, frames)
    row = report.set_index("ticker").loc["AAA"]

    assert row["expected_trading_days"] == 2
    assert row["actual_trading_days"] == 2
    assert row["missing_trading_days"] == 0
    assert row["stale_end_days_vs_spy"] == 0


def test_audit_reports_stale_end_days_against_spy(monkeypatch):
    frames = {
        "SPY": _raw_ohlcv(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        ),
        "AAA": _raw_ohlcv(["2024-01-02", "2024-01-03", "2024-01-04"]),
    }

    report, _ = _build_reports_from_frames(monkeypatch, frames)
    row = report.set_index("ticker").loc["AAA"]

    assert row["missing_trading_days"] == 0
    assert row["stale_end_days_vs_spy"] == 1


def test_audit_yearly_candidate_coverage_excludes_spy(monkeypatch):
    frames = {
        "SPY": _raw_ohlcv(["2023-01-03", "2023-12-29", "2024-01-02"]),
        "AAA": _raw_ohlcv(["2023-01-03"]),
        "BBB": _raw_ohlcv(["2023-12-29", "2024-01-02"]),
    }

    _, yearly_report = _build_reports_from_frames(monkeypatch, frames)
    yearly_report = yearly_report.set_index("year")

    assert yearly_report.loc[2023, "requested_candidate_tickers"] == 2
    assert yearly_report.loc[2023, "candidate_tickers_with_usable_raw_data"] == 2
    assert yearly_report.loc[2023, "candidate_universe_coverage_fraction"] == 1.0
    assert yearly_report.loc[2024, "requested_candidate_tickers"] == 2
    assert yearly_report.loc[2024, "candidate_tickers_with_usable_raw_data"] == 1
    assert yearly_report.loc[2024, "candidate_universe_coverage_fraction"] == 0.5
