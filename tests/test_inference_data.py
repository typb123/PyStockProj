"""Tests for target-free completed-session inference feature preparation."""

import pandas as pd

import src.data.inference_data as inference_data


def test_completed_daily_history_excludes_current_date_before_cutoff():
    raw = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Volume": [1000.0, 1100.0],
        },
        index=pd.to_datetime(["2026-08-27", "2026-08-28"]),
    )

    completed = inference_data.completed_daily_history(
        raw,
        now="2026-08-28 16:29:59-04:00",
    )

    assert completed.index.tolist() == [pd.Timestamp("2026-08-27")]


def test_completed_daily_history_allows_current_date_after_cutoff():
    raw = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Volume": [1000.0, 1100.0],
        },
        index=pd.to_datetime(["2026-08-27", "2026-08-28"]),
    )

    completed = inference_data.completed_daily_history(
        raw,
        now="2026-08-28 16:30:00-04:00",
    )

    assert completed.index.tolist() == [
        pd.Timestamp("2026-08-27"),
        pd.Timestamp("2026-08-28"),
    ]


def test_prepare_completed_ticker_features_is_target_free(
    monkeypatch,
):
    raw = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Volume": [1000.0, 1100.0],
        },
        index=pd.to_datetime(["2026-08-26", "2026-08-27"]),
    )
    monkeypatch.setattr(
        inference_data,
        "fetch_stock_data",
        lambda ticker, period="5y": raw,
    )
    monkeypatch.setattr(
        inference_data,
        "calculate_data",
        lambda data: data.assign(momentum_5d=[0.10, 0.20]),
    )

    result = inference_data.prepare_completed_ticker_features(
        "AAPL",
        now="2026-08-28 12:00:00-04:00",
    )

    assert result["Ticker"].tolist() == ["AAPL", "AAPL"]
    assert result["prediction_date"].tolist() == [
        pd.Timestamp("2026-08-26"),
        pd.Timestamp("2026-08-27"),
    ]
    assert not {
        "targetReturns",
        "raw_forward_return",
        "benchmark_forward_return",
        "excess_forward_return",
        "excess_return_rank_pct_by_date",
    }.intersection(result.columns)


def test_prepare_completed_ticker_features_rejects_no_completed_bars(monkeypatch):
    current_day = pd.DataFrame(
        {
            "Close": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Volume": [1000.0],
        },
        index=pd.to_datetime(["2026-08-28"]),
    )
    monkeypatch.setattr(
        inference_data,
        "fetch_stock_data",
        lambda ticker, period="5y": current_day,
    )

    try:
        inference_data.prepare_completed_ticker_features(
            "AAPL",
            now="2026-08-28 16:29:59-04:00",
        )
    except inference_data.InferenceDataError as error:
        assert "no completed daily bars" in str(error)
    else:
        raise AssertionError("Expected incomplete current-date data to be rejected")
