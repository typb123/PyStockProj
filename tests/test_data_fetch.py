"""Tests for explicit Yahoo Finance market-data semantics."""

import pandas as pd

import src.data.data_fetch as data_fetch


def test_fetch_stock_data_requests_explicit_adjusted_ohlcv(monkeypatch):
    calls = []
    expected = pd.DataFrame({"Close": [100.0]})

    class FakeTicker:
        def history(self, **kwargs):
            calls.append(kwargs)
            return expected

    monkeypatch.setattr(data_fetch.yf, "Ticker", lambda ticker: FakeTicker())

    result = data_fetch.fetch_stock_data("AAPL", period="max")

    assert result is expected
    assert calls == [{"period": "max", "auto_adjust": True}]


def test_yfinance_data_provenance_records_explicit_adjustment_policy():
    provenance = data_fetch.get_yfinance_data_provenance("max")

    assert provenance["provider"] == "Yahoo Finance"
    assert provenance["library"] == "yfinance"
    assert provenance["library_version"]
    assert provenance["auto_adjust"] is True
    assert provenance["price_convention"] == "yfinance_auto_adjusted_ohlcv"
    assert provenance["training_period"] == "max"
