"""Console behavior tests for the Rank-NDCG ranked-watchlist app."""

import builtins

import src.app as app
from src.inference.ranked_watchlist import RankedWatchlistError


def sample_result(*, source_type="configured_universe"):
    return {
        "bundle_id": "bundle-123",
        "prediction_days": 10,
        "as_of_date": "2026-08-28",
        "candidate_source": {
            "type": source_type,
            "universe_name": (
                "large_mega_cap_stocks"
                if source_type == "configured_universe"
                else None
            ),
        },
        "ranked_count": 2,
        "skipped_count": 1,
        "skipped_tickers": [
            {"ticker": "SPY", "reason": "benchmark ticker is excluded"}
        ],
        "ranked_rows": [
            {"rank": 1, "ticker": "NVDA", "ranker_score": 0.42},
            {"rank": 2, "ticker": "AAPL", "ranker_score": 0.13},
        ],
    }


def install_inputs(monkeypatch, responses):
    answers = iter(responses)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))


def test_main_menu_requires_explicit_horizon_and_runs_configured_universe(
    monkeypatch,
    capsys,
):
    calls = []
    install_inputs(monkeypatch, ["invalid", "1", "1", "n"])
    monkeypatch.setattr(
        app,
        "rank_candidate_universe",
        lambda prediction_days, **kwargs: (
            calls.append((prediction_days, kwargs)) or sample_result()
        ),
    )

    app.main_menu()

    assert calls == [(10, {"universe": "large_mega_cap_stocks"})]
    output = capsys.readouterr().out
    assert "Invalid horizon choice" in output
    assert "Horizon: 10 trading days" in output
    assert "Configured universe: large_mega_cap_stocks" in output


def test_main_menu_sends_raw_custom_tickers_to_backend_and_displays_ranking(
    monkeypatch,
    capsys,
):
    calls = []
    install_inputs(monkeypatch, ["2", "2", " nvda, AAPL, spy ", "n"])
    monkeypatch.setattr(
        app,
        "rank_candidate_universe",
        lambda prediction_days, **kwargs: (
            calls.append((prediction_days, kwargs))
            or sample_result(source_type="custom")
        ),
    )

    app.main_menu()

    assert calls == [(20, {"tickers": [" nvda", " AAPL", " spy "]})]
    output = capsys.readouterr().out
    assert "Candidate source: Custom ticker list" in output
    assert "Rank | Ticker | Rank-NDCG score" in output
    assert (
        "Rank-NDCG score is a ranking score, not a predicted return or probability."
        in output
    )
    assert "SPY: benchmark ticker is excluded" in output
    assert "Bundle ID: bundle-123" in output


def test_main_menu_handles_expected_backend_errors_without_traceback(
    monkeypatch,
    capsys,
):
    install_inputs(monkeypatch, ["1", "1", "n"])
    monkeypatch.setattr(
        app,
        "rank_candidate_universe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RankedWatchlistError("no compatible current bundle")
        ),
    )

    app.main_menu()

    output = capsys.readouterr().out
    assert "Unable to generate ranked watchlist: no compatible current bundle" in output
    assert "Traceback" not in output


def test_console_app_does_not_depend_on_legacy_single_stock_predictor():
    assert "predict_spy_relative_return" not in app.__dict__
