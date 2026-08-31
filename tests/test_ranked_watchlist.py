"""Tests for Rank-NDCG bundle loading and same-date watchlist ranking."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_contract import MODEL_FEATURE_COLUMNS
from src.train import artifacts
import src.inference.ranked_watchlist as ranked_watchlist


TEST_PREPROCESSOR_FEATURES = [
    "Close",
    "momentum_5d",
    "relative_momentum_5d",
]
TEST_RANKER_FEATURES = ["relative_momentum_5d", "Close"]


class RecordingScaler:
    def __init__(self):
        self.last_values = None

    def transform(self, values):
        self.last_values = np.asarray(values).copy()
        return values


class RecordingRanker:
    def __init__(self):
        self.last_columns = None
        self.last_values = None

    def predict(self, values):
        self.last_columns = list(values.columns)
        self.last_values = values.copy()
        return values.iloc[:, 0].to_numpy(dtype=float)


class StoredPreparator:
    def __init__(self):
        self.feature_columns = list(MODEL_FEATURE_COLUMNS)
        self.scalar = RecordingScaler()


class SavedRanker:
    def save_model(self, path):
        Path(path).write_text("ranker", encoding="utf-8")


class LoadedRanker:
    loaded_path = None

    def load_model(self, path):
        self.__class__.loaded_path = Path(path)


def make_loaded_bundle():
    scaler = RecordingScaler()
    ranker = RecordingRanker()
    manifest = {
        "bundle_id": "a" * 32,
        "training_population": {
            "universe_name": "large_mega_cap_stocks",
            "candidate_tickers": ["AAPL", "NVDA", "AMD"],
            "candidate_ticker_count": 3,
            "benchmark_ticker": "SPY",
        },
    }
    preparator = type("TestPreparator", (), {})()
    preparator.scalar = scaler
    return ranked_watchlist.LoadedRankNdcgBundle(
        bundle_dir=Path("unused"),
        manifest=manifest,
        data_preparator=preparator,
        model_metadata={},
        preprocessor_features=list(TEST_PREPROCESSOR_FEATURES),
        ranker_features=list(TEST_RANKER_FEATURES),
        ranker=ranker,
    )


def make_history(ticker, rows):
    return pd.DataFrame(
        {
            "Ticker": [ticker] * len(rows),
            "prediction_date": pd.to_datetime([row[0] for row in rows]),
            "Close": [row[1] for row in rows],
            "momentum_5d": [row[2] for row in rows],
        }
    )


def standard_histories():
    return {
        "SPY": make_history(
            "SPY",
            [("2026-08-26", 500.0, 0.04), ("2026-08-27", 505.0, 0.05)],
        ),
        "AAPL": make_history(
            "AAPL",
            [("2026-08-26", 200.0, 0.10), ("2026-08-27", 202.0, 0.20)],
        ),
        "NVDA": make_history(
            "NVDA",
            [("2026-08-26", 180.0, 0.15), ("2026-08-27", 185.0, 0.30)],
        ),
        "AMD": make_history(
            "AMD",
            [("2026-08-26", 150.0, 0.08), ("2026-08-27", 151.0, 0.10)],
        ),
    }


def install_ranking_fakes(monkeypatch, bundle, histories):
    monkeypatch.setattr(
        ranked_watchlist,
        "_load_rank_ndcg_bundle",
        lambda prediction_days: bundle,
    )
    monkeypatch.setattr(
        ranked_watchlist,
        "_prepare_histories",
        lambda tickers, now=None: {
            ticker: histories[ticker] for ticker in tickers
        },
    )


def test_custom_ranking_normalizes_deduplicates_excludes_spy_and_sorts(
    monkeypatch,
):
    bundle = make_loaded_bundle()
    histories = standard_histories()
    install_ranking_fakes(monkeypatch, bundle, histories)

    result = ranked_watchlist.rank_candidate_universe(
        20,
        tickers=[" nvda ", "AAPL", "NVDA", "spy", "amd"],
    )

    assert result["bundle_id"] == "a" * 32
    assert result["prediction_days"] == 20
    assert result["target_mode"] == "cross_sectional_rank_ndcg"
    assert result["as_of_date"] == "2026-08-27"
    assert result["candidate_source"] == {
        "type": "custom",
        "universe_name": None,
        "validation_context": "no_historical_validation_claim",
    }
    assert result["requested_count"] == 3
    assert result["ranked_count"] == 3
    assert result["skipped_count"] == 1
    assert result["skipped_tickers"] == [
        {
            "ticker": "SPY",
            "reason": "benchmark ticker is excluded from candidates",
        }
    ]
    assert result["ranked_rows"] == [
        {"rank": 1, "ticker": "NVDA", "ranker_score": 0.25},
        {
            "rank": 2,
            "ticker": "AAPL",
            "ranker_score": pytest.approx(0.15),
        },
        {"rank": 3, "ticker": "AMD", "ranker_score": 0.05},
    ]
    assert "not a probability or return" in result["score_semantics"]
    assert bundle.ranker.last_columns == TEST_RANKER_FEATURES


def test_named_universe_uses_configured_candidates(monkeypatch):
    bundle = make_loaded_bundle()
    bundle.manifest["training_population"].update(
        {
            "candidate_tickers": ["AAPL", "NVDA"],
            "candidate_ticker_count": 2,
        }
    )
    histories = standard_histories()
    monkeypatch.setattr(
        ranked_watchlist,
        "get_training_tickers",
        lambda universe: ["SPY", " aapl ", "AAPL", "nvda"],
    )
    install_ranking_fakes(monkeypatch, bundle, histories)

    result = ranked_watchlist.rank_candidate_universe(
        10,
        universe="large_mega_cap_stocks",
    )

    assert result["candidate_source"] == {
        "type": "configured_universe",
        "universe_name": "large_mega_cap_stocks",
        "validation_context": "configured_research_universe",
    }
    assert result["requested_count"] == 2
    assert result["ranked_count"] == 2
    assert all(row["ticker"] != "SPY" for row in result["ranked_rows"])


@pytest.mark.parametrize(
    "artifact_universe, artifact_candidates",
    [
        ("broad_sector_etfs", ["AAPL", "NVDA"]),
        ("large_mega_cap_stocks", ["AAPL", "NVDA", "AMD"]),
    ],
)
def test_named_universe_without_matching_artifact_provenance_has_no_validation_claim(
    monkeypatch,
    artifact_universe,
    artifact_candidates,
):
    bundle = make_loaded_bundle()
    bundle.manifest["training_population"].update(
        {
            "universe_name": artifact_universe,
            "candidate_tickers": artifact_candidates,
            "candidate_ticker_count": len(artifact_candidates),
        }
    )
    histories = standard_histories()
    monkeypatch.setattr(
        ranked_watchlist,
        "get_training_tickers",
        lambda universe: ["SPY", "AAPL", "NVDA"],
    )
    install_ranking_fakes(monkeypatch, bundle, histories)

    result = ranked_watchlist.rank_candidate_universe(
        10,
        universe="large_mega_cap_stocks",
    )

    assert result["candidate_source"] == {
        "type": "configured_universe",
        "universe_name": "large_mega_cap_stocks",
        "validation_context": "no_historical_validation_claim",
    }


def test_candidates_never_fall_back_to_a_different_date(monkeypatch):
    bundle = make_loaded_bundle()
    histories = standard_histories()
    histories["AMD"] = make_history("AMD", [("2026-08-26", 150.0, 0.08)])
    install_ranking_fakes(monkeypatch, bundle, histories)

    result = ranked_watchlist.rank_candidate_universe(
        20,
        tickers=["AAPL", "NVDA", "AMD"],
    )

    assert result["as_of_date"] == "2026-08-27"
    assert [row["ticker"] for row in result["ranked_rows"]] == ["NVDA", "AAPL"]
    assert result["skipped_tickers"] == [
        {
            "ticker": "AMD",
            "reason": "missing completed bar on SPY anchor date 2026-08-27",
        }
    ]


def test_partial_ticker_failure_does_not_fail_valid_ranking(monkeypatch):
    bundle = make_loaded_bundle()
    histories = standard_histories()
    histories["AMD"] = RuntimeError("provider unavailable")
    install_ranking_fakes(monkeypatch, bundle, histories)

    result = ranked_watchlist.rank_candidate_universe(
        20,
        tickers=["AAPL", "NVDA", "AMD"],
    )

    assert result["ranked_count"] == 2
    assert result["skipped_tickers"] == [
        {
            "ticker": "AMD",
            "reason": "data preparation failed: provider unavailable",
        }
    ]


def test_spy_relative_features_use_same_anchor_date_and_exact_feature_order(
    monkeypatch,
):
    bundle = make_loaded_bundle()
    histories = standard_histories()
    install_ranking_fakes(monkeypatch, bundle, histories)

    ranked_watchlist.rank_candidate_universe(
        20,
        tickers=["AAPL", "NVDA"],
    )

    assert bundle.ranker.last_columns == ["relative_momentum_5d", "Close"]
    np.testing.assert_allclose(
        bundle.ranker.last_values["relative_momentum_5d"],
        [0.15, 0.25],
    )
    np.testing.assert_allclose(
        bundle.data_preparator.scalar.last_values,
        [[202.0, 0.20, 0.15], [185.0, 0.30, 0.25]],
    )


@pytest.mark.parametrize("prediction_days", [5, 50, None])
def test_ranked_watchlist_rejects_unsupported_horizons(prediction_days):
    with pytest.raises(ranked_watchlist.RankedWatchlistError, match="Unsupported"):
        ranked_watchlist.rank_candidate_universe(
            prediction_days,
            tickers=["AAPL", "NVDA"],
        )


def test_ranked_watchlist_requires_explicit_horizon():
    with pytest.raises(TypeError):
        ranked_watchlist.rank_candidate_universe(tickers=["AAPL", "NVDA"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"universe": "large_mega_cap_stocks", "tickers": ["AAPL", "NVDA"]},
    ],
)
def test_ranked_watchlist_requires_exactly_one_candidate_source(kwargs):
    with pytest.raises(ranked_watchlist.RankedWatchlistError, match="exactly one"):
        ranked_watchlist.rank_candidate_universe(20, **kwargs)


def test_ranked_watchlist_enforces_two_candidates_after_spy_exclusion():
    with pytest.raises(ranked_watchlist.RankedWatchlistError, match="At least 2"):
        ranked_watchlist.rank_candidate_universe(
            20,
            tickers=["SPY", "AAPL", "aapl"],
        )


def test_ranked_watchlist_enforces_two_valid_same_date_candidates(monkeypatch):
    bundle = make_loaded_bundle()
    histories = standard_histories()
    histories["NVDA"] = make_history("NVDA", [("2026-08-26", 180.0, 0.15)])
    install_ranking_fakes(monkeypatch, bundle, histories)

    with pytest.raises(
        ranked_watchlist.RankedWatchlistDataError,
        match="Fewer than 2 valid candidates",
    ):
        ranked_watchlist.rank_candidate_universe(
            20,
            tickers=["AAPL", "NVDA"],
        )


def save_real_test_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    population = {
        "universe_name": "large_mega_cap_stocks",
        "candidate_tickers": ["AAPL", "NVDA"],
        "candidate_ticker_count": 2,
        "benchmark_ticker": "SPY",
    }
    metadata = artifacts.build_model_metadata(
        linear_features=["Close"],
        classifier_features=["momentum_10d"],
        regressor_features=[],
        prediction_days=20,
        target_mode="cross_sectional_rank_ndcg",
        training_population=population,
    )
    preparator = StoredPreparator()
    return artifacts.save_cross_sectional_rank_ndcg_artifacts(
        20,
        data_preparator=preparator,
        all_features=list(MODEL_FEATURE_COLUMNS),
        model_metadata=metadata,
        ranker=SavedRanker(),
    )


def test_bundle_loader_validates_then_loads_matching_serialized_artifacts(
    tmp_path,
    monkeypatch,
):
    paths = save_real_test_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(ranked_watchlist, "XGBRanker", LoadedRanker)

    loaded = ranked_watchlist._load_rank_ndcg_bundle(20)

    assert loaded.bundle_dir == Path(paths["bundle_dir"])
    assert loaded.manifest["bundle_id"] == Path(paths["bundle_dir"]).name
    assert loaded.preprocessor_features == MODEL_FEATURE_COLUMNS
    assert loaded.ranker_features == ["momentum_10d"]
    assert LoadedRanker.loaded_path == Path(paths["ranker"])


def test_bundle_checksum_validation_happens_before_any_serialized_load(
    tmp_path,
    monkeypatch,
):
    paths = save_real_test_bundle(tmp_path, monkeypatch)
    Path(paths["ranker"]).write_text("tampered", encoding="utf-8")
    load_calls = []
    monkeypatch.setattr(
        ranked_watchlist.joblib,
        "load",
        lambda path: load_calls.append(path),
    )

    with pytest.raises(
        ranked_watchlist.RankedWatchlistArtifactError,
        match="Checksum mismatch",
    ):
        ranked_watchlist._load_rank_ndcg_bundle(20)

    assert load_calls == []


def test_missing_current_bundle_fails_without_fallback_or_serialized_load(
    monkeypatch,
):
    monkeypatch.setattr(
        ranked_watchlist,
        "resolve_current_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            artifacts.ArtifactBundleValidationError("current.json is missing")
        ),
    )
    monkeypatch.setattr(
        ranked_watchlist.joblib,
        "load",
        lambda path: pytest.fail("joblib must not load without a valid current bundle"),
    )

    with pytest.raises(
        ranked_watchlist.RankedWatchlistArtifactError,
        match="No compatible current Rank-NDCG bundle for 20d",
    ):
        ranked_watchlist._load_rank_ndcg_bundle(20)


@pytest.mark.parametrize(
    "manifest_update, expected_message",
    [
        ({"target_mode": "excess_return"}, "not a Rank-NDCG"),
        ({"prediction_days": 10}, "horizon"),
        (
            {
                "feature_contract": {
                    "preprocessor": ["Close"],
                    "models": {"ranker": ["Close"]},
                }
            },
            "feature contract",
        ),
    ],
)
def test_bundle_mode_horizon_and_feature_contract_rejected_before_joblib(
    monkeypatch,
    manifest_update,
    expected_message,
):
    manifest = {
        "target_mode": "cross_sectional_rank_ndcg",
        "model_artifact_type": "ranker",
        "prediction_days": 20,
        "feature_contract": {
            "preprocessor": list(MODEL_FEATURE_COLUMNS),
            "models": {"ranker": ["momentum_10d"]},
        },
    }
    manifest.update(manifest_update)
    monkeypatch.setattr(
        ranked_watchlist,
        "resolve_current_bundle",
        lambda *args, **kwargs: (Path("bundle"), manifest),
    )
    monkeypatch.setattr(
        ranked_watchlist.joblib,
        "load",
        lambda path: pytest.fail("joblib must not load incompatible artifacts"),
    )

    with pytest.raises(
        ranked_watchlist.RankedWatchlistArtifactError,
        match=expected_message,
    ):
        ranked_watchlist._load_rank_ndcg_bundle(20)
