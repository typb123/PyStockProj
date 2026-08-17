"""Focused tests for reusable model fitting and selection helpers."""

import numpy as np
import pandas as pd
import pytest

from src.config import XG_PARAMS_REGRESSOR
import src.train.model_training as model_training


def test_xgboost_regressor_candidates_include_default_baseline():
    candidates = model_training.build_xgboost_regressor_candidate_configs()

    assert candidates[0]["candidate_id"] == 0
    assert candidates[0]["candidate_name"] == "candidate_0_baseline"
    assert candidates[0]["params"] == XG_PARAMS_REGRESSOR
    assert len(candidates) >= 2


def test_validation_top_n_selection_chooses_best_candidate_and_preserves_splits(
    monkeypatch,
):
    captured_fit_calls = []

    class FakeRegressor:
        def __init__(self, **params):
            self.params = params
            self.feature_importances_ = np.array([1.0])

        def fit(self, x, y, eval_set=None, verbose=False):
            captured_fit_calls.append(
                {
                    "x": x.copy(),
                    "y": np.asarray(y).copy(),
                    "eval_x": eval_set[0][0].copy(),
                    "eval_y": np.asarray(eval_set[0][1]).copy(),
                    "verbose": verbose,
                }
            )
            return self

        def predict(self, x):
            return np.full(len(x), self.params["validation_score"])

    def fake_model_only_report(split_metadata, ranked_predictions, top_n_values):
        score = float(np.asarray(ranked_predictions)[0])
        return {
            f"top_{top_n}": {
                "model": {"average_basket_excess_return": score + top_n / 10000}
            }
            for top_n in top_n_values
        }

    monkeypatch.setattr(model_training, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        model_training,
        "build_model_only_top_n_basket_backtest_report",
        fake_model_only_report,
    )

    x_train = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    x_val = pd.DataFrame({"feature_a": [5.0, 6.0], "feature_b": [7.0, 8.0]})
    y_train = np.array([0.01, -0.02])
    y_val = np.array([0.03, 0.04])
    validation_metadata = pd.DataFrame({"Ticker": ["AAA", "BBB"]})
    candidates = [
        {
            "candidate_id": 0,
            "candidate_name": "candidate_0_baseline",
            "params": {"validation_score": 0.01},
        },
        {
            "candidate_id": 1,
            "candidate_name": "candidate_1_better",
            "params": {"validation_score": 0.03},
        },
    ]

    selected_model, report = (
        model_training.select_xgboost_regressor_by_validation_top_n(
            x_train,
            y_train,
            x_val,
            y_val,
            validation_metadata,
            candidate_configs=candidates,
        )
    )

    assert selected_model.params == {"validation_score": 0.03}
    assert report["selection_metric"] == "validation_top_n_mean_excess_return"
    assert report["selected_candidate_id"] == 1
    assert report["candidates"][0]["selected"] is False
    assert report["candidates"][1]["selected"] is True
    assert report["candidates"][1]["available_bucket_count"] == 3
    assert "unavailable or NaN buckets are ignored" in report["selection_bucket_policy"]
    assert len(captured_fit_calls) == 2
    for fit_call in captured_fit_calls:
        pd.testing.assert_frame_equal(fit_call["x"], x_train)
        pd.testing.assert_frame_equal(fit_call["eval_x"], x_val)
        np.testing.assert_array_equal(fit_call["y"], y_train)
        np.testing.assert_array_equal(fit_call["eval_y"], y_val)
        assert fit_call["verbose"] is False


def test_rank_ndcg_training_data_filters_labels_groups_and_sorts_by_date():
    features = pd.DataFrame({"feature": [0, 1, 2, 3, 4, 5]})
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-02",
                ]
            ),
            "excess_return_rank_pct_by_date": [0.90, 0.10, 0.30, 0.70, 0.80, np.nan],
        }
    )

    ranked_features, labels, qid, grouped_metadata = (
        model_training._rank_ndcg_training_data(
            features,
            metadata,
        )
    )

    np.testing.assert_array_equal(ranked_features["feature"].to_numpy(), [1, 3, 0, 2])
    np.testing.assert_array_equal(labels, [0, 3, 4, 1])
    np.testing.assert_array_equal(qid, [0, 0, 1, 1])
    assert grouped_metadata["prediction_date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
    ]


def test_rank_percentile_to_ndcg_relevance_uses_expected_boundaries():
    labels = model_training._rank_percentile_to_ndcg_relevance(
        pd.Series([0.20, 0.21, 0.40, 0.41, 0.60, 0.61, 0.79, 0.80, 1.00])
    )

    np.testing.assert_array_equal(labels.to_numpy(), [0, 1, 1, 2, 2, 3, 3, 4, 4])


def test_rank_ndcg_training_data_rejects_no_usable_rows():
    features = pd.DataFrame({"feature": [0, 1]})
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "excess_return_rank_pct_by_date": [np.nan, np.nan],
        }
    )

    with pytest.raises(ValueError, match="No usable ranker training rows"):
        model_training._rank_ndcg_training_data(features, metadata)


def test_rank_ndcg_training_data_requires_two_usable_groups():
    features = pd.DataFrame({"feature": [0, 1, 2]})
    metadata = pd.DataFrame(
        {
            "prediction_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02"]
            ),
            "excess_return_rank_pct_by_date": [0.10, 0.90, 0.50],
        }
    )

    with pytest.raises(ValueError, match="at least 2 usable prediction_date groups"):
        model_training._rank_ndcg_training_data(features, metadata)
