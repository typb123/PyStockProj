"""Data preparation tests for SPY-relative target creation and split metadata."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_contract import (
    ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    BASE_MODEL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
)
from src.data.data_prep import DataPreparator


def make_panel_feature_frame(tickers=("AAA", "BBB", "SPY"), num_dates=50):
    """Build a complete synthetic panel with model features and SPY benchmark rows."""
    dates = pd.bdate_range("2024-01-01", periods=num_dates)
    frames = []

    for ticker_index, ticker in enumerate(tickers):
        data = {}
        for feature in BASE_MODEL_FEATURE_COLUMNS:
            data[feature] = np.arange(num_dates, dtype=float) + ticker_index

        data["Open"] = np.linspace(1.0 + ticker_index, 100.0 + ticker_index, num_dates)
        data["Close"] = np.linspace(
            100.0 + ticker_index * 1000.0,
            100.0 + ticker_index * 1000.0 + num_dates - 1,
            num_dates,
        )
        for column in ABSOLUTE_MOMENTUM_FEATURE_COLUMNS:
            data[column] = 0.01 * (ticker_index + 1)
        for column in SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS:
            data[column] = 0.001 * (ticker_index + 1)
        frame = pd.DataFrame(data)
        frame["Ticker"] = ticker
        frame["prediction_date"] = dates
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def test_create_target_sorts_and_shifts_within_each_ticker():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "prediction_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "Close": [10.0, 100.0, 40.0, 400.0, 20.0, 200.0],
        }
    )

    preparator = DataPreparator()
    result = preparator.create_target(df, prediction_days=1)

    assert result["Ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert result["Close"].tolist() == [10.0, 20.0, 100.0, 200.0]
    np.testing.assert_allclose(result["raw_forward_return"], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(result["targetReturns"], [1.0, 1.0, 1.0, 1.0])
    assert result["forward_end_date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]


def test_data_preparator_defaults_to_ten_prediction_days():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA"] * 12,
            "prediction_date": pd.bdate_range("2024-01-01", periods=12),
            "Close": np.arange(100.0, 112.0),
        }
    )

    result = DataPreparator().create_target(df)
    first_row = result.iloc[0]

    assert first_row["Close"] == 100.0
    assert np.isclose(first_row["raw_forward_return"], 0.10)
    assert len(result) == 2


def test_create_target_uses_five_day_forward_price_when_configured():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA"] * 12,
            "prediction_date": pd.bdate_range("2024-01-01", periods=12),
            "Close": np.arange(100.0, 112.0),
        }
    )

    result = DataPreparator().create_target(df, prediction_days=5)
    first_row = result.iloc[0]

    assert first_row["Close"] == 100.0
    assert np.isclose(first_row["raw_forward_return"], 0.05)
    assert len(result) == 7


def test_create_target_uses_twenty_day_forward_price_when_configured():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA"] * 25,
            "prediction_date": pd.bdate_range("2024-01-01", periods=25),
            "Close": np.arange(100.0, 125.0),
        }
    )

    result = DataPreparator().create_target(df, prediction_days=20)
    first_row = result.iloc[0]

    assert first_row["Close"] == 100.0
    assert np.isclose(first_row["raw_forward_return"], 0.20)
    assert len(result) == 5


def test_prepare_for_train_aligns_benchmark_forward_returns_by_prediction_date():
    dates = pd.bdate_range("2024-01-01", periods=6)
    df = make_panel_feature_frame(tickers=("AAA", "SPY"), num_dates=6)
    df.loc[df["Ticker"] == "AAA", "Close"] = [
        100.0,
        110.0,
        121.0,
        133.1,
        146.41,
        161.051,
    ]
    df.loc[df["Ticker"] == "SPY", "Close"] = [
        200.0,
        202.0,
        204.02,
        206.0602,
        208.120802,
        210.20201,
    ]

    preparator = DataPreparator()
    normalized = preparator._normalize_prediction_date(df)
    raw = preparator._create_raw_forward_returns(normalized, prediction_days=1)
    benchmark_returns = preparator._build_benchmark_forward_returns(raw)
    candidates = preparator._create_benchmark_relative_targets(raw, benchmark_returns)
    first_aaa = candidates[
        (candidates["Ticker"] == "AAA") & (candidates["prediction_date"] == dates[0])
    ].iloc[0]

    assert np.isclose(first_aaa["raw_forward_return"], 0.10)
    assert np.isclose(first_aaa["benchmark_forward_return"], 0.01)
    assert np.isclose(first_aaa["excess_forward_return"], 0.09)
    assert np.isclose(first_aaa["targetReturns"], 0.09)
    assert first_aaa["beat_benchmark_target"] == 1
    assert first_aaa["forward_end_date"] == dates[1]
    assert first_aaa["benchmark_forward_end_date"] == dates[1]


def test_benchmark_relative_targets_drop_mismatched_forward_end_dates():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA", "AAA", "SPY", "SPY", "SPY", "SPY"],
            "prediction_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "Close": [100.0, 110.0, 121.0, 200.0, 202.0, 204.02, 206.0602],
        }
    )

    preparator = DataPreparator()
    raw = preparator._create_raw_forward_returns(
        preparator._normalize_prediction_date(df),
        prediction_days=1,
    )
    benchmark_returns = preparator._build_benchmark_forward_returns(raw)
    candidates = preparator._create_benchmark_relative_targets(raw, benchmark_returns)

    assert not (
        (candidates["Ticker"] == "AAA")
        & (candidates["prediction_date"] == pd.Timestamp("2024-01-01"))
    ).any()
    kept_aaa = candidates[
        (candidates["Ticker"] == "AAA")
        & (candidates["prediction_date"] == pd.Timestamp("2024-01-03"))
    ].iloc[0]
    assert kept_aaa["forward_end_date"] == pd.Timestamp("2024-01-04")
    assert kept_aaa["benchmark_forward_end_date"] == pd.Timestamp("2024-01-04")
    assert np.isclose(kept_aaa["raw_forward_return"], 0.10)
    assert np.isclose(kept_aaa["benchmark_forward_return"], 0.01)
    assert np.isclose(kept_aaa["excess_forward_return"], 0.09)


def test_cross_sectional_ranks_are_computed_within_prediction_date():
    candidates = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C", "D", "E", "F"],
            "prediction_date": pd.to_datetime(
                ["2024-01-01"] * 3 + ["2024-01-02"] * 3
            ),
            "excess_forward_return": [0.10, 0.20, 0.30, 10.0, 20.0, 30.0],
        }
    )

    result = DataPreparator()._add_cross_sectional_ranking_labels(candidates)

    assert result["excess_return_rank_pct_by_date"].tolist() == [
        0.0,
        0.5,
        1.0,
        0.0,
        0.5,
        1.0,
    ]


def test_cross_sectional_ranking_labels_top_bottom_quintiles_and_middle_rows():
    candidates = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C", "D", "E"],
            "prediction_date": pd.to_datetime(["2024-01-01"] * 5),
            "excess_forward_return": [0.00, 0.01, 0.02, 0.03, 0.04],
        }
    )

    result = DataPreparator()._add_cross_sectional_ranking_labels(candidates)

    assert result["excess_return_rank_pct_by_date"].tolist() == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert result["top_quintile_target"].isna().tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert result["top_quintile_target"].dropna().tolist() == [0.0, 1.0]
    assert result["ranking_train_sample"].tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]


def test_cross_sectional_ranking_labels_leave_single_candidate_dates_unavailable():
    candidates = pd.DataFrame(
        {
            "Ticker": ["A"],
            "prediction_date": pd.to_datetime(["2024-01-01"]),
            "excess_forward_return": [0.10],
        }
    )

    result = DataPreparator()._add_cross_sectional_ranking_labels(candidates)

    assert np.isnan(result.loc[0, "excess_return_rank_pct_by_date"])
    assert np.isnan(result.loc[0, "top_quintile_target"])
    assert not result.loc[0, "ranking_train_sample"]


def test_cross_sectional_ranking_labels_use_average_rank_for_ties():
    candidates = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C"],
            "prediction_date": pd.to_datetime(["2024-01-01"] * 3),
            "excess_forward_return": [0.00, 0.10, 0.10],
        }
    )

    result = DataPreparator()._add_cross_sectional_ranking_labels(candidates)

    assert result["excess_return_rank_pct_by_date"].tolist() == [0.0, 0.75, 0.75]
    assert result["top_quintile_target"].isna().tolist() == [False, True, True]
    assert result["ranking_train_sample"].tolist() == [True, False, False]


def test_benchmark_relative_ranking_labels_exclude_spy_candidate_rows():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB", "SPY", "AAA", "BBB", "SPY"],
            "prediction_date": pd.to_datetime(
                ["2024-01-01"] * 3 + ["2024-01-02"] * 3
            ),
            "Close": [100.0, 200.0, 50.0, 110.0, 202.0, 51.0],
        }
    )

    preparator = DataPreparator()
    raw = preparator._create_raw_forward_returns(
        preparator._normalize_prediction_date(df),
        prediction_days=1,
    )
    benchmark_returns = preparator._build_benchmark_forward_returns(raw)
    candidates = preparator._create_benchmark_relative_targets(raw, benchmark_returns)

    assert set(candidates["Ticker"]) == {"AAA", "BBB"}
    assert "SPY" not in set(candidates["Ticker"])
    assert candidates["excess_return_rank_pct_by_date"].tolist() == [1.0, 0.0]


def test_prepare_model_frame_matches_existing_preparation_sequence():
    prediction_days = 3
    df = make_panel_feature_frame(num_dates=20)
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0

    expected_preparator = DataPreparator()
    expected = expected_preparator.prepare_features(df)
    expected = expected_preparator._normalize_prediction_date(expected)
    expected_preparator._validate_required_columns(expected)
    expected = expected_preparator._create_raw_forward_returns(
        expected,
        prediction_days,
    )
    expected = expected_preparator._add_benchmark_relative_momentum(expected)
    benchmark_returns = expected_preparator._build_benchmark_forward_returns(
        expected
    )
    expected = expected_preparator._create_benchmark_relative_targets(
        expected,
        benchmark_returns,
    )
    expected_preparator.feature_columns = list(MODEL_FEATURE_COLUMNS)
    expected = expected_preparator._drop_unusable_rows(expected)

    preparator = DataPreparator()
    prepared, feature_columns = preparator.prepare_model_frame(
        df,
        prediction_days=prediction_days,
    )

    pd.testing.assert_frame_equal(prepared, expected)
    assert feature_columns == MODEL_FEATURE_COLUMNS
    assert preparator.feature_columns == MODEL_FEATURE_COLUMNS
    assert feature_columns is not preparator.feature_columns
    assert not hasattr(preparator.scalar, "mean_")


def test_prepare_for_train_uses_global_date_split_with_embargo_and_drops_spy():
    prediction_days = 3
    df = make_panel_feature_frame(num_dates=50)

    preparator = DataPreparator()
    prepared = preparator.prepare_for_train(
        df, prediction_days=prediction_days, val_size=0.2, test_size=0.2
    )

    metadata = prepared["split_metadata"]
    for split_name in ["train", "val", "test"]:
        split_metadata = metadata[split_name]
        assert set(split_metadata["Ticker"]) == {"AAA", "BBB"}
        assert "SPY" not in set(split_metadata["Ticker"])
        assert (
            not split_metadata[
                [
                    "raw_forward_return",
                    "benchmark_forward_return",
                    "excess_forward_return",
                    "beat_benchmark_target",
                    "forward_end_date",
                    "benchmark_forward_end_date",
                    "excess_return_rank_pct_by_date",
                    "top_quintile_target",
                    "ranking_train_sample",
                ]
            ]
            .isna()
            .any()
            .any()
        )

    train_dates = set(metadata["train"]["prediction_date"])
    val_dates = set(metadata["val"]["prediction_date"])
    test_dates = set(metadata["test"]["prediction_date"])
    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)

    usable_dates = sorted(df["prediction_date"].unique())[:-prediction_days]
    date_positions = {date: index for index, date in enumerate(usable_dates)}
    train_max = max(date_positions[date] for date in train_dates)
    val_min = min(date_positions[date] for date in val_dates)
    val_max = max(date_positions[date] for date in val_dates)
    test_min = min(date_positions[date] for date in test_dates)
    assert val_min - train_max - 1 >= prediction_days
    assert test_min - val_max - 1 >= prediction_days


def test_prepare_for_train_split_embargo_uses_configured_prediction_days():
    prediction_days = 7
    df = make_panel_feature_frame(num_dates=80)

    prepared = DataPreparator().prepare_for_train(
        df,
        prediction_days=prediction_days,
        val_size=0.2,
        test_size=0.2,
    )

    metadata = prepared["split_metadata"]
    usable_dates = sorted(df["prediction_date"].unique())[:-prediction_days]
    date_positions = {date: index for index, date in enumerate(usable_dates)}
    train_max = max(
        date_positions[date] for date in metadata["train"]["prediction_date"]
    )
    val_min = min(date_positions[date] for date in metadata["val"]["prediction_date"])
    val_max = max(date_positions[date] for date in metadata["val"]["prediction_date"])
    test_min = min(date_positions[date] for date in metadata["test"]["prediction_date"])

    assert val_min - train_max - 1 == prediction_days
    assert test_min - val_max - 1 == prediction_days


def test_data_preparator_fits_scaler_on_training_rows_only():
    df = make_panel_feature_frame(num_dates=40)

    preparator = DataPreparator()
    prepared = preparator.prepare_for_train(
        df, prediction_days=1, val_size=0.25, test_size=0.25
    )

    scaler = prepared["scalar"]
    first_feature_index = prepared["feature_names"].index("Open")
    train_source_indices = prepared["split_metadata"]["train"]["_source_index"]
    expected_train_mean = df.loc[train_source_indices, "Open"].mean()

    assert scaler.mean_[first_feature_index] == expected_train_mean


def test_prepare_for_train_returns_validation_split_and_benchmark_labels():
    df = make_panel_feature_frame(num_dates=40)

    preparator = DataPreparator()
    prepared = preparator.prepare_for_train(
        df, prediction_days=1, val_size=0.25, test_size=0.25
    )

    assert set(["x_val", "y_val"]).issubset(prepared)
    assert prepared["x_train"].shape[0] == len(prepared["split_metadata"]["train"])
    assert prepared["x_val"].shape[0] == len(prepared["split_metadata"]["val"])
    assert prepared["x_test"].shape[0] == len(prepared["split_metadata"]["test"])
    assert set(["direction_y_train", "direction_y_val", "direction_y_test"]).issubset(
        prepared
    )
    assert set(["target_y_train", "target_y_val", "target_y_test"]).issubset(prepared)
    assert prepared["feature_names"] == MODEL_FEATURE_COLUMNS

    for split_name, direction_key in [
        ("train", "direction_y_train"),
        ("val", "direction_y_val"),
        ("test", "direction_y_test"),
    ]:
        expected_direction = prepared["split_metadata"][split_name][
            "beat_benchmark_target"
        ].values
        np.testing.assert_array_equal(prepared[direction_key], expected_direction)


def test_prepare_for_train_preserves_available_momentum_columns_in_split_metadata():
    df = make_panel_feature_frame(num_dates=50)
    for window in [5, 10, 20, 50]:
        df[f"momentum_{window}d"] = 0.01 * window

    prepared = DataPreparator().prepare_for_train(
        df,
        prediction_days=1,
        val_size=0.2,
        test_size=0.2,
    )

    for split_metadata in prepared["split_metadata"].values():
        for column in ["momentum_5d", "momentum_10d", "momentum_20d", "momentum_50d"]:
            assert column in split_metadata.columns


def test_prepare_for_train_preserves_forward_end_dates_and_ranking_labels_in_split_metadata():
    df = make_panel_feature_frame(num_dates=50)

    prepared = DataPreparator().prepare_for_train(
        df,
        prediction_days=1,
        val_size=0.2,
        test_size=0.2,
    )

    for split_metadata in prepared["split_metadata"].values():
        assert "forward_end_date" in split_metadata.columns
        assert "benchmark_forward_end_date" in split_metadata.columns
        assert "excess_return_rank_pct_by_date" in split_metadata.columns
        assert "top_quintile_target" in split_metadata.columns
        assert "ranking_train_sample" in split_metadata.columns
        assert (
            split_metadata["forward_end_date"]
            == split_metadata["benchmark_forward_end_date"]
        ).all()
        assert set(split_metadata["excess_return_rank_pct_by_date"]) == {0.0, 1.0}
        assert set(split_metadata["top_quintile_target"]) == {0.0, 1.0}
        assert split_metadata["ranking_train_sample"].all()


def test_relative_momentum_is_date_aligned_to_spy():
    df = make_panel_feature_frame(tickers=("AAA", "SPY"), num_dates=40)
    df["momentum_10d"] = np.nan
    date = pd.Timestamp("2024-01-15")
    df.loc[
        (df["Ticker"] == "AAA") & (df["prediction_date"] == date),
        "momentum_10d",
    ] = 0.25
    df.loc[
        (df["Ticker"] == "SPY") & (df["prediction_date"] == date),
        "momentum_10d",
    ] = 0.10

    result = DataPreparator()._add_benchmark_relative_momentum(df)
    aaa_row = result[
        (result["Ticker"] == "AAA") & (result["prediction_date"] == date)
    ].iloc[0]

    assert np.isclose(aaa_row["relative_momentum_10d"], 0.15)


def test_relative_momentum_is_not_row_position_aligned():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "SPY", "AAA", "SPY"],
            "prediction_date": [dates[0], dates[1], dates[1], dates[0]],
            "momentum_5d": [0.40, 0.20, 0.50, 0.10],
        }
    )

    result = DataPreparator()._add_benchmark_relative_momentum(df)
    aaa_second_date = result[
        (result["Ticker"] == "AAA") & (result["prediction_date"] == dates[1])
    ].iloc[0]

    assert np.isclose(aaa_second_date["relative_momentum_5d"], 0.30)


def test_relative_momentum_uses_only_existing_past_momentum_columns():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "SPY"],
            "prediction_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "momentum_20d": [0.30, 0.05],
            "raw_forward_return": [9.0, -9.0],
            "benchmark_forward_return": [-8.0, 8.0],
            "excess_forward_return": [7.0, -7.0],
        }
    )

    result = DataPreparator()._add_benchmark_relative_momentum(df)

    assert np.isclose(result.loc[0, "relative_momentum_20d"], 0.25)
    assert "relative_momentum_5d" not in result.columns


def test_prepare_for_train_preserves_available_relative_momentum_columns_in_split_metadata():
    df = make_panel_feature_frame(num_dates=50)
    for window in [5, 10, 20, 50]:
        df[f"momentum_{window}d"] = 0.01 * window
        df[f"relative_momentum_{window}d"] = 0.001 * window

    prepared = DataPreparator().prepare_for_train(
        df,
        prediction_days=1,
        val_size=0.2,
        test_size=0.2,
    )

    for split_metadata in prepared["split_metadata"].values():
        for column in [
            "relative_momentum_5d",
            "relative_momentum_10d",
            "relative_momentum_20d",
            "relative_momentum_50d",
        ]:
            assert column in split_metadata.columns


def test_prepare_for_train_rejects_invalid_split_sizes():
    df = make_panel_feature_frame(num_dates=30)
    preparator = DataPreparator()

    invalid_split_args = [
        {"val_size": 0.25, "test_size": 0},
        {"val_size": 0.25, "test_size": 1},
        {"val_size": 0, "test_size": 0.25},
        {"val_size": -0.1, "test_size": 0.25},
        {"val_size": 1, "test_size": 0.25},
        {"val_size": 0.75, "test_size": 0.25},
    ]

    for kwargs in invalid_split_args:
        with pytest.raises(ValueError):
            preparator.prepare_for_train(df, prediction_days=1, **kwargs)


def test_prepare_for_train_rejects_zero_validation_size_with_clear_error():
    df = make_panel_feature_frame(num_dates=30)

    with pytest.raises(ValueError, match="val_size must be greater than 0"):
        DataPreparator().prepare_for_train(
            df,
            prediction_days=1,
            val_size=0,
            test_size=0.25,
        )
