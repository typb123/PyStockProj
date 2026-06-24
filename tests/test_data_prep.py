import numpy as np
import pandas as pd

from src.config import REQUIRED_COLUMNS
from src.data.data_prep import DataPreparator


def make_feature_frame(num_rows=20):
    data = {}
    for feature in REQUIRED_COLUMNS:
        data[feature] = np.arange(num_rows, dtype=float)

    # Make the first feature obviously different in the final test rows.
    # If the scaler is fit on all rows, the mean will include these large values.
    data[REQUIRED_COLUMNS[0]] = np.array(
        [1.0] * (num_rows - 5) + [1000.0] * 5,
        dtype=float,
    )

    df = pd.DataFrame(data)
    df["Close"] = np.linspace(100.0, 119.0, num_rows)
    return df


def test_data_preparator_fits_scaler_on_training_rows_only():
    df = make_feature_frame(num_rows=21)

    preparator = DataPreparator()
    prepared = preparator.prepare_for_train(
        df, prediction_days=1, val_size=0.25, test_size=0.25
    )

    scaler = prepared["scalar"]
    first_feature_index = prepared["feature_names"].index(REQUIRED_COLUMNS[0])

    # prediction_days=1 drops the last row, leaving 20 rows.
    # test_size=0.25 leaves 15 train/validation rows and 5 test rows.
    # val_size=0.25 then leaves 10 train rows and 5 validation rows.
    expected_train_mean = df.iloc[:10][REQUIRED_COLUMNS[0]].mean()

    assert scaler.mean_[first_feature_index] == expected_train_mean


def test_prepare_for_train_returns_validation_split():
    df = make_feature_frame(num_rows=21)

    preparator = DataPreparator()
    prepared = preparator.prepare_for_train(
        df, prediction_days=1, val_size=0.25, test_size=0.25
    )

    assert set(["x_val", "y_val"]).issubset(prepared)
    assert prepared["x_train"].shape[0] == 10
    assert prepared["x_val"].shape[0] == 5
    assert prepared["x_test"].shape[0] == 5
    assert prepared["y_train"].shape[0] == 10
    assert prepared["y_val"].shape[0] == 5
    assert prepared["y_test"].shape[0] == 5


def test_create_target_shifts_within_each_ticker():
    df = pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "Close": [10.0, 20.0, 30.0, 1000.0, 2000.0, 3000.0],
        }
    )

    preparator = DataPreparator()
    result = preparator.create_target(df, prediction_days=1)

    assert result["Ticker"].tolist() == ["AAA", "AAA", "BBB", "BBB"]
    assert result["Close"].tolist() == [10.0, 20.0, 1000.0, 2000.0]
    np.testing.assert_allclose(result["targetReturns"], [1.0, 0.5, 1.0, 0.5])
    assert "target" not in result.columns


def test_prepare_for_train_rejects_invalid_split_sizes():
    df = make_feature_frame(num_rows=21)
    preparator = DataPreparator()

    invalid_split_args = [
        {"val_size": 0.25, "test_size": 0},
        {"val_size": 0.25, "test_size": 1},
        {"val_size": -0.1, "test_size": 0.25},
        {"val_size": 1, "test_size": 0.25},
        {"val_size": 0.75, "test_size": 0.25},
    ]

    for kwargs in invalid_split_args:
        try:
            preparator.prepare_for_train(df, prediction_days=1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
