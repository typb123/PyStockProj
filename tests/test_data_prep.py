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
        [1.0] * 15 + [1000.0] * 5,
        dtype=float,
    )

    df = pd.DataFrame(data)
    df["Close"] = np.linspace(100.0, 119.0, num_rows)
    return df


def test_data_preparator_fits_scaler_on_training_rows_only():
    df = make_feature_frame(num_rows=20)

    preparator = DataPreparator()
    prepared = preparator.prepare_for_train(df, prediction_days=1, test_size=0.25)

    scaler = prepared["scalar"]
    first_feature_index = prepared["feature_names"].index(REQUIRED_COLUMNS[0])

    # prediction_days=1 drops the last row, leaving 19 rows.
    # test_size=0.25 with sklearn train_test_split leaves 14 train rows and 5 test rows.
    expected_train_mean = df.iloc[:14][REQUIRED_COLUMNS[0]].mean()

    assert scaler.mean_[first_feature_index] == expected_train_mean
