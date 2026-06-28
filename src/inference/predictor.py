"""Prediction-time inference using the artifacts saved by trainer.py.

Feature order, preprocessing, and model feature subsets are loaded from training
artifacts so live predictions match the trained inputs.
"""

import joblib
import logging
import copy
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from src.config import (
    MODEL_PATHS,
    LOG_FILE,
    XG_PARAMS_CLASSIFIER,
    XG_PARAMS_REGRESSOR,
)
from src.features.feature_contract import (
    ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
)
from src.data.technical_indicators import calculate_data
from src.data.data_fetch import fetch_stock_data


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s: - %(levelname)s -%(message)s",
)


def _add_prediction_relative_momentum(
    processed_data: pd.DataFrame,
    benchmark_processed_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add same-date SPY-relative trailing momentum for prediction features."""
    df = processed_data.copy()
    benchmark_df = benchmark_processed_data.copy()

    if "prediction_date" not in df.columns:
        df["prediction_date"] = pd.to_datetime(df.index).normalize()
    else:
        df["prediction_date"] = pd.to_datetime(df["prediction_date"]).dt.normalize()

    if "prediction_date" not in benchmark_df.columns:
        benchmark_df["prediction_date"] = pd.to_datetime(benchmark_df.index).normalize()
    else:
        benchmark_df["prediction_date"] = pd.to_datetime(
            benchmark_df["prediction_date"]
        ).dt.normalize()

    available_momentum_columns = [
        column for column in ABSOLUTE_MOMENTUM_FEATURE_COLUMNS if column in df.columns
    ]
    benchmark_momentum_columns = [
        column
        for column in available_momentum_columns
        if column in benchmark_df.columns
    ]
    if not benchmark_momentum_columns:
        return df

    benchmark_momentum = benchmark_df[
        ["prediction_date"] + benchmark_momentum_columns
    ].drop_duplicates(
        subset=["prediction_date"],
        keep="first",
    )
    benchmark_momentum = benchmark_momentum.rename(
        columns={column: f"benchmark_{column}" for column in benchmark_momentum_columns}
    )

    df = df.merge(benchmark_momentum, on="prediction_date", how="left")
    for column in benchmark_momentum_columns:
        relative_column = f"relative_{column}"
        df[relative_column] = df[column] - df[f"benchmark_{column}"]
        df = df.drop(columns=[f"benchmark_{column}"])

    return df


def _select_latest_complete_feature_row(
    processed_data: pd.DataFrame,
    feature_columns: list[str],
    ticker: str,
) -> pd.Series:
    """Return the newest row usable by every saved model feature column."""
    if processed_data.empty:
        raise ValueError(f"No data available for ticker {ticker}")

    missing_cols = [
        column for column in feature_columns if column not in processed_data
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    feature_frame = processed_data.loc[:, feature_columns]
    numeric_feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    finite_mask = pd.DataFrame(
        np.isfinite(numeric_feature_frame.to_numpy(dtype=float)),
        index=numeric_feature_frame.index,
        columns=numeric_feature_frame.columns,
    )
    valid_rows = numeric_feature_frame.notna() & finite_mask
    complete_row_mask = valid_rows.all(axis=1)

    if complete_row_mask.any():
        return numeric_feature_frame.loc[complete_row_mask].iloc[-1]

    latest_original_row = feature_frame.iloc[-1]
    latest_numeric_row = numeric_feature_frame.iloc[-1]
    invalid_features = latest_numeric_row.index[
        latest_original_row.isna()
        | latest_numeric_row.isna()
        | ~np.isfinite(latest_numeric_row)
    ].tolist()
    raise ValueError(
        f"No complete prediction feature row for {ticker}. "
        f"Latest row has invalid required features: {invalid_features}"
    )


def predict_spy_relative_return(ticker: str) -> dict:
    """Predict SPY-relative excess return for one ticker using saved artifacts.

    The linear model is reported as a baseline, while the XGBoost regressor
    supplies the excess-return signal used by the console app.
    """
    try:
        logging.info(f"Loading model for {ticker}...")

        # Load the baseline model, its scaler, and feature metadata from training.
        linear_model = joblib.load(MODEL_PATHS["linear"])
        scaler_lr = joblib.load(MODEL_PATHS["linear_scaler"])
        model_metadata = joblib.load(MODEL_PATHS["model_metadata"])

        classifier_params = copy.deepcopy(XG_PARAMS_CLASSIFIER)
        regressor_params = copy.deepcopy(XG_PARAMS_REGRESSOR)

        xgb_classifier = XGBClassifier(**classifier_params)
        xgb_regressor = XGBRegressor(**regressor_params)

        xgb_classifier.load_model(MODEL_PATHS["classifier"])
        xgb_regressor.load_model(MODEL_PATHS["regressor"])

        # data_preparator.scalar is the saved training-time scaler artifact name.
        data_preparator = joblib.load(MODEL_PATHS["preparator"])
        feature_columns = data_preparator.feature_columns

        # calculate_data is called on one ticker's history, matching its assumptions.
        data = fetch_stock_data(ticker, period="5y")
        processed_data = calculate_data(data)
        if any(
            column in feature_columns
            or column in model_metadata["classifier_features"]
            or column in model_metadata["regressor_features"]
            for column in SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS
        ):
            benchmark_data = (
                data if ticker == "SPY" else fetch_stock_data("SPY", period="5y")
            )
            benchmark_processed_data = calculate_data(benchmark_data)
            processed_data = _add_prediction_relative_momentum(
                processed_data,
                benchmark_processed_data,
            )

        numeric_latest_feature_row = _select_latest_complete_feature_row(
            processed_data,
            feature_columns,
            ticker,
        )

        latest_features = numeric_latest_feature_row.to_numpy(dtype=float).reshape(
            1, -1
        )
        latest_features = data_preparator.scalar.transform(latest_features)
        latest_features_df = pd.DataFrame(latest_features, columns=feature_columns)

        # Feature subsets and order must match the saved training metadata.
        linear_features = model_metadata["linear_features"]
        lr_features = scaler_lr.transform(latest_features_df[linear_features])
        lr_prediction = linear_model.predict(lr_features)[0]

        classifier_features = model_metadata["classifier_features"]
        regressor_features = model_metadata["regressor_features"]
        classifier_input = latest_features_df[classifier_features]
        regressor_input = latest_features_df[regressor_features]

        xgb_classifier.predict(classifier_input)[0]
        predicted_excess_return = xgb_regressor.predict(regressor_input)[0]

        if predicted_excess_return > 0:
            signal = "Expected to outperform SPY"
        else:
            signal = "Expected to underperform SPY"

        # Return model outputs without converting SPY-relative return into a price.
        result = {
            "linear_predicted_return": float(lr_prediction),
            "predicted_excess_return": float(predicted_excess_return),
            "signal": signal,
        }

        logging.info(f"Prediction for {ticker}: {result}")
        return result
    except ValueError:
        logging.error(
            f"Error predicting SPY-relative return for {ticker}: invalid input data"
        )
        raise
    except Exception as e:
        logging.error(f"Error predicting SPY-relative return for {ticker}: {e}")
        return {"error": str(e)}


predict_price = predict_spy_relative_return
