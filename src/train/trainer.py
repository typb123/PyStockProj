"""Training entry point for the stock return prediction models.

The pipeline keeps Linear Regression as a separate baseline artifact, while
XGBoost trains a direction classifier and a return-magnitude regressor.
"""

import joblib
import pandas as pd
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.data.data_prep import DataPreparator
from src.data.technical_indicators import calculate_data
from src.data.data_fetch import fetch_stock_data
from src.train.evaluator import (
    build_actual_return_baseline_report,
    build_classification_report,
    build_combined_signal_report,
    build_probability_summary,
    build_probability_tail_report,
    build_probability_threshold_report,
    build_predicted_return_quantile_report,
    build_regression_report,
    build_return_correlation_report,
    build_trading_relevance_report,
    build_validation_selected_threshold_report,
)
from xgboost import XGBRegressor, XGBClassifier
from src.config import (
    XG_PARAMS_CLASSIFIER,
    XG_PARAMS_REGRESSOR,
    MODEL_PATHS,
    TRAINING_TICKERS,
    PREDICTION_DAYS,
    TEST_SIZE,
)


logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s: - %(levelname)s -%(message)s",
)


def validate_input_data(data):
    """
    Validate the input data structure before feature/target preparation.

    Missing values are logged but not filled here; DataPreparator handles required
    feature/target row dropping without future-looking imputation.
    """
    if data.empty:
        raise ValueError("Input data is empty.")

    nan_count = data.isnull().sum().sum()
    logging.info(f"NaN values before preparation: {nan_count}")

    if nan_count > 0:
        nan_by_column = data.isnull().sum()
        nan_columns = [col for col in data.columns if nan_by_column[col] > 0]
        for col in nan_columns:
            logging.info(
                f"Column {col}: {nan_by_column[col]} NaN values ({nan_by_column[col] / len(data) * 100:.2f}%)"
            )

    logging.info("Data validation complete.")
    return data


def fetch_tickers_data(ticker, period="5y"):
    """
    Fetch one ticker and generate indicators before ticker-level concatenation.

    calculate_data is order-dependent and not grouped internally, so the current
    fetch path calls it while each DataFrame still contains one ticker only.
    """
    try:
        stock_data = fetch_stock_data(ticker, period=period)
        if stock_data.empty:
            logging.warning(f"No data returned for {ticker}. Skipping...")
            return None

        # Log data size before processing
        logging.info(f"Data size before processing for {ticker}: {stock_data.shape}")

        stock_data = calculate_data(stock_data)
        stock_data["Ticker"] = ticker

        # Log data size after processing
        logging.info(f"Data size after processing for {ticker}: {stock_data.shape}")

        time.sleep(0.2)  # Small delay to avoid rate limits
        return stock_data
    except Exception as e:
        logging.warning(f"Failed to fetch data for {ticker}: {e}")
        return None


def prepare_data_parallel(tickers, period="5y") -> pd.DataFrame:
    """Fetch and prepare each ticker independently, then concatenate valid results."""
    logging.info(f"Fetching data for {len(tickers)} tickers...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(
            lambda ticker: fetch_tickers_data(ticker, period=period), tickers
        )

    all_data = [data for data in results if data is not None and not data.empty]

    logging.info(f"Total tickers fetched with valid data: {len(all_data)}")

    if not all_data:
        logging.error("No data fetched for training. Exiting...")
        raise ValueError("No data was fetched for any ticker.")

    return pd.concat(all_data, ignore_index=True)


def cross_validate_model(x, y, model=None, cv=5):
    """
    Perform cross-validation on the given model and data.

    Parameters:
        X (np.ndarray): Feature matrix.
        Y (np.ndarray): Target variable.
        model: Scikit-learn model (default: LinearRegression).
        cv (int): Number of cross-validation folds.

    Returns:
        float: Average cross-validated MSE.
    """
    if model is None:
        model = LinearRegression()

    # Define custom scoring for negative MSE
    scores = cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=cv)
    if scores is None:
        raise ValueError(
            "Cross-validation scoring failed. Please check your model and data."
        )
    avg_mse = -np.mean(scores)
    logging.info(f"Cross-Validated MSE (cv={cv}): {avg_mse:.4f}")
    return avg_mse


def evaluate_model(model, x_test, y_test, model_type="regression"):
    """Log a compact regression or classification report for a held-out split."""
    y_pred = model.predict(x_test)

    if model_type == "regression":
        report = build_regression_report(y_test, y_pred)
        logging.info(
            f"MSE: {report['mse']:.4f}, MAE: {report['mae']:.4f}, R²: {report['r2']:.4f}"
        )

    elif model_type == "classification":
        report = build_classification_report(y_test, y_pred)
        logging.info(f"Accuracy: {report['accuracy']:.4%}")
        logging.info(f"Classification Report: {report}")


def log_xgboost_test_report(
    y_train,
    y_val,
    y_test,
    classifier_predictions,
    classifier_validation_probability_up,
    classifier_probability_up,
    regressor_predictions,
):
    """
    Log final XGBoost diagnostics on test data without tuning on that test data.

    Validation probabilities may select a classifier threshold; test probabilities
    only evaluate the already-selected rule.
    """
    classification_report_data = build_classification_report(
        (y_test > 0).astype(int), classifier_predictions
    )
    regression_report_data = build_regression_report(
        y_test, regressor_predictions, y_train=y_train
    )
    actual_return_baseline_report = build_actual_return_baseline_report(y_test)
    trading_report_data = build_trading_relevance_report(
        y_test, regressor_predictions, classifier_predictions
    )
    probability_summary = build_probability_summary(classifier_probability_up)
    probability_tail_report = build_probability_tail_report(
        classifier_probability_up, y_test
    )
    probability_threshold_report = build_probability_threshold_report(
        classifier_probability_up, y_test
    )
    validation_selected_threshold_report = build_validation_selected_threshold_report(
        classifier_validation_probability_up,
        y_val,
        classifier_probability_up,
        y_test,
    )
    predicted_return_quantile_report = build_predicted_return_quantile_report(
        y_test, regressor_predictions
    )
    return_correlation_report = build_return_correlation_report(
        y_test, regressor_predictions
    )
    combined_signal_report = build_combined_signal_report(
        classifier_probability_up,
        y_test,
        regressor_predictions,
    )

    logging.info(f"XGBoost Test Classification Report: {classification_report_data}")
    logging.info(f"XGBoost Test Regression Report: {regression_report_data}")
    logging.info(
        f"XGBoost Actual Return Baseline Report: {actual_return_baseline_report}"
    )
    logging.info(f"XGBoost Trading Relevance Report: {trading_report_data}")
    logging.info(f"XGBoost Classifier Probability Summary: {probability_summary}")
    logging.info(
        f"XGBoost Classifier Probability Tail Report: {probability_tail_report}"
    )
    logging.info(
        f"XGBoost Classifier Probability Threshold Report: {probability_threshold_report}"
    )
    logging.info(
        f"XGBoost Classifier Validation-Selected Threshold Report: {validation_selected_threshold_report}"
    )
    logging.info(
        f"XGBoost Regressor Predicted Return Quantile Report: {predicted_return_quantile_report}"
    )
    logging.info(
        f"XGBoost Regressor Return Correlation Report: {return_correlation_report}"
    )
    logging.info(f"XGBoost Combined Signal Report: {combined_signal_report}")


def build_model_metadata(
    linear_features,
    classifier_features,
    regressor_features,
    prediction_days,
):
    """Build prediction-time metadata needed to align saved artifacts and features."""
    return {
        "linear_features": linear_features,
        "classifier_features": classifier_features,
        "regressor_features": regressor_features,
        "classifier_feature_names": classifier_features,
        "regressor_feature_names": regressor_features,
        "prediction_days": prediction_days,
    }


def train_models(
    data: pd.DataFrame,
    prediction_days: int = PREDICTION_DAYS,
    classifier_params: dict | None = None,
    regressor_params: dict | None = None,
) -> None:
    """
    Train the linear baseline, XGBoost classifier, and XGBoost regressor.

    Validation data is used for XGBoost eval_set and threshold research; test data
    is reserved for final diagnostics.
    """
    logging.info("Training models with explicitly defined feature arrays.")

    # DataPreparator owns target creation, chronological splits, and shared scaling.
    data = validate_input_data(data)
    data_preparator = DataPreparator()
    prepared_data = data_preparator.prepare_for_train(
        data, prediction_days=prediction_days, test_size=TEST_SIZE
    )

    # Rebuild DataFrames so explicit feature lists preserve their trained column order.
    all_features = prepared_data["feature_names"]
    x_train_full = pd.DataFrame(prepared_data["x_train"], columns=all_features)
    x_val_full = pd.DataFrame(prepared_data["x_val"], columns=all_features)
    x_test_full = pd.DataFrame(prepared_data["x_test"], columns=all_features)
    y_train = prepared_data["y_train"]
    y_val = prepared_data["y_val"]
    y_test = prepared_data["y_test"]

    # Feature lists are intentionally inline for now; prediction metadata mirrors them.
    linear_features = [
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_lag_close_26",
        "chikou_return_26",
        "chikou_above_lag_26",
        "Open",
        "Close",
        "rsi",
        "signalLine",
        "ATR",
        "20_day_avg",
        "macd",
        "BB_Std",
        "obv",
        "dailyReturn",
        "macdHistogram",
        "vma_20",
        "High",
        "Low",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "stoch_k",
        "stoch_d",
        "Volume",
        "10_day_avg",
        "volatility",
        "vma_10",
        "5_day_avg",
    ]
    classifier_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "5_day_avg",
        "10_day_avg",
        "20_day_avg",
        "dailyReturn",
        "volatility",
        "rsi",
        "macd",
        "signalLine",
        "macdHistogram",
        "obv",
        "vma_10",
        "vma_20",
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_lag_close_26",
        "chikou_return_26",
        "chikou_above_lag_26",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "BB_Std",
        "ATR",
        "stoch_k",
        "stoch_d",
    ]
    regressor_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "5_day_avg",
        "10_day_avg",
        "20_day_avg",
        "dailyReturn",
        "volatility",
        "rsi",
        "macd",
        "signalLine",
        "macdHistogram",
        "obv",
        "vma_10",
        "vma_20",
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_lag_close_26",
        "chikou_return_26",
        "chikou_above_lag_26",
        "BB_Middle",
        "BB_Upper",
        "BB_Lower",
        "BB_Std",
        "ATR",
        "stoch_k",
        "stoch_d",
    ]

    # Linear Regression is a separate scaled baseline, not a stacked XGBoost feature.
    scaler_lr = StandardScaler()
    x_train_lr_scaled = scaler_lr.fit_transform(x_train_full[linear_features])
    x_test_lr_scaled = scaler_lr.transform(x_test_full[linear_features])

    logging.info("Training Linear Regression model...")
    linear_model = LinearRegression()
    linear_model.fit(x_train_lr_scaled, y_train)
    evaluate_model(linear_model, x_test_lr_scaled, y_test, model_type="regression")

    # Log feature importances for Linear Regression
    importances_lr = np.abs(linear_model.coef_)
    feature_importance_lr = pd.DataFrame(
        {"feature": linear_features, "importance": importances_lr}
    ).sort_values(by="importance", ascending=False)
    logging.info("Feature Importances for Linear Regression:")
    for _, row in feature_importance_lr.iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")

    classifier_feature_names = classifier_features
    regressor_feature_names = regressor_features

    x_train_classifier = x_train_full[classifier_features]
    x_val_classifier = x_val_full[classifier_features]
    x_test_classifier = x_test_full[classifier_features]
    x_train_regressor = x_train_full[regressor_features]
    x_val_regressor = x_val_full[regressor_features]
    x_test_regressor = x_test_full[regressor_features]

    # The classifier predicts direction only; the regressor predicts return magnitude.
    direction_y_train = (y_train > 0).astype(int)
    direction_y_val = (y_val > 0).astype(int)
    direction_y_test = (y_test > 0).astype(int)

    logging.info("Training XGBoost Classifier...")
    classifier = XGBClassifier(**(classifier_params or XG_PARAMS_CLASSIFIER))
    classifier.fit(
        x_train_classifier,
        direction_y_train,
        eval_set=[(x_val_classifier, direction_y_val)],
    )
    evaluate_model(
        classifier, x_test_classifier, direction_y_test, model_type="classification"
    )

    # Log feature importances for XGBoost Classifier
    importances_clf = classifier.feature_importances_
    feature_importance_clf = pd.DataFrame(
        {"feature": classifier_feature_names, "importance": importances_clf}
    ).sort_values(by="importance", ascending=False)
    logging.info("Feature Importances for XGBoost Classifier:")
    for _, row in feature_importance_clf.iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")

    logging.info("Training XGBoost Regressor...")
    regressor = XGBRegressor(**(regressor_params or XG_PARAMS_REGRESSOR))
    regressor.fit(
        x_train_regressor,
        y_train,
        eval_set=[(x_val_regressor, y_val)],
    )
    evaluate_model(regressor, x_test_regressor, y_test, model_type="regression")
    # Validation probabilities choose the threshold; test probabilities evaluate that chosen rule.
    log_xgboost_test_report(
        y_train,
        y_val,
        y_test,
        classifier.predict(x_test_classifier),
        classifier.predict_proba(x_val_classifier)[:, 1],
        classifier.predict_proba(x_test_classifier)[:, 1],
        regressor.predict(x_test_regressor),
    )

    # Log feature importances for XGBoost Regressor
    importances_reg = regressor.feature_importances_
    feature_importance_reg = pd.DataFrame(
        {"feature": regressor_feature_names, "importance": importances_reg}
    ).sort_values(by="importance", ascending=False)
    logging.info("Feature Importances for XGBoost Regressor:")
    for _, row in feature_importance_reg.iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")

    # Save all preprocessing and feature metadata needed to reproduce training inputs.
    model_metadata = build_model_metadata(
        linear_features,
        classifier_features,
        regressor_features,
        prediction_days,
    )
    joblib.dump(linear_model, MODEL_PATHS["linear"])
    joblib.dump(scaler_lr, MODEL_PATHS["linear_scaler"])
    joblib.dump(data_preparator, MODEL_PATHS["preparator"])
    joblib.dump(all_features, MODEL_PATHS["features"])
    joblib.dump(model_metadata, MODEL_PATHS["model_metadata"])
    classifier.save_model(MODEL_PATHS["classifier"])
    regressor.save_model(MODEL_PATHS["regressor"])
    logging.info("Training completed. Models saved successfully.")


def main():
    data = prepare_data_parallel(TRAINING_TICKERS, period="5y")
    if data.empty:
        logging.error("No data fetched for training. Exiting...")
        return

    logging.info("Starting model training...")
    train_models(data)
    logging.info("Model training completed.")
    print("Model training completed. Check training.log for details.")


if __name__ == "__main__":
    main()
