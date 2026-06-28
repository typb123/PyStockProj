"""Prediction-time behavior tests for saved SPY-relative model artifacts."""

import numpy as np
import pandas as pd
import pytest

import src.app as app
import src.inference.predictor as predictor
from src.config import MODEL_PATHS


class IdentityScaler:
    """Test scaler that preserves feature values while exercising scaler calls."""

    def transform(self, values):
        return values


class FakePreparator:
    """Minimal saved preparator shape used by predictor artifact loading tests."""

    feature_columns = ["Close", "Volume"]
    scalar = IdentityScaler()


class FakeLinearModel:
    """Linear baseline fake returning a stable raw prediction value."""

    def predict(self, values):
        return np.array([0.03], dtype=np.float64)


class FakeClassifier:
    """XGBoost classifier fake that records feature alignment."""

    last_columns = None

    def __init__(self, **kwargs):
        pass

    def load_model(self, path):
        pass

    def predict(self, values):
        self.__class__.last_columns = list(values.columns)
        return np.array([1], dtype=np.int64)


class FakeRegressor:
    """XGBoost regressor fake that records aligned prediction inputs."""

    prediction = 0.05
    last_columns = None
    last_values = None

    def __init__(self, **kwargs):
        pass

    def load_model(self, path):
        pass

    def predict(self, values):
        self.__class__.last_columns = list(values.columns)
        self.__class__.last_values = values.copy()
        return np.array([self.prediction], dtype=np.float32)


def install_predictor_fakes(monkeypatch, latest_row, regressor_prediction=0.05):
    """Install fake artifacts and data fetches for predictor integration tests."""
    metadata = {
        "linear_features": ["Close"],
        "classifier_features": ["Close", "Volume"],
        "regressor_features": ["Close", "Volume"],
    }

    def fake_load(path):
        if path == MODEL_PATHS["linear"]:
            return FakeLinearModel()
        if path == MODEL_PATHS["linear_scaler"]:
            return IdentityScaler()
        if path == MODEL_PATHS["model_metadata"]:
            return metadata
        if path == MODEL_PATHS["preparator"]:
            return FakePreparator()
        raise AssertionError(f"Unexpected artifact path: {path}")

    monkeypatch.setattr(predictor.joblib, "load", fake_load)
    monkeypatch.setattr(predictor, "XGBClassifier", FakeClassifier)
    FakeRegressor.prediction = regressor_prediction
    monkeypatch.setattr(predictor, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        predictor,
        "fetch_stock_data",
        lambda ticker, period="5y": pd.DataFrame([latest_row]),
    )
    monkeypatch.setattr(predictor, "calculate_data", lambda data: data)


def test_predict_spy_relative_return_returns_plain_python_output_types(monkeypatch):
    install_predictor_fakes(
        monkeypatch,
        latest_row={"Close": np.float64(100.0), "Volume": np.float64(1000.0)},
    )

    result = predictor.predict_spy_relative_return("AAPL")

    assert isinstance(result["linear_predicted_return"], float)
    assert isinstance(result["predicted_excess_return"], float)
    assert isinstance(result["signal"], str)
    assert result["linear_predicted_return"] == 0.03
    assert result["predicted_excess_return"] == float(np.float32(0.05))
    assert result["signal"] == "Expected to outperform SPY"
    assert "expected_price" not in result
    assert "direction" not in result


def test_predict_price_alias_points_to_spy_relative_predictor():
    assert predictor.predict_price is predictor.predict_spy_relative_return


def test_predict_spy_relative_return_returns_underperform_signal_for_non_positive_excess_return(
    monkeypatch,
):
    install_predictor_fakes(
        monkeypatch,
        latest_row={"Close": np.float64(100.0), "Volume": np.float64(1000.0)},
        regressor_prediction=-0.01,
    )

    result = predictor.predict_spy_relative_return("AAPL")

    assert result["predicted_excess_return"] == float(np.float32(-0.01))
    assert result["signal"] == "Expected to underperform SPY"


def test_predict_spy_relative_return_rejects_invalid_latest_features(monkeypatch):
    install_predictor_fakes(
        monkeypatch,
        latest_row={"Close": np.nan, "Volume": np.inf},
    )

    with pytest.raises(ValueError) as exc_info:
        predictor.predict_spy_relative_return("AAPL")

    message = str(exc_info.value)
    assert "Close" in message
    assert "Volume" in message


def test_predict_spy_relative_return_uses_latest_complete_feature_row(monkeypatch):
    metadata = {
        "linear_features": ["Close"],
        "classifier_features": ["Close", "Volume"],
        "regressor_features": ["Close", "Volume"],
    }

    class TwoFeaturePreparator:
        feature_columns = ["Close", "Volume"]
        scalar = IdentityScaler()

    def fake_load(path):
        if path == MODEL_PATHS["linear"]:
            return FakeLinearModel()
        if path == MODEL_PATHS["linear_scaler"]:
            return IdentityScaler()
        if path == MODEL_PATHS["model_metadata"]:
            return metadata
        if path == MODEL_PATHS["preparator"]:
            return TwoFeaturePreparator()
        raise AssertionError(f"Unexpected artifact path: {path}")

    processed_data = pd.DataFrame(
        {
            "Close": [100.0, np.nan],
            "Volume": [1000.0, 1100.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    monkeypatch.setattr(predictor.joblib, "load", fake_load)
    monkeypatch.setattr(predictor, "XGBClassifier", FakeClassifier)
    monkeypatch.setattr(predictor, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(predictor, "fetch_stock_data", lambda ticker, period="5y": processed_data)
    monkeypatch.setattr(predictor, "calculate_data", lambda data: data.copy())

    result = predictor.predict_spy_relative_return("AAPL")

    assert "error" not in result
    assert FakeRegressor.last_values["Close"].iloc[0] == 100.0
    assert FakeRegressor.last_values["Volume"].iloc[0] == 1000.0


def test_latest_complete_feature_row_reports_missing_features_when_none_complete():
    processed_data = pd.DataFrame(
        {
            "Close": [np.nan, 100.0],
            "Volume": [1000.0, np.inf],
        }
    )

    with pytest.raises(ValueError) as exc_info:
        predictor._select_latest_complete_feature_row(
            processed_data,
            ["Close", "Volume"],
            "AAPL",
        )

    message = str(exc_info.value)
    assert "No complete prediction feature row for AAPL" in message
    assert "Volume" in message


def test_predict_spy_relative_return_computes_relative_momentum_for_saved_feature_contract(
    monkeypatch,
):
    class MomentumPreparator:
        feature_columns = [
            "Close",
            "Volume",
            "momentum_10d",
            "relative_momentum_10d",
        ]
        scalar = IdentityScaler()

    metadata = {
        "linear_features": ["Close"],
        "classifier_features": [
            "Close",
            "Volume",
            "momentum_10d",
            "relative_momentum_10d",
        ],
        "regressor_features": [
            "Close",
            "Volume",
            "momentum_10d",
            "relative_momentum_10d",
        ],
    }

    def fake_load(path):
        if path == MODEL_PATHS["linear"]:
            return FakeLinearModel()
        if path == MODEL_PATHS["linear_scaler"]:
            return IdentityScaler()
        if path == MODEL_PATHS["model_metadata"]:
            return metadata
        if path == MODEL_PATHS["preparator"]:
            return MomentumPreparator()
        raise AssertionError(f"Unexpected artifact path: {path}")

    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    stock_data = pd.DataFrame(
        {
            "Close": [100.0, 110.0],
            "Volume": [1000.0, 1100.0],
            "momentum_10d": [0.10, 0.20],
        },
        index=dates,
    )
    spy_data = pd.DataFrame(
        {
            "Close": [400.0, 404.0],
            "Volume": [2000.0, 2100.0],
            "momentum_10d": [0.02, 0.05],
        },
        index=dates,
    )

    def fake_fetch_stock_data(ticker, period="5y"):
        return spy_data if ticker == "SPY" else stock_data

    monkeypatch.setattr(predictor.joblib, "load", fake_load)
    monkeypatch.setattr(predictor, "XGBClassifier", FakeClassifier)
    monkeypatch.setattr(predictor, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(predictor, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(predictor, "calculate_data", lambda data: data.copy())

    result = predictor.predict_spy_relative_return("AAPL")

    assert "error" not in result
    assert FakeClassifier.last_columns == metadata["classifier_features"]
    assert FakeRegressor.last_columns == metadata["regressor_features"]
    assert np.isclose(FakeRegressor.last_values["relative_momentum_10d"].iloc[0], 0.15)


def test_predict_spy_relative_return_falls_back_to_latest_complete_relative_momentum_row(
    monkeypatch,
):
    class MomentumPreparator:
        feature_columns = [
            "Close",
            "Volume",
            "momentum_10d",
            "relative_momentum_10d",
        ]
        scalar = IdentityScaler()

    metadata = {
        "linear_features": ["Close"],
        "classifier_features": [
            "Close",
            "Volume",
            "momentum_10d",
            "relative_momentum_10d",
        ],
        "regressor_features": [
            "Close",
            "Volume",
            "momentum_10d",
            "relative_momentum_10d",
        ],
    }

    def fake_load(path):
        if path == MODEL_PATHS["linear"]:
            return FakeLinearModel()
        if path == MODEL_PATHS["linear_scaler"]:
            return IdentityScaler()
        if path == MODEL_PATHS["model_metadata"]:
            return metadata
        if path == MODEL_PATHS["preparator"]:
            return MomentumPreparator()
        raise AssertionError(f"Unexpected artifact path: {path}")

    stock_dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    spy_dates = pd.to_datetime(["2024-01-01"])
    stock_data = pd.DataFrame(
        {
            "Close": [100.0, 110.0],
            "Volume": [1000.0, 1100.0],
            "momentum_10d": [0.10, 0.20],
        },
        index=stock_dates,
    )
    spy_data = pd.DataFrame(
        {
            "Close": [400.0],
            "Volume": [2000.0],
            "momentum_10d": [0.02],
        },
        index=spy_dates,
    )

    def fake_fetch_stock_data(ticker, period="5y"):
        return spy_data if ticker == "SPY" else stock_data

    monkeypatch.setattr(predictor.joblib, "load", fake_load)
    monkeypatch.setattr(predictor, "XGBClassifier", FakeClassifier)
    monkeypatch.setattr(predictor, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(predictor, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(predictor, "calculate_data", lambda data: data.copy())

    result = predictor.predict_spy_relative_return("AAPL")

    assert "error" not in result
    assert FakeRegressor.last_values["Close"].iloc[0] == 100.0
    assert np.isclose(FakeRegressor.last_values["relative_momentum_10d"].iloc[0], 0.08)


def test_get_stock_info_prints_spy_relative_prediction_without_expected_price(
    monkeypatch,
    capsys,
):
    one_month_data = pd.DataFrame(
        {
            "High": [110.0, 120.0],
            "Low": [90.0, 95.0],
        },
        index=pd.to_datetime(["2026-06-01", "2026-06-02"]),
    )
    one_year_data = pd.DataFrame(
        {
            "High": [130.0, 140.0],
            "Low": [80.0, 85.0],
        },
        index=pd.to_datetime(["2026-06-01", "2026-06-02"]),
    )

    def fake_fetch_stock_data(ticker, period="5y"):
        return one_month_data if period == "1mo" else one_year_data

    class FakeTicker:
        def __init__(self, ticker):
            self.info = {"currentPrice": 100.0}

    monkeypatch.setattr(app, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(app.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(
        app, "get_prediction_date", lambda *args, **kwargs: "07/10/2026"
    )
    monkeypatch.setattr(
        app,
        "predict_spy_relative_return",
        lambda ticker: {
            "predicted_excess_return": -0.001864,
            "signal": "Expected to underperform SPY",
            "linear_predicted_return": 0.01,
        },
    )

    app.get_stock_info("AAPL")

    output = capsys.readouterr().out
    assert "Current Price: $100.00" in output
    assert "1-Month High: $120.00" in output
    assert "52-Week Low: $80.00" in output
    assert "Prediction for AAPL on 07/10/2026 (10 trading days ahead):" in output
    assert "Predicted Excess Return vs SPY: -0.1864%" in output
    assert "Signal: Expected to underperform SPY" in output
    assert "Expected Price" not in output
    assert "Predicted Return" not in output
    assert "Direction:" not in output
