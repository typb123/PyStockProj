import numpy as np
import pandas as pd

import src.inference.predictor as predictor
from src.config import MODEL_PATHS


class IdentityScaler:
    def transform(self, values):
        return values


class FakePreparator:
    feature_columns = ["Close", "Volume"]
    scalar = IdentityScaler()


class FakeLinearModel:
    def predict(self, values):
        return np.array([0.03], dtype=np.float64)


class FakeClassifier:
    def __init__(self, **kwargs):
        pass

    def load_model(self, path):
        pass

    def predict(self, values):
        return np.array([1], dtype=np.int64)


class FakeRegressor:
    def __init__(self, **kwargs):
        pass

    def load_model(self, path):
        pass

    def predict(self, values):
        return np.array([0.05], dtype=np.float32)


def install_predictor_fakes(monkeypatch, latest_row):
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
    monkeypatch.setattr(predictor, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(
        predictor,
        "fetch_stock_data",
        lambda ticker, period="5y": pd.DataFrame([latest_row]),
    )
    monkeypatch.setattr(predictor, "calculate_data", lambda data: data)


def test_predict_price_returns_plain_python_output_types(monkeypatch):
    install_predictor_fakes(
        monkeypatch,
        latest_row={"Close": np.float64(100.0), "Volume": np.float64(1000.0)},
    )

    result = predictor.predict_price("AAPL")

    assert isinstance(result["direction"], str)
    assert isinstance(result["linear_predicted_return"], float)
    assert isinstance(result["predicted_return"], float)
    assert isinstance(result["expected_price"], float)
    assert result["direction"] == "Up"
    assert result["linear_predicted_return"] == 0.03
    assert result["predicted_return"] == float(np.float32(0.05))
    assert np.isclose(result["expected_price"], 105.0)


def test_predict_price_rejects_invalid_latest_features(monkeypatch):
    install_predictor_fakes(
        monkeypatch,
        latest_row={"Close": np.nan, "Volume": np.inf},
    )

    try:
        predictor.predict_price("AAPL")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid latest features")

    assert "Close" in message
    assert "Volume" in message
