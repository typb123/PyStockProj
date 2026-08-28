"""Central project configuration for training, artifacts, and research defaults."""

from src.features.feature_contract import (
    ABSOLUTE_MOMENTUM_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS,
)

# Training/evaluation defaults shared by trainer.py.
LOG_FILE = "training.log"
PREDICTION_DAYS = 10
TEST_SIZE = 0.2

# XGBoost model parameters. Keep changes deliberate because they affect comparability.
XG_PARAMS_CLASSIFIER = {
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": -1,
    "verbosity": 0,
    "scale_pos_weight": 1.2,
    "early_stopping_rounds": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "random_state": 42,
}
XG_PARAMS_REGRESSOR = {
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.05,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "random_state": 42,
}

# New training runs publish immutable bundles beneath this root.
MODEL_ARTIFACT_ROOT = "models"

# Legacy inference-only paths. New training runs intentionally do not write these;
# predictor migration belongs to the ranked-watchlist inference milestone.
MODEL_PATHS = {
    "linear": "models/linear_regression_model.pkl",
    "linear_scaler": "models/linear_regression_scaler.pkl",
    "classifier": "models/xgboost_classifier.json",
    "regressor": "models/xgboost_regressor.json",
    "preparator": "models/data_preparator.pkl",
    "features": "models/feature_names.pkl",
    "model_metadata": "models/model_metadata.pkl",
}

# Original mixed training universe retained for reference/comparison.
CURRENT_MIXED_UNIVERSE = [
    "SPY",
    "XLU",
    "XLE",
    "XLV",
    "VOOG",
    "VOOV",
    "VB",
    "TSLA",
    "LAD",
    "JPM",
    "AAPL",
    "SCHW",
    "NVDA",
    "META",
    "GE",
    "INTC",
    "BX",
    "KR",
    "FDX",
    "MSFT",
    "CVX",
    "PSX",
    "PG",
    "PFE",
    "AMGN",
    "BRK-A",
    "BRK-B",
    "RCL",
    "AMZN",
    "HD",
    "DUK",
    "NEE",
    "GS",
    "CAT",
    "MCD",
    "QQQ",
    "VTI",
    "IWM",
    "XLF",
    "XLK",
    "V",
    "MA",
    "GOOGL",
    "COST",
    "WMT",
    "UNH",
    "JNJ",
    "WFC",
    "CRM",
    "ADBE",
    "AMD",
    "PYPL",
    "BA",
    "MMM",
    "UPS",
    "VZ",
    "T",
    "MRK",
    "ABBV",
    "ABT",
    "NOW",
    "CL",
    "IBM",
    "CSCO",
    "HON",
    "MU",
    "QCOM",
    "UBER",
    "MRNA",
    "XYZ",
    "NFLX",
    "CMG",
    "GM",
    "F",
    "BAC",
    "C",
    "MS",
    "AMAT",
    "SBUX",
    "DELL",
    "SNOW",
    "AVGO",
    "CCI",
    "DIA",
    "XBI",
    "XTL",
    "XRT",
    "KO",
    "PEP",
    "MDT",
    "BABA",
    "LMT",
    "ORCL",
    "TMUS",
    "ROKU",
    "COIN",
    "Z",
    "SHOP",
    "LYFT",
    "WDC",
    "PANW",
    "ZS",
    "DDOG",
    "NET",
    "PLTR",
    "SYK",
    "DHR",
    "LLY",
    "ISRG",
    "TXN",
    "CRWD",
    "ZM",
    "PINS",
    "OKTA",
    "NOC",
    "GD",
    "RTX",
    "BAESY",
    "TXT",
    "NVMI",
    "NVST",
    "QQQM",
    "ADI",
    "FTNT",
    "OXY",
    "XOM",
    "EOG",
    "HAL",
    "SLB",
    "PM",
    "CLX",
    "STZ",
    "HSY",
    "MKC",
    "USB",
    "TROW",
    "PRU",
    "MET",
    "CME",
    "BDX",
    "ZBH",
    "BMRN",
    "BIIB",
    "ILMN",
    "O",
    "SPG",
    "AMT",
    "PLD",
    "EQIX",
]

DEFAULT_TRAINING_UNIVERSE = "large_mega_cap_stocks"

LARGE_MEGA_CAP_STOCKS = [
    "SPY",  # benchmark only; candidate rows should still exclude SPY later
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA",
    "BRK-B", "JPM", "V", "MA", "UNH", "LLY", "XOM", "COST",
    "WMT", "HD", "PG", "JNJ", "ABBV", "MRK", "CVX", "KO",
    "PEP", "BAC", "WFC", "GS", "MS", "CAT", "GE", "HON",
    "UPS", "BA", "LMT", "RTX", "NOC", "ORCL", "CRM", "ADBE",
    "CSCO", "AMD", "QCOM", "TXN", "INTC", "IBM", "AMGN", "PFE",
    "ABT", "DHR", "ISRG", "SYK", "MCD", "PM", "SBUX", "CMG",
    "NFLX", "VZ", "T", "NEE", "DUK", "AMT", "PLD", "EQIX",
]

BROAD_SECTOR_ETFS = [
    "SPY",  # benchmark
    "QQQ", "DIA", "IWM", "VTI",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
    "XLI", "XLB", "XLRE", "XLC", "XLU",
    "VUG", "VTV", "MTUM",
]

TRAINING_UNIVERSES = {
    "large_mega_cap_stocks": LARGE_MEGA_CAP_STOCKS,
    "broad_sector_etfs": BROAD_SECTOR_ETFS,
    "current_mixed": CURRENT_MIXED_UNIVERSE,
}


def get_training_tickers(
    universe_name: str = DEFAULT_TRAINING_UNIVERSE,
) -> list[str]:
    """Return a copy of the requested training universe ticker list."""
    try:
        tickers = TRAINING_UNIVERSES[universe_name]
    except KeyError as exc:
        valid_universes = ", ".join(sorted(TRAINING_UNIVERSES))
        raise ValueError(
            f"Unknown training universe '{universe_name}'. "
            f"Valid universes: {valid_universes}."
        ) from exc
    return list(tickers)


TRAINING_TICKERS = get_training_tickers(DEFAULT_TRAINING_UNIVERSE)

# Compatibility aliases for older imports; new code should import feature columns
# from src.features.feature_contract.
MOMENTUM_FEATURE_COLUMNS = ABSOLUTE_MOMENTUM_FEATURE_COLUMNS
RELATIVE_MOMENTUM_FEATURE_COLUMNS = SPY_RELATIVE_MOMENTUM_FEATURE_COLUMNS
