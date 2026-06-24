"""Central project configuration for training, artifacts, and feature contracts."""

# Training/evaluation defaults shared by trainer.py.
LOG_FILE = 'training.log'
EARLY_STOPPING_ROUNDS = 20
PREDICTION_DAYS = 5
TEST_SIZE = 0.2
CV_FOLDS = 5

# XGBoost model parameters. Keep changes deliberate because they affect comparability.
XG_PARAMS_CLASSIFIER = {
    'n_estimators': 1000,
    'max_depth': 5,
    'learning_rate': 0.05,
    'tree_method': 'hist',
    'device': 'cpu',
    'verbosity': 0,
    'scale_pos_weight': 1.2, 
    'early_stopping_rounds': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.05,
    'reg_alpha': 0.5,
    'reg_lambda': 5.0
}
XG_PARAMS_REGRESSOR = {
    'n_estimators': 1000,
    'max_depth': 5,
    'learning_rate': 0.05,
    'tree_method': 'hist',
    'device': 'cpu',
    'verbosity': 0,
    'scale_pos_weight': 1.2, 
    'early_stopping_rounds': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.05,
    'reg_alpha': 0.5,
    'reg_lambda': 5.0
}

# Artifact keys are compatibility-sensitive: trainer.py writes them and predictor.py reads them.
MODEL_PATHS = {
    'linear': 'models/linear_regression_model.pkl',
    'linear_scaler': 'models/linear_regression_scaler.pkl',
    'classifier': 'models/xgboost_classifier.pkl',
    'regressor': 'models/xgboost_regressor.pkl',
    'preparator': 'models/data_preparator.pkl',
    'features': 'models/feature_names.pkl',
    'model_metadata': 'models/model_metadata.pkl'
}

# Default training universe used by trainer.py; not a guarantee of robustness or uniqueness.
TRAINING_TICKERS = [
    "SPY", "XLU", "XLE", "XLV", "VOOG",
    "VOOV", "VB", "TSLA", "LAD", "JPM",
    "AAPL", "SCHW", "NVDA", "META", "GE",
    "INTC", "BX", "KR", "FDX", "MSFT",
    "CVX", "PSX", "PG", "PFE", "AMGN",
    "BRK-A", "BRK-B", "RCL", "AMZN", "HD",
    "DUK", "NEE", "GS", "CAT", "MCD",
    "QQQ", "VTI", "IWM", "XLF", "XLK",
    "V", "MA", "GOOGL", "COST", "WMT",
    "UNH", "JNJ", "WFC", "CRM", "ADBE",
    "AMD", "PYPL", "BA", "MMM", "UPS",
    "VZ", "T", "MRK", "ABBV", "ABT",
    "NOW", "CL", "IBM", "CSCO", "HON",
    "MU", "QCOM", "UBER", "MRNA", "SQ",
    "NFLX", "CMG", "GM", "F", "BAC",
    "C", "MS", "AMAT", "SBUX", "DELL",
    "SNOW", "AVGO", "CCI", "DIA", "XBI",
    "XTL", "XRT", "KO", "PEP", "MDT",
    "BABA", "LMT", "ORCL", "TMUS", "ROKU",
    "COIN", "Z", "SHOP", "LYFT", "WDC",
    "PANW", "ZS", "CRM", "DDOG", "NET",
    "PLTR", "SYK", "DHR", "LLY", "ISRG",
    "TXN", "CRWD", "ZM", "PINS", "OKTA",
    "NOC", "GD", "RTX", "BAESY", "TXT", 
    "NVMI", "NVST", "QQQM", "ADI", "FTNT", 
    "OXY", "XOM", "EOG", "HAL", "SLB", 
    "PM", "CLX", "STZ", "HSY", "MKC", 
    "USB", "TROW", "PRU", "MET", "CME", 
    "BDX", "ZBH", "BMRN", "BIIB", "ILMN", 
    "O", "SPG", "AMT", "PLD", "EQIX"
    ]

# Shared feature contract after technical indicator generation and before model training/prediction.
REQUIRED_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    '5_day_avg', '10_day_avg', '20_day_avg',
    'dailyReturn', 'volatility', 'rsi',
    'macd', 'signalLine', 'macdHistogram',
    'obv', 'vma_10', 'vma_20',
    'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
    'chikou_lag_close_26', 'chikou_return_26', 'chikou_above_lag_26',
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Std',  # Bollinger Bands
    'ATR',  # Average True Range
    'stoch_k', 'stoch_d'  # Stochastic Oscillator
]
