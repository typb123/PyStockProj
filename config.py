LOG_FILE = 'training.log'
EARLY_STOPPING_ROUNDS = 20
PREDICTION_DAYS = 5
TEST_SIZE = 0.2
CV_FOLDS = 5
XG_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 5,
    'learning_rate': 0.05,
    'tree_method': 'hist',
    'device': 'cuda',
    'verbosity': 2,
    'scale_pos_weight': 1.2, 
    'early_stopping_rounds': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.05,
    'reg_alpha': 0.5,
    'reg_lambda': 5.0
}

MODEL_PATHS = {
    'linear': 'models/linear_regression_model.pkl',
    'classifier': 'models/xgboost_classifier.pkl',
    'regressor': 'models/xgboost_regressor.pkl',
    'preparator': 'models/data_preparator.pkl',
    'features': 'models/feature_names.pkl'
}
#These are the 150 tickers I am currently training on
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
REQUIRED_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    '5_day_avg', '10_day_avg', '20_day_avg',
    'dailyReturn', 'volatility', 'rsi',
    'macd', 'signalLine', 'macdHistogram',
    'obv', 'vma_10', 'vma_20',
    'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Std',  # Bollinger Bands
    'ATR',  # Average True Range
    'stoch_k', 'stoch_d'  # Stochastic Oscillator
]
