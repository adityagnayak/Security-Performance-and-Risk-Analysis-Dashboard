"""
Configuration file for Security Analysis Dashboard
Customize these settings for your specific needs
"""

# Default Analysis Settings
DEFAULT_PERIOD = "5y"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_RISK_FREE_RATE = 0.04  # 4% annual
DEFAULT_VAR_CONFIDENCE = 0.95  # 95%

# Benchmark Options
BENCHMARK_OPTIONS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "NASDAQ 100 ETF",
    "DIA": "Dow Jones ETF",
    "IWM": "Russell 2000 ETF",
    "^GSPC": "S&P 500 Index",
    "^DJI": "Dow Jones Index",
    "^IXIC": "NASDAQ Composite",
    "EFA": "MSCI EAFE ETF",
    "EEM": "Emerging Markets ETF",
    "AGG": "US Aggregate Bond ETF"
}

# Period Options
PERIOD_OPTIONS = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "3Y": "3y",
    "5Y": "5y",
    "10Y": "10y",
    "Max": "max"
}

# Moving Average Windows
MA_WINDOWS = [50, 100, 200]

# Chart Settings
CHART_TEMPLATE = "plotly_white"
CHART_HEIGHT_MAIN = 600
CHART_HEIGHT_SECONDARY = 400
COLOR_SCHEME = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "warning": "#d62728",
    "success": "#2ca02c"
}

# Performance Metrics Display
SHOW_METRICS = {
    "beta": True,
    "alpha": True,
    "sharpe": True,
    "sortino": True,
    "volatility": True,
    "max_drawdown": True,
    "correlation": True,
    "information_ratio": True,
    "calmar_ratio": True
}

# Fundamental Metrics Display
SHOW_FUNDAMENTALS = {
    "market_cap": True,
    "pe_ratio": True,
    "peg_ratio": True,
    "price_to_book": True,
    "dividend_yield": True,
    "profit_margin": True,
    "roe": True,
    "roa": True,
    "debt_to_equity": True,
    "current_ratio": True,
    "revenue_growth": True,
    "earnings_growth": True
}

# VaR Settings
VAR_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
VAR_TIME_HORIZON = 1  # days

# Number Formatting
DECIMAL_PLACES = {
    "percentage": 2,
    "ratio": 3,
    "currency": 2,
    "general": 4
}

# Data Caching
CACHE_TTL = 3600  # seconds (1 hour)
ENABLE_CACHING = True

# API Settings
YFINANCE_TIMEOUT = 30  # seconds
MAX_RETRIES = 3

# Export Settings
EXPORT_FORMATS = ["CSV", "Excel", "PDF"]
INCLUDE_CHARTS_IN_EXPORT = True

# Advanced Features (Future Implementation)
ENABLE_MONTE_CARLO = False
ENABLE_OPTIONS_ANALYTICS = False
ENABLE_SENTIMENT_ANALYSIS = False
ENABLE_ML_PREDICTIONS = False

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "dashboard.log"
ENABLE_LOGGING = False

# UI Customization
SIDEBAR_STATE = "expanded"
LAYOUT = "wide"
PAGE_ICON = "📊"
PAGE_TITLE = "Security Performance & Risk Analysis"

# Tooltips and Help Text
HELP_TEXT = {
    "beta": "Measures systematic risk relative to the benchmark. β>1 indicates higher volatility than market.",
    "alpha": "Risk-adjusted excess return. Positive alpha indicates outperformance after accounting for risk.",
    "sharpe": "Return per unit of total risk. Higher values indicate better risk-adjusted performance.",
    "sortino": "Similar to Sharpe but only considers downside volatility.",
    "var": "Maximum expected loss over one day at given confidence level.",
    "cvar": "Average loss in worst-case scenarios beyond VaR threshold."
}

# Data Validation
MIN_DATA_POINTS = 30  # Minimum trading days required
MIN_VOLUME = 10000  # Minimum average daily volume

# Performance Thresholds (for color coding)
THRESHOLDS = {
    "sharpe_good": 1.0,
    "sharpe_excellent": 2.0,
    "sortino_good": 1.5,
    "sortino_excellent": 2.5,
    "max_drawdown_warning": -0.20,  # -20%
    "max_drawdown_critical": -0.40,  # -40%
}
