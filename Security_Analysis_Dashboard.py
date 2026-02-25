"""
Security Performance & Risk Analysis Dashboard
Institutional-grade financial analysis tool for equity securities.
Supports global markets, dual-mode interface, and comprehensive risk metrics.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Security Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  :root {
    --bg: #ffffff;
    --bg2: #f4f5f7;
    --surface: #ffffff;
    --border: #d8dde6;
    --text: #111827;
    --text2: #4b5563;
    --accent: #8b1a1a;
    --accent2: #166534;
    --accent3: #1e3a8a;
    --card-bg: #1c2b3a;
    --card-label: #94a3b8;
    --card-value: #f1f5f9;
    --card-shadow: 0 2px 8px rgba(0,0,0,0.10);
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1117;
      --bg2: #161b26;
      --surface: #1c2333;
      --border: #2d3748;
      --text: #e2e8f0;
      --text2: #94a3b8;
      --accent: #ef4444;
      --accent2: #22c55e;
      --accent3: #60a5fa;
      --card-bg: #1c2b3a;
      --card-label: #94a3b8;
      --card-value: #f1f5f9;
      --card-shadow: 0 2px 8px rgba(0,0,0,0.45);
    }
  }

  html, body, [class*="css"] {
    font-family: Helvetica, 'Helvetica Neue', Arial, sans-serif !important;
    color: var(--text) !important;
    background-color: var(--bg) !important;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, .stDeployButton { display: none !important; }
  .block-container { padding-top: 1.5rem !important; padding-bottom: 5rem !important; }

  /* Main header */
  .dashboard-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 2rem;
  }
  .dashboard-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.2;
    font-family: Helvetica, 'Helvetica Neue', Arial, sans-serif;
  }
  .dashboard-header .subtitle {
    font-size: 0.9rem;
    color: var(--text2);
    margin-top: 0.4rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }

  /* ── Company info cards (dark slate style) ── */
  .company-cards-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.4rem;
    flex-wrap: wrap;
  }
  .company-card {
    flex: 1;
    min-width: 160px;
    background: var(--card-bg);
    border: 1px solid #2d3f52;
    border-left: 3px solid #3b6ea5;
    border-radius: 6px;
    padding: 0.85rem 1.1rem 0.9rem;
    box-shadow: var(--card-shadow);
  }
  .company-card-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--card-label);
    margin-bottom: 0.35rem;
    font-weight: 500;
  }
  .company-card-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--card-value);
    line-height: 1.15;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ── Price overview cards ── */
  .price-cards-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
  }
  .price-card {
    flex: 1;
    min-width: 160px;
    background: var(--card-bg);
    border: 1px solid #2d3f52;
    border-radius: 6px;
    padding: 0.85rem 1.1rem 0.9rem;
    box-shadow: var(--card-shadow);
  }
  .price-card-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--card-label);
    margin-bottom: 0.35rem;
    font-weight: 500;
  }
  .price-card-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--card-value);
    line-height: 1.15;
  }
  .price-card.highlight { border-left: 3px solid #f59e0b; }

  /* ── Risk metric cards ── */
  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent3);
    border-radius: 5px;
    padding: 0.9rem 1.1rem;
    box-shadow: var(--card-shadow);
    margin-bottom: 1rem;
    transition: box-shadow 0.2s;
  }
  .metric-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.14); }
  .metric-card.green { border-left-color: var(--accent2); }
  .metric-card.blue  { border-left-color: var(--accent3); }
  .metric-card.red   { border-left-color: var(--accent); }

  .metric-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text2);
    margin-bottom: 0.3rem;
    font-weight: 600;
  }
  .metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
  }
  .metric-tooltip {
    font-size: 0.72rem;
    color: var(--text2);
    margin-top: 0.25rem;
  }

  /* Section headers */
  .section-header {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.45rem;
    margin: 1.6rem 0 1rem;
  }

  /* Warning/info boxes */
  .warn-box {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-left: 4px solid #d97706;
    border-radius: 5px;
    padding: 0.85rem 1.1rem;
    margin: 0.8rem 0;
    color: #78350f;
    font-size: 0.88rem;
  }
  @media (prefers-color-scheme: dark) {
    .warn-box { background: #1c1500; border-color: #78350f; color: #fcd34d; }
  }
  .info-box {
    background: #eff6ff;
    border: 1px solid #93c5fd;
    border-left: 4px solid #3b82f6;
    border-radius: 5px;
    padding: 0.85rem 1.1rem;
    margin: 0.8rem 0;
    color: #1e3a8a;
    font-size: 0.88rem;
  }
  @media (prefers-color-scheme: dark) {
    .info-box { background: #0c1a30; border-color: #1d4ed8; color: #93c5fd; }
  }

  /* Fundamentals table */
  .fund-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
  .fund-table tr { border-bottom: 1px solid var(--border); }
  .fund-table tr:last-child { border-bottom: none; }
  .fund-table td { padding: 0.48rem 0.7rem; color: var(--text); }
  .fund-table td:first-child { color: var(--text2); width: 55%; font-size: 0.83rem; }
  .fund-table td:last-child { font-weight: 700; text-align: right; }

  /* Disclaimer */
  .disclaimer {
    position: fixed;
    bottom: 0;
    left: 6rem;
    right: 6rem;
    background: #cc0000;
    color: #ffffff;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 0.5rem 1.5rem;
    text-align: center;
    z-index: 9999;
    letter-spacing: 0.02em;
    font-family: Helvetica, Arial, sans-serif;
  }

  /* Mobile responsive */
  @media (max-width: 768px) {
    .dashboard-header h1 { font-size: 1.4rem; }
    .metric-value { font-size: 1.25rem; }
    .company-card-value { font-size: 1rem; }
    .price-card-value { font-size: 1.1rem; }
    .company-cards-row, .price-cards-row { flex-direction: column; }
    button, .stButton > button { min-height: 44px !important; font-size: 16px !important; }
    body, p, div { font-size: 16px !important; }
    .disclaimer { position: relative; margin-top: 2rem; }
  }

  /* Streamlit button styling */
  .stButton > button {
    font-family: Helvetica, Arial, sans-serif !important;
    border-radius: 4px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] label {
    font-size: 0.82rem !important;
    color: var(--text2) !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* VaR table */
  .var-wrapper { overflow-x: auto; margin: 0.5rem 0 0.8rem; border-radius: 5px; border: 1px solid var(--border); }
  .var-table { width:100%; border-collapse:collapse; font-size:0.86rem; min-width:420px; }
  .var-table th {
    background: var(--bg2);
    color: var(--text2);
    padding: 0.55rem 0.9rem;
    text-align: center;
    border-bottom: 2px solid var(--border);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
  }
  .var-table td { padding: 0.5rem 0.9rem; text-align: center; border-bottom: 1px solid var(--border); color: var(--text); }
  .var-table tr:last-child td { border-bottom: none; }
  .var-table tr:hover td { background: var(--bg2); }

  /* Performance table */
  .perf-wrapper { overflow-x: auto; margin: 0.5rem 0; border-radius: 5px; border: 1px solid var(--border); }
  .perf-table { width:100%; border-collapse:collapse; font-size:0.86rem; min-width:500px; }
  .perf-table th {
    background: var(--bg2);
    color: var(--text2);
    padding: 0.55rem 0.9rem;
    text-align: left;
    border-bottom: 2px solid var(--border);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
  }
  .perf-table td { padding: 0.5rem 0.9rem; border-bottom: 1px solid var(--border); color: var(--text); }
  .perf-table tr:last-child td { border-bottom: none; }
  .perf-table tr:hover td { background: var(--bg2); }

  /* Mode toggle */
  .mode-badge {
    display: inline-block;
    padding: 0.2rem 0.9rem;
    border-radius: 3px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
  }
  .mode-basic    { background: #f0f4f8; color: #4a5568; border: 1px solid #cbd5e0; }
  .mode-advanced { background: #1e2d42; color: #93c5fd; border: 1px solid #2d4a6e; }

  /* Price section header */
  .price-section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.9rem;
    margin-top: 0.2rem;
  }
  .price-section-title .icon { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
BENCHMARKS = {
    "US": {"SPY": "S&P 500 (SPY)", "QQQ": "NASDAQ 100 (QQQ)", "DIA": "Dow Jones (DIA)",
           "IWM": "Russell 2000 (IWM)", "^GSPC": "S&P 500 Index", "^DJI": "Dow Jones Index",
           "^IXIC": "NASDAQ Index"},
    "India": {"^NSEI": "Nifty 50", "^BSESN": "BSE Sensex"},
    "Global": {"^FTSE": "FTSE 100 (UK)", "^N225": "Nikkei 225 (Japan)",
               "^HSI": "Hang Seng (HK)", "^AXJO": "ASX 200 (Australia)",
               "^GSPTSE": "TSX (Canada)", "^FCHI": "CAC 40 (France)",
               "^GDAXI": "DAX (Germany)", "^SSMI": "SMI (Switzerland)"}
}

ALL_BENCHMARKS = {}
for region in BENCHMARKS.values():
    ALL_BENCHMARKS.update(region)

CURRENCY_MAP = {
    'USD': '$', 'EUR': '€', 'GBP': '£', 'GBp': 'p', 'JPY': '¥', 'CNY': '¥',
    'INR': '₹', 'AUD': 'A$', 'CAD': 'C$', 'CHF': 'CHF ', 'HKD': 'HK$',
    'SGD': 'S$', 'KRW': '₩', 'BRL': 'R$', 'RUB': '₽', 'ZAR': 'R',
    'MXN': 'MX$', 'SEK': 'kr', 'NOK': 'kr', 'DKK': 'kr', 'PLN': 'zł',
    'TRY': '₺', 'THB': '฿', 'IDR': 'Rp', 'MYR': 'RM', 'PHP': '₱',
    'NZD': 'NZ$', 'ILS': '₪', 'AED': 'AED ', 'SAR': 'SAR '
}

EXCHANGE_BENCHMARK = {
    '.L': '^FTSE', '.NS': '^NSEI', '.BO': '^BSESN', '.T': '^N225',
    '.HK': '^HSI', '.AX': '^AXJO', '.TO': '^GSPTSE', '.PA': '^FCHI',
    '.DE': '^GDAXI', '.SW': '^SSMI'
}

EXCHANGE_CURRENCY = {
    '.L': 'GBp', '.NS': 'INR', '.BO': 'INR', '.T': 'JPY',
    '.HK': 'HKD', '.AX': 'AUD', '.TO': 'CAD', '.PA': 'EUR',
    '.DE': 'EUR', '.SW': 'CHF'
}

EXCHANGE_NAME = {
    '.L': 'London Stock Exchange', '.NS': 'NSE India', '.BO': 'BSE India',
    '.T': 'Tokyo Stock Exchange', '.HK': 'Hong Kong Stock Exchange',
    '.AX': 'Australian Securities Exchange', '.TO': 'Toronto Stock Exchange',
    '.PA': 'Euronext Paris', '.DE': 'Deutsche Börse', '.SW': 'SIX Swiss Exchange'
}

BENCHMARK_NAMES = {
    '^FTSE': 'FTSE 100', '^NSEI': 'Nifty 50', '^BSESN': 'BSE Sensex',
    '^N225': 'Nikkei 225', '^HSI': 'Hang Seng', '^AXJO': 'ASX 200',
    '^GSPTSE': 'TSX', '^FCHI': 'CAC 40', '^GDAXI': 'DAX', '^SSMI': 'SMI',
    'SPY': 'S&P 500 (SPY)', 'QQQ': 'NASDAQ 100 (QQQ)', 'DIA': 'Dow Jones (DIA)',
    'IWM': 'Russell 2000 (IWM)', '^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ'
}


# ─── SecurityAnalyzer Class ────────────────────────────────────────────────────
class SecurityAnalyzer:
    """
    Institutional-grade security analysis engine.
    Encapsulates all data fetching and metric calculations.
    """

    def __init__(self, ticker: str, benchmark: str = "SPY"):
        self.ticker = ticker.upper().strip()
        self.benchmark = benchmark
        self.data = None
        self.benchmark_data = None
        self.info = {}
        self.returns = None
        self.benchmark_returns = None
        self.benchmark_warning = None

    def fetch_data(self, period: str = "5y", start_date=None, end_date=None) -> bool:
        """
        Fetch stock and benchmark data with retry logic.
        Returns True on success, False on failure.
        """
        for attempt in range(3):
            try:
                if attempt > 0:
                    delay = attempt * 2
                    st.info(f"⏳ Rate limited. Retrying in {delay}s... (attempt {attempt+1}/3)")
                    time.sleep(delay)

                ticker_obj = yf.Ticker(self.ticker)

                # Fetch price data
                if start_date and end_date:
                    self.data = ticker_obj.history(start=start_date, end=end_date)
                else:
                    self.data = ticker_obj.history(period=period)

                if self.data is None or self.data.empty:
                    if attempt == 2:
                        return False
                    continue

                # Fetch info only after price data succeeds
                try:
                    raw_info = ticker_obj.info
                    self.info = raw_info if raw_info and len(raw_info) > 1 else {}
                except Exception:
                    self.info = {}

                # Fetch benchmark data
                bench_obj = yf.Ticker(self.benchmark)
                if start_date and end_date:
                    self.benchmark_data = bench_obj.history(start=start_date, end=end_date)
                else:
                    self.benchmark_data = bench_obj.history(period=period)

                if self.benchmark_data is None or self.benchmark_data.empty:
                    if attempt == 2:
                        return False
                    continue

                # Calculate returns
                self.returns = self.data['Close'].pct_change().dropna()
                self.benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
                return True

            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'too many requests' in err_str:
                    if attempt < 2:
                        continue
                if attempt == 2:
                    st.error(f"Data fetch failed: {e}")
                    return False
        return False

    def get_currency(self) -> str:
        """Detect currency with fallback to ticker suffix inference."""
        if self.info:
            currency = self.info.get('currency')
            if currency:
                return currency

        # Fallback: infer from ticker suffix
        for suffix, currency in EXCHANGE_CURRENCY.items():
            if suffix in self.ticker:
                return currency
        return 'USD'

    def get_currency_symbol(self) -> str:
        """Return currency symbol string."""
        currency = self.get_currency()
        return CURRENCY_MAP.get(currency, currency + ' ')

    def get_exchange_suffix(self) -> str:
        """Return exchange suffix from ticker."""
        for suffix in EXCHANGE_BENCHMARK.keys():
            if suffix in self.ticker:
                return suffix
        return ''

    def format_number(self, value, prefix: str = '', suffix: str = '',
                       decimals: int = 2, is_pct: bool = False,
                       is_large: bool = False) -> str:
        """Universal number formatter."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'N/A'
        try:
            if is_large:
                if abs(value) >= 1e12:
                    return f"{prefix}{value/1e12:.2f}T{suffix}"
                elif abs(value) >= 1e9:
                    return f"{prefix}{value/1e9:.2f}B{suffix}"
                elif abs(value) >= 1e6:
                    return f"{prefix}{value/1e6:.2f}M{suffix}"
                else:
                    return f"{prefix}{value:,.0f}{suffix}"
            if is_pct:
                return f"{value*100:.{decimals}f}%"
            return f"{prefix}{value:,.{decimals}f}{suffix}"
        except Exception:
            return 'N/A'

    # ── Core Metrics ───────────────────────────────────────────────────────────

    def _aligned_returns(self):
        """Return aligned stock and benchmark return series."""
        merged = pd.concat([self.returns, self.benchmark_returns], axis=1, join='inner')
        merged.columns = ['stock', 'bench']
        return merged.dropna()

    def calculate_beta(self) -> float:
        """Calculate beta (systematic risk vs benchmark)."""
        merged = self._aligned_returns()

        if len(merged) == 0:
            self.benchmark_warning = (
                f"No overlapping trading days between {self.ticker} and {self.benchmark}. "
                "Please use a benchmark from the same market/exchange."
            )
            return np.nan

        if len(merged) < 30:
            self.benchmark_warning = (
                f"Only {len(merged)} overlapping trading days — metrics may be unreliable."
            )

        try:
            cov_matrix = np.cov(merged['stock'], merged['bench'])
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            if np.isnan(beta):
                self.benchmark_warning = "Beta is NaN — benchmark variance is zero."
            return float(beta)
        except Exception:
            return np.nan

    def calculate_alpha(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Jensen's Alpha."""
        beta = self.calculate_beta()
        if np.isnan(beta):
            return np.nan
        merged = self._aligned_returns()
        if len(merged) == 0:
            return np.nan
        ann_factor = 252
        stock_ann = (1 + merged['stock'].mean()) ** ann_factor - 1
        bench_ann = (1 + merged['bench'].mean()) ** ann_factor - 1
        return stock_ann - (risk_free_rate + beta * (bench_ann - risk_free_rate))

    def calculate_annualized_return(self) -> float:
        """Calculate annualized return."""
        if self.returns is None or len(self.returns) == 0:
            return np.nan
        n_years = len(self.returns) / 252
        total_return = (1 + self.returns).prod() - 1
        if n_years <= 0:
            return np.nan
        return (1 + total_return) ** (1 / n_years) - 1

    def calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        if self.returns is None or len(self.returns) == 0:
            return np.nan
        return float(self.returns.std() * np.sqrt(252))

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if self.data is None or self.data.empty:
            return np.nan
        prices = self.data['Close']
        roll_max = prices.cummax()
        drawdown = (prices - roll_max) / roll_max
        return float(drawdown.min())

    def calculate_drawdown_series(self) -> pd.Series:
        """Return full drawdown series."""
        prices = self.data['Close']
        roll_max = prices.cummax()
        return (prices - roll_max) / roll_max

    def calculate_sharpe(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe Ratio."""
        if self.returns is None:
            return np.nan
        daily_rf = risk_free_rate / 252
        excess = self.returns - daily_rf
        if excess.std() == 0:
            return np.nan
        return float((excess.mean() / excess.std()) * np.sqrt(252))

    def calculate_sortino(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sortino Ratio (downside risk only)."""
        if self.returns is None:
            return np.nan
        daily_rf = risk_free_rate / 252
        excess = self.returns - daily_rf
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return np.nan
        return float((excess.mean() / downside.std()) * np.sqrt(252))

    def calculate_information_ratio(self) -> float:
        """Calculate Information Ratio (active return / tracking error)."""
        merged = self._aligned_returns()
        if len(merged) == 0:
            return np.nan
        active = merged['stock'] - merged['bench']
        if active.std() == 0:
            return np.nan
        return float((active.mean() / active.std()) * np.sqrt(252))

    def calculate_calmar(self) -> float:
        """Calculate Calmar Ratio (return / max drawdown)."""
        ann_ret = self.calculate_annualized_return()
        max_dd = self.calculate_max_drawdown()
        if np.isnan(ann_ret) or np.isnan(max_dd) or max_dd == 0:
            return np.nan
        return float(ann_ret / abs(max_dd))

    def calculate_correlation(self) -> float:
        """Calculate correlation with benchmark."""
        merged = self._aligned_returns()
        if len(merged) < 2:
            return np.nan
        return float(merged['stock'].corr(merged['bench']))

    def calculate_skewness(self) -> float:
        """Calculate return distribution skewness."""
        if self.returns is None:
            return np.nan
        return float(stats.skew(self.returns.dropna()))

    def calculate_kurtosis(self) -> float:
        """Calculate excess kurtosis."""
        if self.returns is None:
            return np.nan
        return float(stats.kurtosis(self.returns.dropna()))

    def calculate_var(self, confidence: float = 0.95) -> dict:
        """
        Calculate Value at Risk metrics.
        Returns Historical VaR, Parametric VaR, and CVaR.
        """
        if self.returns is None or len(self.returns) == 0:
            return {'hist': np.nan, 'param': np.nan, 'cvar': np.nan}

        r = self.returns.dropna().values
        alpha = 1 - confidence

        # Historical VaR
        hist_var = float(np.percentile(r, alpha * 100))

        # Parametric VaR (assumes normal distribution)
        z = stats.norm.ppf(alpha)
        param_var = float(r.mean() + z * r.std())

        # CVaR (Expected Shortfall)
        cvar = float(r[r <= hist_var].mean()) if len(r[r <= hist_var]) > 0 else hist_var

        return {'hist': hist_var, 'param': param_var, 'cvar': cvar}

    def get_moving_averages(self) -> dict:
        """Calculate 50, 100, 200-day moving averages."""
        close = self.data['Close']
        return {
            'ma50': close.rolling(50).mean(),
            'ma100': close.rolling(100).mean(),
            'ma200': close.rolling(200).mean()
        }

    def get_cumulative_returns(self) -> tuple:
        """Return cumulative return series for stock and benchmark."""
        stock_cum = (1 + self.returns).cumprod() - 1
        bench_cum = (1 + self.benchmark_returns).cumprod() - 1
        return stock_cum, bench_cum

    def get_fundamentals(self, mode: str = 'basic') -> dict:
        """Extract fundamental metrics from info dict."""
        info = self.info

        def safe(key, scale=1, is_pct=False, is_large=False):
            val = info.get(key)
            if val is None:
                return 'N/A'
            try:
                v = float(val) / scale
                if is_pct:
                    return f"{v*100:.2f}%"
                if is_large:
                    return self.format_number(v, is_large=True)
                return f"{v:.2f}"
            except Exception:
                return 'N/A'

        sym = self.get_currency_symbol()
        mktcap = info.get('marketCap')

        basic = {
            'Market Cap': self.format_number(mktcap, prefix=sym, is_large=True) if mktcap else 'N/A',
            'P/E Ratio': safe('trailingPE'),
            'Dividend Yield': safe('dividendYield', is_pct=True),
            'ROE': safe('returnOnEquity', is_pct=True),
        }

        if mode == 'basic':
            return basic

        advanced = {
            **basic,
            'Forward P/E': safe('forwardPE'),
            'PEG Ratio': safe('pegRatio'),
            'Price/Book': safe('priceToBook'),
            'ROA': safe('returnOnAssets', is_pct=True),
            'Profit Margin': safe('profitMargins', is_pct=True),
            'Debt/Equity': safe('debtToEquity'),
            'Current Ratio': safe('currentRatio'),
            'Revenue Growth': safe('revenueGrowth', is_pct=True),
            'Earnings Growth': safe('earningsGrowth', is_pct=True),
            'EPS (TTM)': safe('trailingEps'),
            '52W High': f"{sym}{info.get('fiftyTwoWeekHigh', 'N/A')}",
            '52W Low': f"{sym}{info.get('fiftyTwoWeekLow', 'N/A')}",
        }
        return advanced

    def check_benchmark_mismatch(self) -> dict | None:
        """Check if selected benchmark matches ticker exchange."""
        suffix = self.get_exchange_suffix()
        if not suffix:
            return None  # US stock — SPY is fine

        suggested = EXCHANGE_BENCHMARK.get(suffix)
        if suggested and suggested != self.benchmark:
            bench_name = BENCHMARK_NAMES.get(suggested, suggested)
            exchange_name = EXCHANGE_NAME.get(suffix, suffix)
            return {
                'exchange': exchange_name,
                'suggested': suggested,
                'suggested_name': bench_name,
                'current': self.benchmark
            }
        return None


# ─── Chart Helpers ─────────────────────────────────────────────────────────────

CHART_LAYOUT = dict(
    font_family="Helvetica, 'Helvetica Neue', Arial, sans-serif",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.12)', zeroline=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.12)', zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    hovermode='x unified',
)


def build_price_chart(analyzer: SecurityAnalyzer) -> go.Figure:
    """Build interactive price chart with MAs and volume."""
    data = analyzer.data
    mas = analyzer.get_moving_averages()
    name = analyzer.ticker

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    # Price line
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        name=name, line=dict(color='#c94040', width=1.8),
        hovertemplate='%{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Moving averages
    for period, color, label in [(50, '#4a7ec4', 'MA50'), (100, '#c48a1a', 'MA100'), (200, '#4a9e42', 'MA200')]:
        ma = mas[f'ma{period}']
        fig.add_trace(go.Scatter(
            x=data.index, y=ma,
            name=label, line=dict(color=color, width=1, dash='dot'),
            hovertemplate=f'{label}: %{{y:.2f}}<extra></extra>'
        ), row=1, col=1)

    # 52W high/low annotations
    high_52 = data['Close'].rolling(252).max()
    low_52 = data['Close'].rolling(252).min()
    fig.add_trace(go.Scatter(
        x=data.index, y=high_52, name='52W High',
        line=dict(color='rgba(74,158,66,0.4)', width=0.8, dash='dash'),
        hovertemplate='52W High: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=low_52, name='52W Low',
        line=dict(color='rgba(201,64,64,0.4)', width=0.8, dash='dash'),
        hovertemplate='52W Low: %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Volume bars
    colors = ['#4a9e42' if c >= o else '#c94040'
              for c, o in zip(data['Close'], data['Open'])]
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'],
        name='Volume', marker_color=colors, opacity=0.65,
        hovertemplate='Vol: %{y:,.0f}<extra></extra>'
    ), row=2, col=1)

    fig.update_layout(**CHART_LAYOUT, title=f'{name} — Price History & Volume', height=500)
    fig.update_yaxes(title_text=f'Price ({analyzer.get_currency_symbol()})', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    return fig


def build_returns_chart(analyzer: SecurityAnalyzer) -> go.Figure:
    """Build cumulative returns comparison chart."""
    stock_cum, bench_cum = analyzer.get_cumulative_returns()
    bench_name = BENCHMARK_NAMES.get(analyzer.benchmark, analyzer.benchmark)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_cum.index, y=stock_cum * 100,
        name=analyzer.ticker, line=dict(color='#c94040', width=2),
        hovertemplate='%{y:.1f}%<extra>' + analyzer.ticker + '</extra>'
    ))
    fig.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum * 100,
        name=bench_name, line=dict(color='#4a7ec4', width=2, dash='dot'),
        hovertemplate='%{y:.1f}%<extra>' + bench_name + '</extra>'
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(128,128,128,0.4)')
    fig.update_layout(**CHART_LAYOUT, title='Cumulative Returns (%)', height=380)
    fig.update_yaxes(title_text='Return (%)')
    return fig


def build_drawdown_chart(analyzer: SecurityAnalyzer) -> go.Figure:
    """Build drawdown chart."""
    dd = analyzer.calculate_drawdown_series() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        name='Drawdown', fill='tozeroy',
        line=dict(color='#c94040', width=1),
        fillcolor='rgba(201,64,64,0.18)',
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    fig.update_layout(**CHART_LAYOUT, title='Drawdown from Peak (%)', height=300)
    fig.update_yaxes(title_text='Drawdown (%)')
    return fig


def build_distribution_chart(analyzer: SecurityAnalyzer) -> go.Figure:
    """Build returns distribution with normal overlay."""
    r = analyzer.returns.dropna() * 100
    mu, sigma = r.mean(), r.std()
    x_range = np.linspace(r.min(), r.max(), 300)
    normal_curve = stats.norm.pdf(x_range, mu, sigma)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=60, name='Daily Returns',
        marker_color='#4a7ec4', opacity=0.7,
        histnorm='probability density',
        hovertemplate='%{x:.2f}%: %{y:.4f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_curve,
        name='Normal Distribution',
        line=dict(color='#c94040', width=2),
        hovertemplate='%{y:.4f}<extra></extra>'
    ))
    fig.add_vline(x=mu, line_dash='dash', line_color='rgba(200,160,0,0.8)',
                  annotation_text=f'Mean: {mu:.2f}%')
    fig.update_layout(**CHART_LAYOUT, title='Daily Returns Distribution', height=340)
    fig.update_xaxes(title_text='Daily Return (%)')
    fig.update_yaxes(title_text='Probability Density')
    return fig


# ─── UI Component Helpers ──────────────────────────────────────────────────────

def metric_card(label: str, value: str, tooltip: str = '', color: str = '') -> str:
    """Render a metric card HTML block."""
    cls = f'metric-card {color}'
    tip = f'<div class="metric-tooltip">{tooltip}</div>' if tooltip else ''
    return f"""
    <div class="{cls}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {tip}
    </div>
    """


def color_for_value(val: float, positive_good: bool = True) -> str:
    """Return card color class based on value sign."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ''
    if positive_good:
        return 'green' if val > 0 else 'red'
    return 'red' if val > 0 else 'green'


def detect_benchmark_suggestion(ticker: str) -> str | None:
    """Return suggested benchmark based on ticker suffix."""
    for suffix, bench in EXCHANGE_BENCHMARK.items():
        if suffix in ticker.upper():
            return bench
    return None


def resolve_ticker(raw: str) -> str:
    """Basic format check; return ticker as-is for yfinance to validate."""
    cleaned = raw.strip().upper()
    if ' ' in cleaned or len(cleaned) == 0:
        return ''
    return cleaned


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return configuration dict."""
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Setup")
        st.markdown("---")

        # Mode selection
        mode = st.radio(
            "Analysis Mode",
            options=["Basic", "Advanced"],
            index=0 if st.session_state.get('mode', 'Basic') == 'Basic' else 1,
            help="Basic: 4 key metrics for new investors. Advanced: 14+ metrics for professionals."
        )
        st.session_state['mode'] = mode

        st.markdown("---")

        # Ticker input
        ticker_input = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get('last_ticker', 'AAPL'),
            placeholder="AAPL, RELIANCE.NS, AZN.L…",
            help="Enter exchange ticker symbol. For Indian stocks use .NS or .BO suffix."
        )

        # Benchmark selection
        suggested_bench = detect_benchmark_suggestion(ticker_input)
        default_bench = suggested_bench or st.session_state.get('last_benchmark', 'SPY')

        st.markdown("**Benchmark**")
        bench_options = list(ALL_BENCHMARKS.keys())
        try:
            default_idx = bench_options.index(default_bench)
        except ValueError:
            default_idx = 0

        benchmark = st.selectbox(
            "Benchmark",
            options=bench_options,
            index=default_idx,
            format_func=lambda x: f"{x} — {ALL_BENCHMARKS.get(x, x)}",
            label_visibility='collapsed'
        )

        if suggested_bench and suggested_bench != benchmark:
            st.markdown(
                f'<div class="info-box">💡 Suggested: <b>{suggested_bench}</b> for this market</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Time period
        period_choice = st.radio(
            "Time Period",
            options=["Preset", "Custom"],
            index=0,
            horizontal=True
        )

        start_date = end_date = None
        period = '5y'
        if period_choice == "Preset":
            period = st.selectbox("Period", ['1y', '3y', '5y', '10y', 'max'], index=2)
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From", value=datetime.now() - timedelta(days=365*5))
            with col2:
                end_date = st.date_input("To", value=datetime.now())

        st.markdown("---")

        # Advanced settings
        if mode == "Advanced":
            st.markdown("**VaR Confidence Level**")
            var_conf = st.select_slider(
                "Confidence",
                options=[0.90, 0.95, 0.99],
                value=0.95,
                format_func=lambda x: f"{int(x*100)}%",
                label_visibility='collapsed'
            )
            st.markdown("---")
        else:
            var_conf = 0.95

        # Analyze button
        analyze_clicked = st.button("📊 Analyze", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.78rem;color:var(--text2);text-align:center;">'
            'Data via Yahoo Finance<br>Up to 30-min delay</div>',
            unsafe_allow_html=True
        )

    return {
        'ticker': ticker_input,
        'benchmark': benchmark,
        'period': period,
        'start_date': start_date,
        'end_date': end_date,
        'mode': mode,
        'var_conf': var_conf,
        'analyze': analyze_clicked,
    }


# ─── Main Analysis Display ─────────────────────────────────────────────────────

def render_analysis(analyzer: SecurityAnalyzer, mode: str, var_conf: float):
    """Render the full analysis dashboard."""
    sym = analyzer.get_currency_symbol()

    # ── Extract company information
    name     = analyzer.info.get('longName') or analyzer.info.get('shortName') or analyzer.ticker
    sector   = analyzer.info.get('sector')   or '—'
    industry = analyzer.info.get('industry') or '—'
    currency = analyzer.get_currency()

    # ── BLOCK 1 — Company Information Cards
    st.markdown(
        f"""<div class="company-cards-row">
          <div class="company-card">
            <div class="company-card-label">Company</div>
            <div class="company-card-value" title="{name}">{name}</div>
          </div>
          <div class="company-card">
            <div class="company-card-label">Sector</div>
            <div class="company-card-value">{sector}</div>
          </div>
          <div class="company-card">
            <div class="company-card-label">Industry</div>
            <div class="company-card-value" title="{industry}">{industry}</div>
          </div>
          <div class="company-card">
            <div class="company-card-label">Currency</div>
            <div class="company-card-value">{currency}</div>
          </div>
        </div>""",
        unsafe_allow_html=True
    )

    # ── Benchmark mismatch warning
    mismatch = analyzer.check_benchmark_mismatch()
    if mismatch:
        st.markdown(
            f'<div class="warn-box">⚠️ <b>Potential benchmark mismatch:</b> {mismatch["exchange"]} — '
            f'suggested benchmark: <b>{mismatch["suggested_name"]} ({mismatch["suggested"]})</b>. '
            f'You selected <b>{mismatch["current"]}</b>.<br><br>'
            f'💡 <i>Why?</i> Securities from different markets often don\'t have overlapping trading days, '
            f'which results in unreliable metrics (NaN values for Beta, correlation, etc.)</div>',
            unsafe_allow_html=True
        )
        if st.button(f"✨ Switch to {mismatch['suggested']} & Re-analyze", type="primary"):
            st.session_state['force_benchmark'] = mismatch['suggested']
            st.session_state['force_ticker'] = analyzer.ticker
            st.session_state['force_analyze'] = True
            st.rerun()

    # ── Benchmark warning (NaN / overlap issues)
    if analyzer.benchmark_warning:
        st.markdown(
            f'<div class="warn-box">⚠️ {analyzer.benchmark_warning}</div>',
            unsafe_allow_html=True
        )

    # ── BLOCK 2 — Price Overview Cards
    sym = analyzer.get_currency_symbol()
    current_price = analyzer.data['Close'].iloc[-1] if analyzer.data is not None and not analyzer.data.empty else None
    high_52 = analyzer.info.get('fiftyTwoWeekHigh') or (analyzer.data['Close'].rolling(252).max().iloc[-1] if analyzer.data is not None else None)
    low_52  = analyzer.info.get('fiftyTwoWeekLow')  or (analyzer.data['Close'].rolling(252).min().iloc[-1] if analyzer.data is not None else None)

    pct_from_high = ((current_price - high_52) / high_52 * 100) if (current_price and high_52) else None
    pct_color = '#ef4444' if (pct_from_high is not None and pct_from_high < 0) else '#22c55e'

    cp_str  = f"{sym}{current_price:,.2f}" if current_price else "N/A"
    h52_str = f"{sym}{high_52:,.2f}"       if high_52       else "N/A"
    l52_str = f"{sym}{low_52:,.2f}"        if low_52        else "N/A"
    pfh_str = f"{pct_from_high:+.2f}%"     if pct_from_high is not None else "N/A"

    st.markdown('<div class="price-section-title"><span class="icon">💰</span> Price Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="price-cards-row">
          <div class="price-card">
            <div class="price-card-label">Current Price</div>
            <div class="price-card-value">{cp_str}</div>
          </div>
          <div class="price-card">
            <div class="price-card-label">52-Week High</div>
            <div class="price-card-value">{h52_str}</div>
          </div>
          <div class="price-card">
            <div class="price-card-label">52-Week Low</div>
            <div class="price-card-value">{l52_str}</div>
          </div>
          <div class="price-card highlight">
            <div class="price-card-label">% From 52W High</div>
            <div class="price-card-value" style="color:{pct_color}">{pfh_str}</div>
          </div>
        </div>""",
        unsafe_allow_html=True
    )

    # ── BLOCK 3 — Price History & Volume Chart
    st.markdown('<div class="section-header">Price History &amp; Volume</div>', unsafe_allow_html=True)
    fig_price = build_price_chart(analyzer)
    st.plotly_chart(fig_price, width="stretch")

    # ── Pre-calculate risk metrics
    beta       = analyzer.calculate_beta()
    ann_ret    = analyzer.calculate_annualized_return()
    volatility = analyzer.calculate_volatility()
    max_dd     = analyzer.calculate_max_drawdown()

    if mode == "Advanced":
        alpha      = analyzer.calculate_alpha()
        sharpe     = analyzer.calculate_sharpe()
        sortino    = analyzer.calculate_sortino()
        info_ratio = analyzer.calculate_information_ratio()
        calmar     = analyzer.calculate_calmar()
        correlation = analyzer.calculate_correlation()
        skewness   = analyzer.calculate_skewness()
        kurtosis   = analyzer.calculate_kurtosis()

    # ── BLOCK 4 — Key Risk Metrics
    st.markdown('<div class="section-header">Key Risk Metrics</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card(
            "Beta",
            f"{beta:.2f}" if not np.isnan(beta) else "N/A",
            "Sensitivity to benchmark movements. Beta=1 moves with market.",
            color='' if np.isnan(beta) else ('red' if beta > 1.5 else 'green' if beta < 0.8 else '')
        ), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card(
            "Annualized Return",
            f"{ann_ret*100:.1f}%" if not np.isnan(ann_ret) else "N/A",
            "Average yearly return over the selected period.",
            color=color_for_value(ann_ret)
        ), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card(
            "Volatility (Ann.)",
            f"{volatility*100:.1f}%" if not np.isnan(volatility) else "N/A",
            "Annualized standard deviation of daily returns.",
            color='red' if not np.isnan(volatility) and volatility > 0.3 else 'green'
        ), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_card(
            "Max Drawdown",
            f"{max_dd*100:.1f}%" if not np.isnan(max_dd) else "N/A",
            "Worst peak-to-trough decline in the period.",
            color='red'
        ), unsafe_allow_html=True)

    # ── BLOCK 5 — Advanced Risk Metrics (Advanced mode only)
    if mode == "Advanced":
        st.markdown('<div class="section-header">Advanced Risk &amp; Performance</div>', unsafe_allow_html=True)

        cols2 = st.columns(4)
        adv_row1 = [
            ("Alpha (Ann.)",      f"{alpha*100:.2f}%"  if not np.isnan(alpha)      else "N/A", "Excess return above CAPM expectation.",                 color_for_value(alpha)),
            ("Sharpe Ratio",      f"{sharpe:.3f}"       if not np.isnan(sharpe)      else "N/A", "Risk-adjusted return (total risk). >1 is good.",        color_for_value(sharpe)),
            ("Sortino Ratio",     f"{sortino:.3f}"      if not np.isnan(sortino)     else "N/A", "Risk-adjusted return (downside risk only). >1 is good.", color_for_value(sortino)),
            ("Information Ratio", f"{info_ratio:.3f}"   if not np.isnan(info_ratio)  else "N/A", "Active return per unit of tracking error.",              color_for_value(info_ratio)),
        ]
        for i, (lbl, val, tip, col) in enumerate(adv_row1):
            with cols2[i]:
                st.markdown(metric_card(lbl, val, tip, col), unsafe_allow_html=True)

        cols3 = st.columns(4)
        adv_row2 = [
            ("Calmar Ratio",    f"{calmar:.3f}"      if not np.isnan(calmar)      else "N/A", "Annualized return / max drawdown.",                    color_for_value(calmar)),
            ("Correlation",     f"{correlation:.3f}" if not np.isnan(correlation) else "N/A", "Correlation with benchmark (–1 to +1).",               ''),
            ("Skewness",        f"{skewness:.3f}"    if not np.isnan(skewness)    else "N/A", "Negative = left tail risk, Positive = right tail.",    color_for_value(skewness)),
            ("Excess Kurtosis", f"{kurtosis:.3f}"    if not np.isnan(kurtosis)    else "N/A", ">0 = fat tails (more extreme events than normal).",   ''),
        ]
        for i, (lbl, val, tip, col) in enumerate(adv_row2):
            with cols3[i]:
                st.markdown(metric_card(lbl, val, tip, col), unsafe_allow_html=True)

        # ── BLOCK 6 — Value at Risk (single f-string, no concatenation, avoids Streamlit markdown parse bug)
        st.markdown('<div class="section-header">Value at Risk Analysis</div>', unsafe_allow_html=True)

        v90 = analyzer.calculate_var(0.90)
        v95 = analyzer.calculate_var(0.95)
        v99 = analyzer.calculate_var(0.99)

        def _fmt_var(v):
            return f"{v*100:.3f}%" if not np.isnan(v) else "N/A"

        var_html = (
            '<div class="var-wrapper">'
            '<table class="var-table">'
            '<thead><tr>'
            '<th>Confidence</th>'
            '<th>Historical VaR</th>'
            '<th>Parametric VaR</th>'
            '<th>CVaR (Expected Shortfall)</th>'
            '</tr></thead>'
            '<tbody>'
            f'<tr><td><strong>90%</strong></td>'
            f'<td style="color:#ef4444;font-weight:700">{_fmt_var(v90["hist"])}</td>'
            f'<td style="color:#f59e0b;font-weight:700">{_fmt_var(v90["param"])}</td>'
            f'<td style="color:#dc2626;font-weight:700">{_fmt_var(v90["cvar"])}</td></tr>'
            f'<tr><td><strong>95%</strong></td>'
            f'<td style="color:#ef4444;font-weight:700">{_fmt_var(v95["hist"])}</td>'
            f'<td style="color:#f59e0b;font-weight:700">{_fmt_var(v95["param"])}</td>'
            f'<td style="color:#dc2626;font-weight:700">{_fmt_var(v95["cvar"])}</td></tr>'
            f'<tr><td><strong>99%</strong></td>'
            f'<td style="color:#ef4444;font-weight:700">{_fmt_var(v99["hist"])}</td>'
            f'<td style="color:#f59e0b;font-weight:700">{_fmt_var(v99["param"])}</td>'
            f'<td style="color:#dc2626;font-weight:700">{_fmt_var(v99["cvar"])}</td></tr>'
            '</tbody></table></div>'
        )
        st.markdown(var_html, unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.75rem;color:var(--text2);margin-top:0.4rem;">'
            'VaR = maximum expected daily loss at the given confidence level. '
            'CVaR (Expected Shortfall) = average loss when VaR is exceeded.</p>',
            unsafe_allow_html=True
        )

    # ── BLOCK 7 — Fundamental Metrics
    st.markdown('<div class="section-header">Fundamental Metrics</div>', unsafe_allow_html=True)
    fundamentals = analyzer.get_fundamentals(mode='basic' if mode == 'Basic' else 'advanced')
    items = list(fundamentals.items())
    mid = len(items) // 2 if mode == 'Advanced' else 2
    col_a, col_b = st.columns(2)

    def fund_table_html(pairs):
        rows = ''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in pairs)
        return f'<table class="fund-table">{rows}</table>'

    with col_a:
        st.markdown(fund_table_html(items[:mid]), unsafe_allow_html=True)
    with col_b:
        st.markdown(fund_table_html(items[mid:]), unsafe_allow_html=True)

    # ── BLOCK 8 — Cumulative Returns Chart
    bench_name = BENCHMARK_NAMES.get(analyzer.benchmark, analyzer.benchmark)
    st.markdown('<div class="section-header">Cumulative Returns</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:0.8rem;color:var(--text2);margin:-0.5rem 0 0.5rem;">Benchmark: {bench_name}</p>',
        unsafe_allow_html=True
    )
    fig_returns = build_returns_chart(analyzer)
    st.plotly_chart(fig_returns, width="stretch")

    # ── BLOCK 9 — Drawdown & Distribution (Advanced only)
    if mode == "Advanced":
        col_dd, col_dist = st.columns(2)
        with col_dd:
            st.markdown('<div class="section-header">Drawdown from Peak</div>', unsafe_allow_html=True)
            fig_dd = build_drawdown_chart(analyzer)
            st.plotly_chart(fig_dd, width="stretch")
        with col_dist:
            st.markdown('<div class="section-header">Daily Returns Distribution</div>', unsafe_allow_html=True)
            fig_dist = build_distribution_chart(analyzer)
            st.plotly_chart(fig_dist, width="stretch")

        # ── BLOCK 10 — Performance Summary Table
        st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
        period_data = {
            '1 Month': 21, '3 Months': 63, '6 Months': 126,
            '1 Year': 252, '3 Years': 756, '5 Years': 1260
        }
        perf_rows = []
        for label, days in period_data.items():
            if len(analyzer.returns) < days:
                continue
            sub       = analyzer.returns.iloc[-days:]
            sub_bench = analyzer.benchmark_returns.iloc[-days:] if len(analyzer.benchmark_returns) >= days else None
            ret    = (1 + sub).prod() - 1
            vol    = sub.std() * np.sqrt(252)
            b_ret  = (1 + sub_bench).prod() - 1 if sub_bench is not None else np.nan
            excess = ret - b_ret if not np.isnan(b_ret) else np.nan
            perf_rows.append({
                'Period':         label,
                'Return':         f"{ret*100:.1f}%",
                f'{bench_name}':  f"{b_ret*100:.1f}%" if not np.isnan(b_ret) else 'N/A',
                'Excess Return':  f"{excess*100:.1f}%" if not np.isnan(excess) else 'N/A',
                'Volatility':     f"{vol*100:.1f}%",
            })

        if perf_rows:
            headers = list(perf_rows[0].keys())
            th_cells = ''.join(f'<th>{h}</th>' for h in headers)
            td_rows = ''
            for row in perf_rows:
                td_cells = ''
                for i, (k, v) in enumerate(row.items()):
                    clr = ''
                    if i >= 1:
                        try:
                            pct = float(v.replace('%', ''))
                            clr = 'color:#22c55e;font-weight:700' if pct > 0 else 'color:#ef4444;font-weight:700'
                        except Exception:
                            pass
                    td_cells += f'<td style="{clr}">{v}</td>'
                td_rows += f'<tr>{td_cells}</tr>'
            perf_html = (
                f'<div class="perf-wrapper"><table class="perf-table">'
                f'<thead><tr>{th_cells}</tr></thead>'
                f'<tbody>{td_rows}</tbody>'
                f'</table></div>'
            )
            st.markdown(perf_html, unsafe_allow_html=True)


# ─── Error Display ─────────────────────────────────────────────────────────────

def render_ticker_error(ticker: str):
    """Render helpful error message for invalid/not found ticker."""
    st.markdown(f"""
    <div style="padding:1.5rem;border:1px solid var(--border);border-radius:4px;margin:1rem 0;">
      <h3 style="color:var(--accent);margin:0 0 0.8rem;">❌ Could not find ticker '{ticker}'</h3>
      <p style="color:var(--text2);">Please check:</p>
      <ul style="color:var(--text2);margin-left:1.2rem;">
        <li>Ticker symbol is correct (e.g., AAPL, MSFT, GOOGL)</li>
        <li>For Indian stocks, add .NS or .BO (e.g., RELIANCE.NS, TCS.BO)</li>
        <li>For international stocks, check the exchange suffix</li>
      </ul>
      <p style="color:var(--text2);margin-top:0.8rem;font-style:italic;">Common formats:</p>
      <ul style="color:var(--text2);margin-left:1.2rem;">
        <li>US stocks: AAPL, TSLA, NVDA</li>
        <li>Indian stocks: RELIANCE.NS, TCS.BO, INFY.NS</li>
        <li>UK stocks: HSBA.L, BP.L</li>
        <li>Japanese stocks: 7203.T, 9984.T</li>
        <li>Hong Kong stocks: 0700.HK, 9988.HK</li>
        <li>Canadian stocks: SHOP.TO, RY.TO</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""

    # ── Initialize session state
    for key, default in [
        ('mode', 'Basic'), ('analyzer', None), ('last_ticker', 'AAPL'),
        ('last_benchmark', 'SPY'), ('force_analyze', False),
        ('force_benchmark', None), ('force_ticker', None)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Header
    st.markdown("""
    <div class="dashboard-header">
      <h1>Security Performance &amp; Risk Analysis</h1>
      <div class="subtitle">Institutional-Grade Equity Analysis &nbsp;·&nbsp; Global Markets</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Render sidebar and get config
    config = render_sidebar()
    ticker_raw = config['ticker']
    benchmark = config['benchmark']
    mode = config['mode']
    period = config['period']
    start_date = config['start_date']
    end_date = config['end_date']
    var_conf = config['var_conf']
    analyze = config['analyze']

    # ── Handle forced re-analysis (from mismatch switch button)
    if st.session_state.get('force_analyze'):
        benchmark = st.session_state['force_benchmark']
        ticker_raw = st.session_state['force_ticker']
        st.session_state['force_analyze'] = False
        analyze = True

    # ── Resolve ticker
    ticker = resolve_ticker(ticker_raw)

    # ── Check if we should analyze
    if analyze and ticker:
        with st.spinner(f"Fetching data for {ticker}…"):
            analyzer = SecurityAnalyzer(ticker, benchmark)
            success = analyzer.fetch_data(
                period=period,
                start_date=str(start_date) if start_date else None,
                end_date=str(end_date) if end_date else None
            )

        if success:
            st.session_state['analyzer'] = analyzer
            st.session_state['last_ticker'] = ticker
            st.session_state['last_benchmark'] = benchmark
        else:
            render_ticker_error(ticker)
            return

    elif analyze and not ticker:
        st.warning("Please enter a valid ticker symbol.")
        return

    # ── Render analysis if we have data
    analyzer = st.session_state.get('analyzer')

    if analyzer is None:
        # Welcome state
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:var(--text2);">
          <div style="font-size:3rem;margin-bottom:1rem;">📈</div>
          <h2 style="font-weight:400;font-style:italic;color:var(--text2);">
            Enter a ticker symbol in the sidebar to begin your analysis.
          </h2>
          <p style="margin-top:1rem;font-size:0.95rem;">
            Supports 30+ currencies · Global exchanges · Dual-mode interface
          </p>
          <div style="margin-top:2rem;display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;font-size:0.9rem;">
            <span>🇺🇸 US: AAPL, MSFT, NVDA</span>
            <span>🇬🇧 UK: HSBA.L, BP.L</span>
            <span>🇮🇳 India: RELIANCE.NS, TCS.BO</span>
            <span>🇯🇵 Japan: 7203.T, 9984.T</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Mode badge
        mode_label = mode.upper()
        mode_cls = 'mode-basic' if mode == 'Basic' else 'mode-advanced'
        st.markdown(
            f'<span class="mode-badge {mode_cls}">{mode_label} MODE</span>',
            unsafe_allow_html=True
        )
        render_analysis(analyzer, mode, var_conf)

    # ── Fixed disclaimer
    st.markdown("""
    <div class="disclaimer">
      ⚠️ DISCLAIMER: This tool has been developed as an educational platform and not for financial advice.
      Data displayed on this dashboard may be delayed by up to 30 minutes.
      Please consult a registered financial advisor before making any financial decisions.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
