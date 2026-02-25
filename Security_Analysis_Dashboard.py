"""
Security Performance and Risk Analysis Dashboard
A comprehensive tool for equity analysis and risk metrics calculation
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Security Performance & Risk Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS (responsive and adaptive to browser theme)
st.markdown("""
    <style>
    /* ===== FONT FAMILY ===== */
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* ===== LIGHT/DARK MODE ADAPTIVE COLORS ===== */
    /* Light mode (browser default) */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-metric: #e8eef3;
        --bg-sidebar: #f0f2f6;
        --text-primary: #1f1f1f;
        --text-secondary: #4a4a4a;
        --border-color: #d0d0d0;
        --accent-blue: #1f77b4;
    }
    
    /* Dark mode (browser prefers dark) */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0e1117;
            --bg-secondary: #1e1e1e;
            --bg-metric: #2c3e50;
            --bg-sidebar: #1e1e1e;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --border-color: #4a4a4a;
            --accent-blue: #1f77b4;
        }
    }
    
    /* ===== MOBILE RESPONSIVENESS ===== */
    /* Base mobile styles */
    @media only screen and (max-width: 768px) {
        .main {
            padding: 0.5rem !important;
        }
        
        /* Stack columns on mobile */
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        
        /* Full width metrics on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
        }
        
        /* Adjust disclaimer for mobile */
        .disclaimer {
            position: relative !important;
            left: 0 !important;
            font-size: 12px !important;
            padding: 10px !important;
            margin-top: 2rem !important;
        }
        
        /* Sidebar adjustments for mobile */
        [data-testid="stSidebar"] {
            width: 100% !important;
        }
        
        /* Smaller fonts on mobile */
        h1 {
            font-size: 1.5rem !important;
        }
        
        h2 {
            font-size: 1.2rem !important;
        }
        
        h3 {
            font-size: 1rem !important;
        }
        
        /* Chart height adjustments */
        .js-plotly-plot {
            height: 300px !important;
        }
    }
    
    /* Tablet styles */
    @media only screen and (min-width: 769px) and (max-width: 1024px) {
        .main {
            padding: 0.75rem !important;
        }
        
        .disclaimer {
            left: 16rem !important;
        }
    }
    
    /* Desktop styles */
    @media only screen and (min-width: 1025px) {
        .main {
            padding: 0rem 1rem;
        }
        
        .disclaimer {
            left: 21rem !important;
        }
    }
    
    /* ===== CORE LAYOUT ===== */
    .main {
        background-color: var(--bg-primary);
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
    }
    
    [data-testid="stSidebar"] > div {
        background-color: var(--bg-sidebar);
    }
    
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }
    
    /* ===== METRICS CARDS ===== */
    .stMetric {
        background-color: var(--bg-metric) !important;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid var(--accent-blue);
        margin-bottom: 10px;
    }
    
    .stMetric label, 
    .stMetric div,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: var(--text-primary) !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1 {
        color: var(--text-primary);
        font-weight: 600;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border-color);
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    h2, h3 {
        color: var(--text-primary);
        font-weight: 500;
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    p, label, span, div {
        color: var(--text-primary) !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* ===== ALERT MESSAGES ===== */
    .stAlert,
    [data-testid="stAlert"] {
        background-color: var(--bg-metric) !important;
        border-left: 4px solid var(--accent-blue);
        color: var(--text-primary) !important;
    }
    
    .stSuccess,
    [data-testid="stAlert"][data-baseweb="notification"] {
        background-color: var(--bg-metric) !important;
        color: var(--text-primary) !important;
    }
    
    .stInfo {
        background-color: var(--bg-metric) !important;
        color: var(--text-primary) !important;
        border-left: 4px solid #17a2b8 !important;
    }
    
    .stWarning {
        background-color: var(--bg-metric) !important;
        color: var(--text-primary) !important;
        border-left: 4px solid #ffc107 !important;
    }
    
    .stError {
        background-color: var(--bg-metric) !important;
        color: var(--text-primary) !important;
        border-left: 4px solid #dc3545 !important;
    }
    
    [data-testid="stAlert"] p, 
    [data-testid="stAlert"] div {
        color: var(--text-primary) !important;
    }
    
    /* ===== INPUTS ===== */
    input, select, textarea {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background-color: var(--bg-metric);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }
    
    .stButton > button:hover {
        background-color: var(--accent-blue);
        color: #ffffff;
        border-color: var(--accent-blue);
    }
    
    /* ===== DATAFRAMES ===== */
    .dataframe {
        color: var(--text-primary) !important;
        background-color: var(--bg-secondary) !important;
    }
    
    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ===== DISCLAIMER ===== */
    .disclaimer {
        position: fixed;
        bottom: 0;
        right: 0;
        background-color: #ff0000;
        color: #ffffff;
        padding: 15px;
        text-align: center;
        font-weight: bold;
        font-size: 14px;
        z-index: 999;
        border-top: 3px solid #ffffff;
        font-family: 'Times New Roman', Times, serif !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
    }
    
    /* Ensure disclaimer text is always white */
    .disclaimer, .disclaimer * {
        color: #ffffff !important;
    }
    
    /* ===== MOBILE SPECIFIC OPTIMIZATIONS ===== */
    @media only screen and (max-width: 768px) {
        /* Touch-friendly buttons */
        .stButton > button {
            min-height: 44px;
            font-size: 16px;
            padding: 12px 24px;
        }
        
        /* Larger input fields for mobile */
        input, select {
            font-size: 16px !important;
            min-height: 44px;
        }
        
        /* Prevent zoom on input focus (iOS) */
        input:focus, select:focus, textarea:focus {
            font-size: 16px !important;
        }
        
        /* Scrollable tables on mobile */
        .dataframe-container {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
    }
    
    /* ===== ACCESSIBILITY ===== */
    /* High contrast for better readability */
    @media (prefers-contrast: high) {
        :root {
            --border-color: #000000;
        }
        
        .stMetric {
            border-left-width: 5px;
        }
    }
    
    /* Reduced motion for users who prefer it */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Add disclaimer at the bottom
st.markdown("""
    <div class="disclaimer">
        ⚠️ DISCLAIMER: This tool has been developed as an educational platform and not for financial advice. 
        Data displayed on this dashboard may be delayed by up to 30 minutes. 
        Please consult a registered financial advisor before making any financial decisions.
    </div>
    """, unsafe_allow_html=True)


class SecurityAnalyzer:
    """Main class for security analysis and risk calculations"""
    
    def __init__(self, ticker, benchmark="SPY"):
        self.ticker = ticker.upper()
        self.benchmark = benchmark
        self.data = None
        self.benchmark_data = None
        self.info = None
        self.suggested_benchmark = None
        self.benchmark_warning = None
    
    def get_appropriate_benchmark(self):
        """
        Automatically suggest the appropriate benchmark based on ticker exchange.
        Returns (suggested_benchmark, warning_message)
        """
        ticker = self.ticker.upper()
        
        # Exchange-specific benchmark mappings
        exchange_benchmarks = {
            '.NS': ('^NSEI', 'NSE India - suggested benchmark: Nifty 50 (^NSEI)'),
            '.BO': ('^BSESN', 'BSE India - suggested benchmark: BSE Sensex (^BSESN)'),
            '.L': ('^FTSE', 'London Stock Exchange - suggested benchmark: FTSE 100 (^FTSE)'),
            '.T': ('^N225', 'Tokyo Stock Exchange - suggested benchmark: Nikkei 225 (^N225)'),
            '.HK': ('^HSI', 'Hong Kong Stock Exchange - suggested benchmark: Hang Seng (^HSI)'),
            '.AX': ('^AXJO', 'Australian Stock Exchange - suggested benchmark: ASX 200 (^AXJO)'),
            '.TO': ('^GSPTSE', 'Toronto Stock Exchange - suggested benchmark: S&P/TSX (^GSPTSE)'),
            '.PA': ('^FCHI', 'Paris Stock Exchange - suggested benchmark: CAC 40 (^FCHI)'),
            '.DE': ('^GDAXI', 'Frankfurt Stock Exchange - suggested benchmark: DAX (^GDAXI)'),
            '.SW': ('^SSMI', 'Swiss Exchange - suggested benchmark: SMI (^SSMI)'),
        }
        
        # Check if ticker has an exchange suffix
        for suffix, (benchmark, message) in exchange_benchmarks.items():
            if ticker.endswith(suffix):
                return benchmark, message
        
        # Default to SPY for US stocks (no suffix or common US patterns)
        if ticker.startswith('^'):
            # It's an index itself
            return 'SPY', 'Index ticker - using S&P 500 (SPY) as benchmark'
        
        # Check if it's likely a US stock (no suffix, 1-5 letters)
        if '.' not in ticker and len(ticker) <= 5:
            return 'SPY', 'US Stock - using S&P 500 (SPY) as benchmark'
        
        # Unknown exchange
        return None, 'Unknown exchange - please select appropriate benchmark manually'
    
    def check_benchmark_compatibility(self):
        """
        Check if the selected benchmark is appropriate for the ticker.
        Returns (is_compatible, warning_message)
        """
        ticker = self.ticker.upper()
        benchmark = self.benchmark.upper()
        
        # Get suggested benchmark
        suggested, message = self.get_appropriate_benchmark()
        
        # If user's benchmark doesn't match suggested, create warning
        if suggested and benchmark != suggested:
            return False, f"⚠️ Potential benchmark mismatch: {message}. You selected {benchmark}."
        
        return True, None
        
    def search_ticker(self, query):
        """
        Search for ticker symbol from company name or ticker.
        Simplified to avoid rate limiting - minimal validation.
        """
        try:
            query = query.strip().upper()
            
            # If it has a dot or caret, assume it's already properly formatted
            if '.' in query or '^' in query:
                return query, query
            
            # If no suffix and not too long, try adding .NS for Indian stocks
            if len(query) <= 10 and ' ' not in query:
                # First, try as-is (might be US stock)
                return query, query
            
            # If it looks like a company name (has space or very long), flag it
            if ' ' in query or len(query) > 10:
                return None, "name_detected"
            
            # Default: return as-is
            return query, query
            
        except Exception as e:
            # On any error, just return the query as-is
            return query.upper(), query.upper()
    
    def fetch_data(self, period="5y", start_date=None, end_date=None):
        """Fetch historical price data with retry logic"""
        import time
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Add small delay between retries to avoid rate limiting
                if attempt > 0:
                    time.sleep(retry_delay * attempt)
                
                ticker_obj = yf.Ticker(self.ticker)
                
                # Fetch security data
                if start_date and end_date:
                    self.data = ticker_obj.history(start=start_date, end=end_date)
                else:
                    self.data = ticker_obj.history(period=period)
                
                # Check if we got any data
                if self.data is None or len(self.data) == 0:
                    if attempt < max_retries - 1:
                        continue  # Retry
                    return False
                
                # Only fetch info if data fetch succeeded (to avoid rate limits)
                try:
                    self.info = ticker_obj.info
                except:
                    self.info = {}  # Continue without info
                
                # Fetch benchmark data
                benchmark_obj = yf.Ticker(self.benchmark)
                if start_date and end_date:
                    self.benchmark_data = benchmark_obj.history(start=start_date, end=end_date)
                else:
                    self.benchmark_data = benchmark_obj.history(period=period)
                
                # Success if we got here
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if 'rate' in error_msg or '429' in error_msg or 'many requests' in error_msg:
                    if attempt < max_retries - 1:
                        st.warning(f"⏳ Rate limited by Yahoo Finance. Retrying in {retry_delay * (attempt + 1)} seconds... (Attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        st.error("⚠️ Yahoo Finance rate limit reached. Please wait a minute and try again.")
                        return False
                
                # Other errors
                if attempt < max_retries - 1:
                    continue
                else:
                    st.error(f"Error fetching data: {str(e)}")
                    return False
        
        return False
    
    def calculate_moving_averages(self):
        """Calculate 50, 100, and 200-day moving averages"""
        self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA100'] = self.data['Close'].rolling(window=100).mean()
        self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
    
    def calculate_returns(self):
        """Calculate daily and cumulative returns"""
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Cumulative_Return'] = (1 + self.data['Daily_Return']).cumprod() - 1
        
        self.benchmark_data['Daily_Return'] = self.benchmark_data['Close'].pct_change()
        self.benchmark_data['Cumulative_Return'] = (1 + self.benchmark_data['Daily_Return']).cumprod() - 1
    
    def get_52week_metrics(self):
        """Get 52-week high and low"""
        last_year = self.data.tail(252)
        return {
            '52W_High': last_year['High'].max(),
            '52W_Low': last_year['Low'].min(),
            'Current_Price': self.data['Close'].iloc[-1]
        }
    
    def calculate_volatility(self, window=252):
        """Calculate annualized volatility"""
        returns = self.data['Daily_Return'].dropna()
        volatility = returns.std() * np.sqrt(window)
        return volatility
    
    def calculate_beta(self):
        """Calculate Beta relative to benchmark"""
        # Align dates
        merged = pd.merge(
            self.data[['Daily_Return']],
            self.benchmark_data[['Daily_Return']],
            left_index=True,
            right_index=True,
            suffixes=('_stock', '_benchmark')
        ).dropna()
        
        # Check if there's sufficient overlapping data
        if len(merged) < 30:
            self.benchmark_warning = f"⚠️ WARNING: Only {len(merged)} overlapping trading days between {self.ticker} and {self.benchmark}. Beta calculation may be unreliable. Consider using a benchmark from the same market."
        
        # Check if there's NO overlapping data
        if len(merged) == 0:
            self.benchmark_warning = f"🚨 ERROR: No overlapping trading days between {self.ticker} and {self.benchmark}. These securities likely trade on different exchanges/timezones. Please select an appropriate benchmark for this market."
            return np.nan
        
        covariance = merged['Daily_Return_stock'].cov(merged['Daily_Return_benchmark'])
        benchmark_variance = merged['Daily_Return_benchmark'].var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        
        # Check if beta is NaN
        if np.isnan(beta):
            if self.benchmark_warning is None:
                self.benchmark_warning = f"⚠️ Beta calculation resulted in NaN. This typically means {self.ticker} and {self.benchmark} don't have overlapping data or trade on different exchanges."
        
        return beta
    
    def calculate_alpha(self, risk_free_rate=0.04):
        """Calculate Jensen's Alpha"""
        beta = self.calculate_beta()
        
        # Annualized returns
        stock_return = self.data['Daily_Return'].mean() * 252
        benchmark_return = self.benchmark_data['Daily_Return'].mean() * 252
        
        # Alpha = Stock Return - (Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate))
        alpha = stock_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
        
        return alpha
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.04):
        """Calculate Sharpe Ratio"""
        returns = self.data['Daily_Return'].dropna()
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = self.calculate_volatility()
        
        sharpe = excess_returns / volatility if volatility != 0 else 0
        return sharpe
    
    def calculate_sortino_ratio(self, risk_free_rate=0.04):
        """Calculate Sortino Ratio (downside deviation)"""
        returns = self.data['Daily_Return'].dropna()
        excess_returns = returns.mean() * 252 - risk_free_rate
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        
        sortino = excess_returns / downside_std if downside_std != 0 else 0
        return sortino
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative = (1 + self.data['Daily_Return'].fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return max_dd
    
    def calculate_var(self, confidence_level=0.95):
        """Calculate Value at Risk (VaR)"""
        returns = self.data['Daily_Return'].dropna()
        
        # Historical VaR
        var_historical = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR
        mean = returns.mean()
        std = returns.std()
        var_parametric = stats.norm.ppf(1 - confidence_level, mean, std)
        
        return {
            'Historical_VaR': var_historical,
            'Parametric_VaR': var_parametric
        }
    
    def calculate_cvar(self, confidence_level=0.95):
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)"""
        returns = self.data['Daily_Return'].dropna()
        var = self.calculate_var(confidence_level)['Historical_VaR']
        cvar = returns[returns <= var].mean()
        return cvar
    
    def calculate_correlation(self):
        """Calculate correlation with benchmark"""
        merged = pd.merge(
            self.data[['Daily_Return']],
            self.benchmark_data[['Daily_Return']],
            left_index=True,
            right_index=True,
            suffixes=('_stock', '_benchmark')
        ).dropna()
        
        correlation = merged['Daily_Return_stock'].corr(merged['Daily_Return_benchmark'])
        return correlation
    
    def calculate_information_ratio(self):
        """Calculate Information Ratio"""
        merged = pd.merge(
            self.data[['Daily_Return']],
            self.benchmark_data[['Daily_Return']],
            left_index=True,
            right_index=True,
            suffixes=('_stock', '_benchmark')
        ).dropna()
        
        active_returns = merged['Daily_Return_stock'] - merged['Daily_Return_benchmark']
        tracking_error = active_returns.std() * np.sqrt(252)
        
        info_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0
        return info_ratio
    
    def calculate_calmar_ratio(self):
        """Calculate Calmar Ratio"""
        annual_return = self.data['Daily_Return'].mean() * 252
        max_dd = abs(self.calculate_max_drawdown())
        calmar = annual_return / max_dd if max_dd != 0 else 0
        return calmar
    
    def get_fundamental_metrics(self):
        """Extract fundamental metrics from yfinance"""
        if not self.info:
            return {}
        
        metrics = {
            'Market_Cap': self.info.get('marketCap', 'N/A'),
            'PE_Ratio': self.info.get('trailingPE', 'N/A'),
            'Forward_PE': self.info.get('forwardPE', 'N/A'),
            'PEG_Ratio': self.info.get('pegRatio', 'N/A'),
            'Price_to_Book': self.info.get('priceToBook', 'N/A'),
            'Dividend_Yield': self.info.get('dividendYield', 'N/A'),
            'Profit_Margin': self.info.get('profitMargins', 'N/A'),
            'ROE': self.info.get('returnOnEquity', 'N/A'),
            'ROA': self.info.get('returnOnAssets', 'N/A'),
            'Debt_to_Equity': self.info.get('debtToEquity', 'N/A'),
            'Current_Ratio': self.info.get('currentRatio', 'N/A'),
            'Revenue_Growth': self.info.get('revenueGrowth', 'N/A'),
            'Earnings_Growth': self.info.get('earningsGrowth', 'N/A'),
        }
        
        return metrics
    
    def get_currency_symbol(self):
        """Get currency symbol based on stock's country/exchange"""
        currency = None
        
        # Try to get currency from info first
        if self.info and isinstance(self.info, dict) and len(self.info) > 0:
            currency = self.info.get('currency')
        
        # If no currency in info, infer from ticker suffix
        if not currency:
            ticker_upper = self.ticker.upper()
            if '.L' in ticker_upper:
                currency = 'GBp'  # UK stocks usually in pence
            elif '.NS' in ticker_upper or '.BO' in ticker_upper:
                currency = 'INR'  # Indian stocks
            elif '.T' in ticker_upper:
                currency = 'JPY'  # Japanese stocks
            elif '.HK' in ticker_upper:
                currency = 'HKD'  # Hong Kong stocks
            elif '.AX' in ticker_upper:
                currency = 'AUD'  # Australian stocks
            elif '.TO' in ticker_upper:
                currency = 'CAD'  # Canadian stocks
            elif '.PA' in ticker_upper:
                currency = 'EUR'  # Paris stocks
            elif '.DE' in ticker_upper or '.SW' in ticker_upper:
                currency = 'EUR'  # German/Swiss stocks (often EUR)
            else:
                currency = 'USD'  # Default to USD for US stocks
        
        # Handle GBp (pence) - special case for UK stocks
        if currency == 'GBp':
            return 'p'  # Pence symbol
        
        # Currency symbol mapping
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'GBp': 'p',  # Pence
            'JPY': '¥',
            'CNY': '¥',
            'INR': '₹',
            'AUD': 'A$',
            'CAD': 'C$',
            'CHF': 'CHF ',
            'HKD': 'HK$',
            'SGD': 'S$',
            'KRW': '₩',
            'BRL': 'R$',
            'RUB': '₽',
            'ZAR': 'R',
            'MXN': 'MX$',
            'SEK': 'kr',
            'NOK': 'kr',
            'DKK': 'kr',
            'PLN': 'zł',
            'TRY': '₺',
            'THB': '฿',
            'IDR': 'Rp',
            'MYR': 'RM',
            'PHP': '₱',
            'NZD': 'NZ$',
            'ILS': '₪',
            'AED': 'AED ',
            'SAR': 'SAR ',
        }
        
        return currency_symbols.get(currency, currency + ' ' if currency else '$')
    
    def get_currency_name(self):
        """Get full currency name"""
        currency = None
        
        # Try to get from info first
        if self.info and isinstance(self.info, dict) and len(self.info) > 0:
            currency = self.info.get('currency')
        
        # If no currency in info, infer from ticker suffix
        if not currency:
            ticker_upper = self.ticker.upper()
            if '.L' in ticker_upper:
                currency = 'GBp'
            elif '.NS' in ticker_upper or '.BO' in ticker_upper:
                currency = 'INR'
            elif '.T' in ticker_upper:
                currency = 'JPY'
            elif '.HK' in ticker_upper:
                currency = 'HKD'
            elif '.AX' in ticker_upper:
                currency = 'AUD'
            elif '.TO' in ticker_upper:
                currency = 'CAD'
            elif '.PA' in ticker_upper:
                currency = 'EUR'
            elif '.DE' in ticker_upper or '.SW' in ticker_upper:
                currency = 'EUR'
            else:
                currency = 'USD'
        
        # Handle pence specially
        if currency == 'GBp':
            return 'GBp (Pence)'
        
        return currency if currency else 'USD'


def format_number(num, is_percentage=False, is_currency=False, currency_symbol='$'):
    """Format numbers for display with dynamic currency support"""
    if isinstance(num, str) or num == 'N/A':
        return num
    
    if is_percentage:
        return f"{num * 100:.2f}%"
    elif is_currency:
        if num >= 1e12:
            return f"{currency_symbol}{num/1e12:.2f}T"
        elif num >= 1e9:
            return f"{currency_symbol}{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{currency_symbol}{num/1e6:.2f}M"
        else:
            return f"{currency_symbol}{num:,.2f}"
    else:
        return f"{num:.4f}"


def create_price_chart(analyzer):
    """Create interactive price chart with moving averages"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{analyzer.ticker} Price & Moving Averages', 'Volume')
    )
    
    # Price and MAs
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['Close'],
                  name='Close', line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['MA50'],
                  name='MA50', line=dict(color='#ff7f0e', width=1.5, dash='dot')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['MA100'],
                  name='MA100', line=dict(color='#2ca02c', width=1.5, dash='dash')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['MA200'],
                  name='MA200', line=dict(color='#d62728', width=1.5, dash='dashdot')),
        row=1, col=1
    )
    
    # Volume
    colors = ['red' if row['Close'] < row['Open'] else 'green' 
              for _, row in analyzer.data.iterrows()]
    
    fig.add_trace(
        go.Bar(x=analyzer.data.index, y=analyzer.data['Volume'],
              name='Volume', marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def create_returns_chart(analyzer):
    """Create cumulative returns comparison chart"""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=analyzer.data['Cumulative_Return'] * 100,
                  name=analyzer.ticker, line=dict(color='#1f77b4', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=analyzer.benchmark_data.index, 
                  y=analyzer.benchmark_data['Cumulative_Return'] * 100,
                  name=analyzer.benchmark, line=dict(color='#ff7f0e', width=2))
    )
    
    fig.update_layout(
        title=f'Cumulative Returns: {analyzer.ticker} vs {analyzer.benchmark}',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_drawdown_chart(analyzer):
    """Create drawdown chart"""
    cumulative = (1 + analyzer.data['Daily_Return'].fillna(0)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=analyzer.data.index, y=drawdown,
                  fill='tozeroy', name='Drawdown',
                  line=dict(color='#d62728', width=1))
    )
    
    fig.update_layout(
        title='Drawdown Analysis',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_returns_distribution(analyzer):
    """Create returns distribution histogram"""
    returns = analyzer.data['Daily_Return'].dropna() * 100
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(x=returns, nbinsx=50, name='Daily Returns',
                    marker_color='#1f77b4', opacity=0.7)
    )
    
    # Add normal distribution overlay
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    y = stats.norm.pdf(x, mu, sigma) * len(returns) * (returns.max() - returns.min()) / 50
    
    fig.add_trace(
        go.Scatter(x=x, y=y, name='Normal Distribution',
                  line=dict(color='#ff7f0e', width=2))
    )
    
    fig.update_layout(
        title='Daily Returns Distribution',
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def main():
    """Main application"""
    
    # Initialize session state
    if 'analyzed_data' not in st.session_state:
        st.session_state.analyzed_data = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Basic"
    
    # Header
    st.title("📊 Security Performance & Risk Analysis")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Analysis Mode Toggle
        st.session_state.analysis_mode = st.radio(
            "📊 Analysis Mode",
            ["Basic", "Advanced"],
            horizontal=True,
            help="Basic: Essential metrics only | Advanced: Full detailed analysis"
        )
        
        st.markdown("---")
        
        # Security input
        security_input = st.text_input(
            "Ticker Symbol",
            placeholder="e.g., AAPL, MSFT, RELIANCE.NS, TCS.BO",
            help="Enter TICKER SYMBOL (e.g., AAPL, MSFT, RELIANCE.NS, TCS.BO). Use Yahoo Finance format.",
            key="security_input"
        )
        
        # Add helpful hint below input
        st.caption("💡 Tip: Use ticker symbols. For Indian stocks add .NS (NSE) or .BO (BSE)")
        
        # Benchmark
        benchmark = st.selectbox(
            "Benchmark Index",
            ["SPY", "QQQ", "DIA", "IWM", "^GSPC", "^DJI", "^IXIC", "^NSEI", "^BSESN", "^FTSE", "^N225", "^HSI", "^AXJO", "^GSPTSE", "^FCHI", "^GDAXI", "^SSMI"],
            help="Select benchmark for comparative analysis\nUS: SPY/QQQ/DIA | India: ^NSEI/^BSESN | UK: ^FTSE | Japan: ^N225 | HK: ^HSI | AU: ^AXJO | CA: ^GSPTSE | FR: ^FCHI | DE: ^GDAXI | CH: ^SSMI",
            key="benchmark"
        )
        
        # Time period
        st.subheader("Analysis Period")
        period_type = st.radio(
            "Period Type",
            ["Preset", "Custom"],
            horizontal=True,
            key="period_type"
        )
        
        if period_type == "Preset":
            period = st.selectbox(
                "Select Period",
                ["1Y", "3Y", "5Y", "10Y", "Max"],
                index=2,
                key="period"
            )
            period_map = {
                "1Y": "1y",
                "3Y": "3y",
                "5Y": "5y",
                "10Y": "10y",
                "Max": "max"
            }
            selected_period = period_map[period]
            start_date = None
            end_date = None
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365*5),
                    key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    key="end_date"
                )
            selected_period = None
        
        # Advanced options (only show in Advanced mode)
        if st.session_state.analysis_mode == "Advanced":
            st.markdown("---")
            st.subheader("Advanced Settings")
            
            # Risk-free rate
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.1,
                help="Used for Sharpe and Sortino ratio calculations",
                key="risk_free_rate"
            ) / 100
            
            # VaR confidence level
            var_confidence = st.slider(
                "VaR Confidence Level (%)",
                min_value=90,
                max_value=99,
                value=95,
                step=1,
                key="var_confidence"
            ) / 100
        else:
            # Use defaults for Basic mode
            risk_free_rate = 0.04
            var_confidence = 0.95
        
        analyze_button = st.button("🔍 Analyze Security", type="primary", width="stretch")
    
    # Main content
    if analyze_button and security_input:
        with st.spinner("Searching for security..."):
            # Try to find the ticker (handles company names too)
            temp_analyzer = SecurityAnalyzer(security_input, benchmark)
            resolved_ticker, name_or_status = temp_analyzer.search_ticker(security_input)
            
            # If search_ticker returns None, just try using the input as-is
            if not resolved_ticker:
                resolved_ticker = security_input.upper()
                name_or_status = resolved_ticker
            
            # Always try to analyze - let fetch_data determine if it's valid
            if resolved_ticker:
                # Show what we're analyzing
                if name_or_status and name_or_status != "name_detected":
                    st.info(f"Analyzing: {name_or_status} ({resolved_ticker})")
                else:
                    st.info(f"Analyzing: {resolved_ticker}")
                
                # Check if benchmark is appropriate for this ticker
                temp_analyzer_check = SecurityAnalyzer(resolved_ticker, benchmark)
                suggested_benchmark, suggestion_msg = temp_analyzer_check.get_appropriate_benchmark()
                is_compatible, warning_msg = temp_analyzer_check.check_benchmark_compatibility()
                
                # Show benchmark suggestion if needed
                if not is_compatible and suggested_benchmark:
                    st.warning(f"""
                    {warning_msg}
                    
                    **Recommended Action**: Consider changing benchmark to **{suggested_benchmark}** for more accurate correlation and beta calculations.
                    
                    💡 **Why?** Securities from different markets/exchanges often don't have overlapping trading days, which results in unreliable metrics (NaN values for Beta, correlation, etc.)
                    """)
                    
                                
                # Check if we need to force analysis with new benchmark
                if 'force_analyze' in st.session_state and st.session_state.force_analyze:
                    # Override benchmark with forced value
                    benchmark = st.session_state.force_benchmark
                    resolved_ticker = st.session_state.force_ticker
                    # Clear flags
                    st.session_state.force_analyze = False
                    st.info(f"🔄 Re-analyzing {resolved_ticker} with recommended benchmark: {benchmark}")
                
                with st.spinner("Fetching and analyzing data..."):
                    # Initialize analyzer with resolved ticker
                    analyzer = SecurityAnalyzer(resolved_ticker, benchmark)
                    
                    # Fetch data
                    if selected_period:
                        success = analyzer.fetch_data(period=selected_period)
                    else:
                        success = analyzer.fetch_data(start_date=start_date, end_date=end_date)
                    
                    if success and len(analyzer.data) > 0:
                        # Store in session state
                        st.session_state.analyzer = analyzer
                        st.session_state.analyzed_data = {
                            'risk_free_rate': risk_free_rate,
                            'var_confidence': var_confidence,
                            'currency_symbol': None,
                            'currency_name': None
                        }
                    else:
                        st.error(f"""
                        ❌ Could not find ticker '{resolved_ticker}'
                        
                        **Please check:**
                        - Ticker symbol is correct (e.g., AAPL, MSFT, GOOGL)
                        - For Indian stocks, add .NS or .BO (e.g., RELIANCE.NS, TCS.BO)
                        - For international stocks, check the exchange suffix
                        
                        **Common formats:**
                        - US stocks: AAPL, TSLA, NVDA
                        - Indian stocks: RELIANCE.NS, TCS.BO, INFY.NS
                        - UK stocks: HSBA.L, BP.L
                        - Japanese stocks: 7203.T, 9984.T
                        """)
    
    # Display results from session state (persists across reruns)
    if st.session_state.analyzer is not None:
        analyzer = st.session_state.analyzer
        
        # Calculate metrics
        analyzer.calculate_moving_averages()
        analyzer.calculate_returns()
        
        # Get currency symbol for this security
        currency_symbol = analyzer.get_currency_symbol()
        currency_name = analyzer.get_currency_name()
        
        # Get parameters from session state
        risk_free_rate = st.session_state.analyzed_data.get('risk_free_rate', 0.04)
        var_confidence = st.session_state.analyzed_data.get('var_confidence', 0.95)
        
        # Display security information
        st.success(f"✅ Successfully loaded data for {analyzer.ticker}")
        
        if analyzer.info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Company", analyzer.info.get('shortName', analyzer.ticker))
            with col2:
                st.metric("Sector", analyzer.info.get('sector', 'N/A'))
            with col3:
                st.metric("Industry", analyzer.info.get('industry', 'N/A'))
            with col4:
                st.metric("Currency", currency_name)
        
        st.markdown("---")
        
        # Price overview
        st.subheader("💰 Price Overview")
        metrics_52w = analyzer.get_52week_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Current Price",
                format_number(metrics_52w['Current_Price'], is_currency=True, currency_symbol=currency_symbol)
            )
        with col2:
            st.metric(
                "52-Week High",
                format_number(metrics_52w['52W_High'], is_currency=True, currency_symbol=currency_symbol)
            )
        with col3:
            st.metric(
                "52-Week Low",
                format_number(metrics_52w['52W_Low'], is_currency=True, currency_symbol=currency_symbol)
            )
        with col4:
            pct_from_high = ((metrics_52w['Current_Price'] - metrics_52w['52W_High']) / 
                            metrics_52w['52W_High'] * 100)
            st.metric(
                "% From 52W High",
                f"{pct_from_high:.2f}%"
            )
        
        # Price chart
        st.plotly_chart(create_price_chart(analyzer), width="stretch")
        
        st.markdown("---")
        
        # Key risk metrics
        st.subheader("📈 Risk & Performance Metrics")
        
        beta = analyzer.calculate_beta()
        
        # Display benchmark warning if there are issues
        if analyzer.benchmark_warning:
            st.error(analyzer.benchmark_warning)
            
            # Suggest appropriate benchmark
            suggested_benchmark, suggestion_msg = analyzer.get_appropriate_benchmark()
            if suggested_benchmark and suggested_benchmark != analyzer.benchmark:
                st.info(f"""
                💡 **Suggestion**: {suggestion_msg}
                
                Go back and change your benchmark to **{suggested_benchmark}** for accurate metrics, or use the sidebar to re-analyze with the correct benchmark.
                """)
        
        alpha = analyzer.calculate_alpha(risk_free_rate)
        sharpe = analyzer.calculate_sharpe_ratio(risk_free_rate)
        sortino = analyzer.calculate_sortino_ratio(risk_free_rate)
        volatility = analyzer.calculate_volatility()
        max_dd = analyzer.calculate_max_drawdown()
        correlation = analyzer.calculate_correlation()
        annual_return = analyzer.data['Daily_Return'].mean() * 252
        
        if st.session_state.analysis_mode == "Basic":
            # Basic Mode: Show essential metrics only
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                beta_display = f"{beta:.3f}" if not np.isnan(beta) else "N/A"
                st.metric("Beta", beta_display, 
                        help="Volatility vs market. <1 = less volatile, >1 = more volatile")
            
            with col2:
                st.metric("Annualized Return", format_number(annual_return, is_percentage=True),
                        help="Average yearly return")
            
            with col3:
                st.metric("Volatility", format_number(volatility, is_percentage=True),
                        help="Risk measure - higher = more price swings")
            
            with col4:
                st.metric("Max Drawdown", format_number(max_dd, is_percentage=True),
                        help="Worst peak-to-trough decline")
        
        else:
            # Advanced Mode: Show all metrics
            info_ratio = analyzer.calculate_information_ratio()
            calmar = analyzer.calculate_calmar_ratio()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                beta_display = f"{beta:.3f}" if not np.isnan(beta) else "N/A"
                st.metric("Beta", beta_display)
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            
            with col2:
                alpha_display = format_number(alpha, is_percentage=True) if not np.isnan(alpha) else "N/A"
                st.metric("Alpha", alpha_display)
                st.metric("Sortino Ratio", f"{sortino:.3f}")
            
            with col3:
                st.metric("Volatility", format_number(volatility, is_percentage=True))
                info_display = f"{info_ratio:.3f}" if not np.isnan(info_ratio) else "N/A"
                st.metric("Information Ratio", info_display)
            
            with col4:
                st.metric("Max Drawdown", format_number(max_dd, is_percentage=True))
                calmar_display = f"{calmar:.3f}" if not np.isnan(calmar) else "N/A"
                st.metric("Calmar Ratio", calmar_display)
            
            with col5:
                corr_display = f"{correlation:.3f}" if not np.isnan(correlation) else "N/A"
                st.metric("Correlation w/ Benchmark", corr_display)
                st.metric("Annualized Return", format_number(annual_return, is_percentage=True))
        
        st.markdown("---")
        
        # Value at Risk (Advanced mode only)
        if st.session_state.analysis_mode == "Advanced":
            st.subheader("⚠️ Value at Risk (VaR) Analysis")
            
            var_metrics = analyzer.calculate_var(var_confidence)
            cvar = analyzer.calculate_cvar(var_confidence)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Historical VaR ({int(var_confidence*100)}%)",
                    format_number(var_metrics['Historical_VaR'], is_percentage=True),
                    help="Maximum expected loss over one day at given confidence level"
                )
            
            with col2:
                st.metric(
                    f"Parametric VaR ({int(var_confidence*100)}%)",
                    format_number(var_metrics['Parametric_VaR'], is_percentage=True),
                    help="VaR assuming normal distribution of returns"
                )
            
            with col3:
                st.metric(
                    f"CVaR / Expected Shortfall ({int(var_confidence*100)}%)",
                    format_number(cvar, is_percentage=True),
                    help="Average loss in worst-case scenarios beyond VaR"
                )
            
            st.markdown("---")
        
        # Charts - show cumulative returns for all, other charts only in advanced
        if st.session_state.analysis_mode == "Basic":
            st.plotly_chart(create_returns_chart(analyzer), width="stretch")
        else:
            # Advanced mode: show all charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_returns_chart(analyzer), width="stretch")
            
            with col2:
                st.plotly_chart(create_drawdown_chart(analyzer), width="stretch")
            
            # Distribution chart
            st.plotly_chart(create_returns_distribution(analyzer), width="stretch")
        
        st.markdown("---")
        
        # Fundamental metrics
        st.subheader("💼 Fundamental Metrics")
        
        fundamentals = analyzer.get_fundamental_metrics()
        
        if fundamentals:
            if st.session_state.analysis_mode == "Basic":
                # Basic mode: Show key fundamentals only
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Market Cap", 
                            format_number(fundamentals.get('Market_Cap'), is_currency=True, currency_symbol=currency_symbol),
                            help="Company size")
                
                with col2:
                    st.metric("P/E Ratio", 
                            format_number(fundamentals.get('PE_Ratio')),
                            help="Price to Earnings - valuation metric")
                
                with col3:
                    st.metric("Dividend Yield", 
                            format_number(fundamentals.get('Dividend_Yield'), is_percentage=True),
                            help="Annual dividend as % of price")
                
                with col4:
                    st.metric("ROE", 
                            format_number(fundamentals.get('ROE'), is_percentage=True),
                            help="Return on Equity - profitability")
            
            else:
                # Advanced mode: Show all fundamentals
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Market Cap", 
                            format_number(fundamentals.get('Market_Cap'), is_currency=True, currency_symbol=currency_symbol))
                    st.metric("P/E Ratio", 
                            format_number(fundamentals.get('PE_Ratio')))
                    st.metric("PEG Ratio", 
                            format_number(fundamentals.get('PEG_Ratio')))
                
                with col2:
                    st.metric("Price to Book", 
                            format_number(fundamentals.get('Price_to_Book')))
                    st.metric("Dividend Yield", 
                            format_number(fundamentals.get('Dividend_Yield'), is_percentage=True))
                    st.metric("Profit Margin", 
                            format_number(fundamentals.get('Profit_Margin'), is_percentage=True))
                
                with col3:
                    st.metric("ROE", 
                            format_number(fundamentals.get('ROE'), is_percentage=True))
                    st.metric("ROA", 
                            format_number(fundamentals.get('ROA'), is_percentage=True))
                    st.metric("Current Ratio", 
                            format_number(fundamentals.get('Current_Ratio')))
                
                with col4:
                    st.metric("Debt to Equity", 
                            format_number(fundamentals.get('Debt_to_Equity')))
                    st.metric("Revenue Growth", 
                            format_number(fundamentals.get('Revenue_Growth'), is_percentage=True))
                    st.metric("Earnings Growth", 
                            format_number(fundamentals.get('Earnings_Growth'), is_percentage=True))
        
        # Performance summary table (Advanced mode only)
        if st.session_state.analysis_mode == "Advanced":
            st.markdown("---")
            st.subheader("📊 Performance Summary")
            
            returns_data = analyzer.data['Daily_Return'].dropna()
            
            summary_data = {
                'Metric': [
                    'Total Return', 'Annualized Return', 'Annualized Volatility',
                    'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown',
                    'Calmar Ratio', 'Beta', 'Alpha', 'Correlation',
                    'Skewness', 'Kurtosis', 'Best Day', 'Worst Day'
                ],
                'Value': [
                    format_number(analyzer.data['Cumulative_Return'].iloc[-1], is_percentage=True),
                    format_number(returns_data.mean() * 252, is_percentage=True),
                    format_number(volatility, is_percentage=True),
                    f"{sharpe:.3f}",
                    f"{sortino:.3f}",
                    format_number(max_dd, is_percentage=True),
                    f"{calmar:.3f}",
                    f"{beta:.3f}",
                    format_number(alpha, is_percentage=True),
                    f"{correlation:.3f}",
                    f"{returns_data.skew():.3f}",
                    f"{returns_data.kurtosis():.3f}",
                    format_number(returns_data.max(), is_percentage=True),
                    format_number(returns_data.min(), is_percentage=True)
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, width="stretch", hide_index=True)
    
    elif analyze_button and not security_input:
        st.warning("⚠️ Please enter a security symbol or name.")
    
    elif st.session_state.analyzer is None:
        # Welcome screen
        st.info("""
        ### 👋 Welcome to the Security Performance & Risk Analysis Dashboard
        
        This comprehensive tool provides institutional-grade analysis for equity securities.
        
        **📊 Two Analysis Modes:**
        - **Basic Mode**: Essential metrics for quick analysis (perfect for beginners)
        - **Advanced Mode**: Comprehensive analysis with all metrics (for experienced analysts)
        
        **Features Include:**
        - **Price Analysis**: Historical prices with 50, 100, and 200-day moving averages
        - **Risk Metrics**: Beta, volatility, Sharpe Ratio, and returns
        - **Value at Risk**: VaR and CVaR analysis (Advanced mode)
        - **Performance Analytics**: Returns vs benchmark, correlation, and drawdown analysis
        - **Fundamental Metrics**: Valuation ratios, profitability, and growth metrics
        - **Global Support**: Works with stocks from US, India (NSE/BSE), UK, Japan, and more!
        
        **Getting Started:**
        1. Choose **Basic** or **Advanced** mode in the sidebar
        2. Enter a security symbol or name (e.g., AAPL, RELIANCE.NS, ^NSEI)
        3. Select your benchmark and analysis period
        4. Click "Analyze Security" to generate analysis
        
        **Tips:**
        - Start with Basic mode if you're new to investing
        - Switch to Advanced mode anytime for detailed metrics
        - The dashboard remembers your analysis - change modes without re-analyzing!
        - Use NSE stocks with .NS suffix (e.g., RELIANCE.NS, TCS.NS)
        - Use BSE stocks with .BO suffix (e.g., RELIANCE.BO)
        """)


if __name__ == "__main__":
    main()
