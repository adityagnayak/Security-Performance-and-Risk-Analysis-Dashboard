# Security-Analysis-Dashboard
This repository hosts all the program files and documentation for a Security Analysis Dashboard
# Security Performance & Risk Analysis Dashboard

A professional, institutional-grade financial analysis tool built with Python and Streamlit for comprehensive equity security analysis and risk assessment.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

### Price Analysis
- **Historical Price Charts**: Interactive candlestick and line charts with volume
- **Moving Averages**: 50-day, 100-day, and 200-day moving averages
- **52-Week Metrics**: High, low, and current price positioning

### Risk Metrics
- **Beta**: Systematic risk relative to benchmark
- **Alpha**: Jensen's Alpha for risk-adjusted performance
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Information Ratio**: Active return per unit of tracking error
- **Calmar Ratio**: Return over maximum drawdown

### Value at Risk (VaR)
- **Historical VaR**: Based on actual return distribution
- **Parametric VaR**: Assumes normal distribution
- **CVaR (Expected Shortfall)**: Average loss in worst-case scenarios
- **Customizable Confidence Levels**: 90%, 95%, 99%

### Performance Analytics
- **Returns Analysis**: Daily, cumulative, and annualized returns
- **Volatility Metrics**: Annualized volatility and tracking error
- **Drawdown Analysis**: Maximum drawdown and recovery periods
- **Correlation Analysis**: Correlation with benchmark indices
- **Distribution Analysis**: Return distribution with normality tests

### Fundamental Metrics
- **Valuation Ratios**: P/E, PEG, Price-to-Book, Forward P/E
- **Profitability**: ROE, ROA, Profit Margin
- **Financial Health**: Debt-to-Equity, Current Ratio
- **Growth Metrics**: Revenue Growth, Earnings Growth
- **Dividend Metrics**: Dividend Yield

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/security-analysis-dashboard.git
cd security-analysis-dashboard
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run security_analysis_dashboard.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

## 📊 Usage Guide

### Basic Workflow

1. **Enter Security Symbol**
   - Input either a ticker symbol (e.g., `AAPL`, `MSFT`, `SPY`)
   - Or company name (e.g., `Apple Inc.`, `Microsoft`)
   - The tool automatically resolves names to ticker symbols

2. **Select Benchmark**
   - Choose from major indices: SPY, QQQ, DIA, IWM
   - Or market indices: ^GSPC, ^DJI, ^IXIC

3. **Choose Analysis Period**
   - **Preset**: 1Y, 3Y, 5Y, 10Y, or Max
   - **Custom**: Select specific start and end dates

4. **Configure Parameters**
   - Set risk-free rate for Sharpe/Sortino calculations (default: 4%)
   - Adjust VaR confidence level (90-99%, default: 95%)

5. **Analyze**
   - Click "Analyze Security" to generate comprehensive analysis

### Example Queries

**Individual Stocks:**
- `AAPL` or `Apple Inc.`
- `MSFT` or `Microsoft`
- `GOOGL` or `Alphabet`

**ETFs:**
- `SPY` or `S&P 500 ETF`
- `QQQ` or `Nasdaq 100 ETF`
- `VOO` or `Vanguard S&P 500`

**Indices:**
- `^GSPC` (S&P 500 Index)
- `^DJI` (Dow Jones Industrial Average)
- `^IXIC` (NASDAQ Composite)

## 📈 Metrics Explained

### Beta (β)
Measures systematic risk. 
- β > 1: More volatile than market
- β < 1: Less volatile than market
- β = 1: Moves with market

### Alpha (α)
Risk-adjusted outperformance vs. benchmark
- α > 0: Outperforming after risk adjustment
- α < 0: Underperforming after risk adjustment

### Sharpe Ratio
Return per unit of total risk
- Higher is better
- > 1: Good, > 2: Very good, > 3: Excellent

### Sortino Ratio
Return per unit of downside risk
- Similar to Sharpe but only considers downside volatility
- Better measure for asymmetric returns

### Value at Risk (VaR)
Maximum expected loss at confidence level
- 95% VaR of -2%: 95% chance loss won't exceed 2%

### CVaR (Conditional VaR)
Average loss when VaR threshold is breached
- More conservative than VaR
- Captures tail risk better

### Maximum Drawdown
Largest peak-to-trough decline
- Indicates worst historical loss period
- Key metric for risk tolerance

## 🛠️ Technical Architecture

### Data Sources
- **Yahoo Finance API** (via yfinance): Real-time and historical data
- Fetches OHLCV data, fundamental metrics, and corporate actions

### Key Libraries
- **Streamlit**: Interactive web dashboard
- **yfinance**: Financial data retrieval
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **SciPy**: Statistical functions

### Calculations

**Beta Calculation:**
```
β = Cov(R_stock, R_market) / Var(R_market)
```

**Alpha Calculation (Jensen's Alpha):**
```
α = R_stock - [R_f + β * (R_market - R_f)]
```

**Sharpe Ratio:**
```
Sharpe = (R_p - R_f) / σ_p
```

**VaR (Historical):**
```
VaR = Percentile(returns, 1 - confidence_level)
```

**CVaR:**
```
CVaR = E[returns | returns ≤ VaR]
```

## 🎨 Customization

### Modifying Benchmarks
Edit the benchmark list in `security_analysis_dashboard.py`:
```python
benchmark = st.selectbox(
    "Benchmark Index",
    ["SPY", "QQQ", "DIA", "IWM", "YOUR_CUSTOM_BENCHMARK"],
    help="Select benchmark for comparative analysis"
)
```

### Adding Custom Metrics
Extend the `SecurityAnalyzer` class:
```python
def calculate_custom_metric(self):
    # Your calculation logic
    return metric_value
```

### Styling
Modify the CSS in the `st.markdown()` section to customize appearance.

## 📦 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

### Heroku

1. Create `Procfile`:
```
web: streamlit run security_analysis_dashboard.py --server.port=$PORT
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Docker

1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY security_analysis_dashboard.py .

EXPOSE 8501

CMD ["streamlit", "run", "security_analysis_dashboard.py"]
```

2. Build and run:
```bash
docker build -t security-dashboard .
docker run -p 8501:8501 security-dashboard
```

## 🔧 Troubleshooting

### Common Issues

**Issue: "No data found" error**
- Check ticker symbol is correct
- Verify security has sufficient trading history
- Try using ticker symbol instead of company name

**Issue: Missing dependencies**
```bash
pip install --upgrade -r requirements.txt
```

**Issue: Slow loading**
- Reduce analysis period
- Check internet connection
- Yahoo Finance API may be rate-limited

**Issue: Chart not displaying**
- Clear browser cache
- Restart Streamlit server
- Update plotly: `pip install --upgrade plotly`

## 📝 Best Practices

### For Accurate Analysis
1. Use at least 1 year of data for meaningful statistics
2. Compare similar securities (stocks vs stocks, ETFs vs ETFs)
3. Consider market conditions during analysis period
4. Review multiple timeframes for comprehensive view

### Performance Optimization
1. Cache data using `@st.cache_data` for repeated analyses
2. Limit historical data range for faster loading
3. Use preset periods instead of custom dates when possible

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
git clone https://github.com/yourusername/security-analysis-dashboard.git
cd security-analysis-dashboard
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Guidelines
- Follow PEP 8 style guide
- Add docstrings to new functions
- Update README for new features
- Test thoroughly before submitting PR

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It should not be considered financial advice. Always conduct your own research and consult with qualified financial advisors before making investment decisions.

Past performance does not guarantee future results. All investments carry risk, including the potential loss of principal.

## 🙏 Acknowledgments

- Data provided by Yahoo Finance
- Built with Streamlit
- Inspired by quantitative finance best practices

## 📧 Contact

For questions, suggestions, or issues:
- Open an issue on GitHub
- Email: your.email@example.com

## 🗺️ Roadmap

- [ ] Add Monte Carlo simulation for portfolio risk
- [ ] Implement options Greeks calculator
- [ ] Add earnings analysis and forecasting
- [ ] Include sentiment analysis from news
- [ ] Multi-security portfolio analysis
- [ ] Export reports to PDF
- [ ] Real-time streaming data
- [ ] Machine learning price predictions
- [ ] ESG metrics integration

---

**Built with ❤️ for financial analysis and investment research**
