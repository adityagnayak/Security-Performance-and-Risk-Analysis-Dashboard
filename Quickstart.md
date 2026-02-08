# Quick Start Guide
# Security Performance & Risk Analysis Dashboard

## Installation (5 minutes)

### Option 1: Automated Setup (Recommended)

**On macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```cmd
setup.bat
```

### Option 2: Manual Setup

1. **Create virtual environment:**
```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

2. **Install packages:**
```bash
pip install -r requirements.txt
```

3. **Test installation:**
```bash
python test_dashboard.py
```

## Running the Dashboard

```bash
streamlit run security_analysis_dashboard.py
```

The dashboard will open at: http://localhost:8501

## First Analysis (2 minutes)

### Example 1: Analyze Apple Stock

1. **Enter Security:** `AAPL` or `Apple Inc.`
2. **Select Benchmark:** `SPY` (S&P 500)
3. **Choose Period:** `5Y` (5 years)
4. **Click:** "🔍 Analyze Security"

**What you'll see:**
- Price chart with moving averages (50, 100, 200-day)
- 52-week high/low metrics
- Risk metrics (Beta, Alpha, Sharpe, Sortino)
- Value at Risk (VaR) analysis
- Returns comparison vs benchmark
- Drawdown analysis
- Fundamental metrics

### Example 2: Compare ETF to Market

1. **Enter Security:** `QQQ` (Nasdaq 100 ETF)
2. **Select Benchmark:** `SPY` (S&P 500)
3. **Choose Period:** `3Y`
4. **Adjust Risk-Free Rate:** `4.0%`
5. **Analyze**

**Analysis Tips:**
- Lower Beta (<1) = Less volatile than market
- Higher Sharpe Ratio (>1) = Better risk-adjusted returns
- Correlation closer to 1 = Moves with benchmark
- Max Drawdown shows worst decline

### Example 3: Custom Date Range

1. **Enter Security:** `MSFT`
2. **Period Type:** Select "Custom"
3. **Start Date:** 2020-01-01
4. **End Date:** 2024-12-31
5. **VaR Confidence:** 99%
6. **Analyze**

## Understanding Key Metrics

### Risk Metrics
- **Beta**: Volatility vs market (1.0 = market volatility)
- **Alpha**: Excess return after risk adjustment
- **Sharpe Ratio**: Return per unit of risk (>1 = good)
- **Sortino Ratio**: Like Sharpe, but only downside risk
- **Max Drawdown**: Largest peak-to-trough decline

### Value at Risk (VaR)
- **95% VaR of -2%**: 95% chance daily loss won't exceed 2%
- **CVaR**: Average loss when VaR is breached
- Use for risk budgeting and position sizing

### Performance Metrics
- **Volatility**: Annualized standard deviation
- **Correlation**: Relationship with benchmark (-1 to 1)
- **Information Ratio**: Active return per tracking error

## Common Use Cases

### 1. Stock Screening
```
Compare multiple stocks:
- Analyze AAPL with SPY benchmark
- Analyze MSFT with SPY benchmark
- Compare Beta, Sharpe, and Alpha
```

### 2. Portfolio Risk Assessment
```
Check portfolio holdings:
- Enter each holding
- Note Beta and correlation
- Sum weighted Betas for portfolio Beta
```

### 3. Entry/Exit Timing
```
Use technical indicators:
- Price vs 200-day MA (trend)
- Recent drawdown (opportunity?)
- Volatility levels (risk)
```

### 4. Competitive Analysis
```
Compare similar companies:
- AAPL vs MSFT (Tech)
- JPM vs BAC (Finance)
- XOM vs CVX (Energy)
```

## Advanced Features

### Custom Risk-Free Rate
Adjust based on current treasury yields:
- Check 10-year treasury rate
- Input in sidebar
- Affects Sharpe and Alpha calculations

### Multiple Time Periods
Analyze different horizons:
- **1Y**: Recent performance
- **3Y**: Medium-term trends
- **5Y**: Full market cycle
- **10Y**: Long-term analysis
- **Max**: Complete history

### VaR Confidence Levels
- **90%**: Less conservative
- **95%**: Standard (regulatory)
- **99%**: Very conservative

## Interpreting Results

### Good Signals
✅ Positive Alpha
✅ Sharpe Ratio > 1
✅ Low correlation if diversifying
✅ Recovering from drawdown
✅ Price above 200-day MA

### Warning Signals
⚠️ Negative Alpha
⚠️ High volatility vs returns
⚠️ Large drawdowns
⚠️ Price below MAs
⚠️ Declining fundamentals

## Export and Sharing

### Take Screenshots
- Use browser screenshot tool
- Capture specific charts
- Share analysis with team

### Save Analysis
- Bookmark security + period
- Document findings
- Track over time

## Troubleshooting

### "No data found"
- Check ticker symbol spelling
- Try official ticker (not company name)
- Verify security has trading history

### Slow loading
- Reduce date range
- Use preset periods
- Check internet connection

### Missing metrics
- Some stocks lack fundamental data
- ETFs may not have all metrics
- Use what's available

## Next Steps

1. **Test with known stocks** (AAPL, MSFT, GOOGL)
2. **Compare to benchmarks** (SPY, QQQ, DIA)
3. **Analyze your portfolio holdings**
4. **Explore different time periods**
5. **Experiment with settings**

## Getting Help

- 📖 Read full README.md
- 🐛 Check GitHub Issues
- 💬 Contact: your.email@example.com

## Keyboard Shortcuts

While in dashboard:
- `R` - Rerun analysis
- `C` - Clear cache
- `?` - Help menu
- `⌘/Ctrl + K` - Command palette

---

**Ready to start?**

```bash
streamlit run security_analysis_dashboard.py
```

Happy analyzing! 📊
