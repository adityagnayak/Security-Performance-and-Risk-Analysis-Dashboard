"""
Test Script for Security Analysis Dashboard
Run this to validate the installation and functionality
"""

import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    print("-" * 50)
    
    required_packages = {
        'streamlit': 'streamlit',
        'yfinance': 'yf',
        'pandas': 'pd',
        'numpy': 'np',
        'plotly': 'plotly',
        'scipy': 'scipy'
    }
    
    failed = []
    
    for package, import_name in required_packages.items():
        try:
            if import_name == package:
                importlib.import_module(package)
            else:
                exec(f"import {package} as {import_name}")
            print(f"✓ {package:15s} - OK")
        except ImportError as e:
            print(f"✗ {package:15s} - FAILED: {str(e)}")
            failed.append(package)
    
    print("-" * 50)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_data_fetch():
    """Test data fetching functionality"""
    print("\n\nTesting data fetch...")
    print("-" * 50)
    
    try:
        import yfinance as yf
        
        # Test with a reliable ticker
        print("Fetching AAPL data...")
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1mo")
        
        if len(data) > 0:
            print(f"✓ Successfully fetched {len(data)} days of data")
            print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            # Test info
            info = ticker.info
            if 'symbol' in info:
                print(f"✓ Company info retrieved: {info.get('shortName', 'N/A')}")
            
            print("-" * 50)
            print("✅ Data fetch test passed!")
            return True
        else:
            print("✗ No data retrieved")
            print("-" * 50)
            print("❌ Data fetch test failed!")
            return False
            
    except Exception as e:
        print(f"✗ Error during data fetch: {str(e)}")
        print("-" * 50)
        print("❌ Data fetch test failed!")
        return False


def test_calculations():
    """Test key calculations"""
    print("\n\nTesting calculations...")
    print("-" * 50)
    
    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        # Create sample data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        # Test volatility calculation
        volatility = returns.std() * np.sqrt(252)
        print(f"✓ Volatility calculation: {volatility:.4f}")
        
        # Test VaR
        var_95 = np.percentile(returns, 5)
        print(f"✓ VaR (95%) calculation: {var_95:.4f}")
        
        # Test cumulative returns
        cumulative = (1 + pd.Series(returns)).cumprod() - 1
        total_return = cumulative.iloc[-1]
        print(f"✓ Cumulative return: {total_return:.4f}")
        
        # Test drawdown
        cumulative_wealth = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative_wealth.expanding().max()
        drawdown = (cumulative_wealth - running_max) / running_max
        max_dd = drawdown.min()
        print(f"✓ Max drawdown: {max_dd:.4f}")
        
        print("-" * 50)
        print("✅ Calculation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during calculations: {str(e)}")
        print("-" * 50)
        print("❌ Calculation tests failed!")
        return False


def test_visualizations():
    """Test visualization creation"""
    print("\n\nTesting visualizations...")
    print("-" * 50)
    
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
        
        # Test chart creation
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'))
        
        print("✓ Successfully created Plotly chart")
        print(f"  Data points: {len(dates)}")
        
        print("-" * 50)
        print("✅ Visualization tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during visualization test: {str(e)}")
        print("-" * 50)
        print("❌ Visualization tests failed!")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("SECURITY ANALYSIS DASHBOARD - TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Fetching", test_data_fetch),
        ("Calculations", test_calculations),
        ("Visualizations", test_visualizations)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print("=" * 50)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Dashboard is ready to use.")
        print("\nTo run the dashboard:")
        print("  streamlit run security_analysis_dashboard.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("Install missing packages with: pip install -r requirements.txt")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
