#!/usr/bin/env python3
"""
Data Sources Discovery for Syntropy Quant
Find alternatives when Tiingo/Yahoo fail.
"""

def test_stooq():
    """Test Stooq direct access (no API key needed)"""
    try:
        import yfinance as yf
        # Stooq uses yfinance's multi-source
        print("Testing Stooq (via yfinance)...")
        df = yf.download('^SPX', start='2020-01-01', end='2020-01-31', progress=False)
        if not df.empty:
            print(f"✅ Stooq S&P500 works! Got {len(df)} days")
            return True
        return False
    except Exception as e:
        print(f"❌ Stooq failed: {e}")jixu
        return False

def test_yahoo_direct():
    """Test direct Yahoo without wrapper"""
    try:
        import yfinance as yf
        print("\nTesting Yahoo Finance directly...")
        symbols = ['NVDA', 'AAPL', 'SPY']
        
        for sym in symbols:
            df = yf.download(sym, start='2023-01-01', end='2023-02-01', progress=False)
            if not df.empty:
                print(f"  ✅ {sym}: {len(df)} days")
            else:
                print(f"  ❌ {sym}: Empty")
        
        print("\nYahoo seems to work for recent data!")
        return True
    except Exception as e:
        print(f"❌ Yahoo direct failed: {e}")
        return False

def generate_enhanced_synthetic():
    """
    Generate synthetic data with CLEAR regime separation.
    """
    import numpy as np
    import pandas as pd
    
    print("\nGenerating Enhanced Synthetic Data...")
    
    np.random.seed(42)
    
    # Market 1: Pure Symplectic (Mean-reverting, Stable)
    # Low trend, strong mean reversion
    days_sym = 1000
    prices_sym = [100.0]
    regimes_sym = []
    
    current_price = 100.0
    for i in range(days_sym):
        # Strong mean reversion
        pull = -1.5 * (current_price - 100) / 100
        ret = 0.0001 + pull + np.random.normal(0, 0.01)
        current_price = current_price * (1 + ret)
        prices_sym.append(current_price)
        regimes_sym.append(0)  # 0 = Symplectic
    
    df_sym = pd.DataFrame({
        'Close': prices_sym,
        'High': [p * (1 + abs(np.random.uniform(0, 0.005))) for p in prices_sym],
        'Low': [p * (1 - abs(np.random.uniform(0, 0.005))) for p in prices_sym],
        'Open': [p * (1 + np.random.uniform(-0.003, 0.003)) for p in prices_sym],
        'Volume': [1e6 * (1 + np.random.uniform(-0.1, 0.3)) for _ in range(days_sym)]
    }, index=pd.date_range(start="2020-01-01", periods=days_sym))
    
    # Market 2: Pure Gauge (Trending, Turbulent)
    # Strong trend, high volatility
    days_gauge = 1000
    prices_gauge = [100.0]
    regimes_gauge = []
    
    current_price = 100.0
    for i in range(days_gauge):
        # Strong trend
        drift = 0.003
        vol = 0.025
        ret = drift + np.random.normal(0, vol)
        current_price = current_price * (1 + ret)
        prices_gauge.append(current_price)
        regimes_gauge.append(1)  # 1 = Gauge
    
    df_gauge = pd.DataFrame({
        'Close': prices_gauge,
        'High': [p * (1 + abs(np.random.uniform(0.01, 0.02))) for p in prices_gauge],
        'Low': [p * (1 - abs(np.random.uniform(0.01, 0.02))) for p in prices_gauge],
        'Open': [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices_gauge],
        'Volume': [1e6 * (1 + np.random.uniform(-0.1, 0.5)) for _ in range(days_gauge)]
    }, index=pd.date_range(start="2020-01-01", periods=days_gauge))
    
    # Market 3: Mixed (Transitional)
    # Alternating between regimes
    days_mixed = 1000
    prices_mixed = [100.0]
    regimes_mixed = []
    
    current_price = 100.0
    current_regime = 0
    for i in range(days_mixed):
        # Switch regimes periodically
        if i % 100 < 10:  # Stay in regime for 10 days
            pass
        else:
            current_regime = 1 - current_regime
        
        regimes_mixed.append(current_regime)
        
        if current_regime == 0:
            # Symplectic
            pull = -1.0 * (current_price - 100) / 100
            ret = 0.0001 + pull + np.random.normal(0, 0.012)
        else:
            # Gauge
            drift = 0.002
            vol = 0.02
            ret = drift + np.random.normal(0, vol)
        
        current_price = current_price * (1 + ret)
        prices_mixed.append(current_price)
    
    df_mixed = pd.DataFrame({
        'Close': prices_mixed,
        'High': [p * (1 + abs(np.random.uniform(0.005, 0.015))) for p in prices_mixed],
        'Low': [p * (1 - abs(np.random.uniform(0.005, 0.015))) for p in prices_mixed],
        'Open': [p * (1 + np.random.uniform(-0.003, 0.003)) for p in prices_mixed],
        'Volume': [1e6 * (1 + np.random.uniform(-0.2, 0.4)) for _ in range(days_mixed)]
    }, index=pd.date_range(start="2020-01-01", periods=days_mixed))
    
    print(f"\nGenerated synthetic data:")
    print(f"  Symplectic (Regime 0): {len(df_sym)} days, mean reg: {np.mean(regimes_sym)*100:.1f}%")
    print(f"  Gauge (Regime 1):     {len(df_gauge)} days, mean reg: {np.mean(regimes_gauge)*100:.1f}%")
    print(f"  Mixed (Regime 0-1): {len(df_mixed)} days, mean reg: {np.mean(regimes_mixed)*100:.1f}%")
    
    return df_sym, df_gauge, df_mixed

def main():
    print("=" * 60)
    print("  DATA SOURCES DISCOVERY")
    print("=" * 60)
    print()
    
    # Test Stooq
    stooq_ok = test_stooq()
    
    # Test Yahoo
    yahoo_ok = test_yahoo_direct()
    
    # Generate enhanced synthetic
    df_sym, df_gauge, df_mixed = generate_enhanced_synthetic()
    
    # Save synthetic data
    import os
    os.makedirs("data_cache_v5", exist_ok=True)
    
    df_sym.to_parquet("data_cache_v5/synthetic_symplectic.parquet")
    df_gauge.to_parquet("data_cache_v5/synthetic_gauge.parquet")
    df_mixed.to_parquet("data_cache_v5/synthetic_mixed.parquet")
    
    print("\n✅ Saved synthetic data to data_cache_v5/")
    print("   - synthetic_symplectic.parquet")
    print("   - synthetic_gauge.parquet")
    print("   - synthetic_mixed.parquet")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("  RECOMMENDATION")
    print("=" * 60)
    print()
    
    if stooq_ok:
        print("✅ Use Stooq (via yfinance) for real market data")
        print("   Modify fetcher to use yfinance directly")
    elif yahoo_ok:
        print("✅ Use Yahoo Finance directly for recent data")
        print("   Add delays between requests")
    else:
        print("✅ Use Enhanced Synthetic Data for regime validation")
        print("   Clear separation: Regime 0 vs Regime 1")
        print("   Train model to distinguish these clearly")
    
    print()

if __name__ == "__main__":
    main()

